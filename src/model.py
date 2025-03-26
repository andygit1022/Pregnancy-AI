# src/model.py
import numpy as np

##############################
# TensorFlow / Keras imports
##############################
import tensorflow as tf
from tensorflow import keras
import keras.src as K

#####################
# XGBoost / LightGBM
#####################
import xgboost as xgb
from xgboost.callback import EarlyStopping

import lightgbm as lgb


from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
#################
# CatBoost
#################
from catboost import CatBoostClassifier

#########################
# ElasticNet (Logistic)
#########################
from sklearn.linear_model import LogisticRegression






#########################
# Metrics
#########################
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score

from .PARAMS import (
    EPOCHS, BATCH_SIZE, SAVE_EPOCH,
    MLP_LEARNING_RATE, MLP_OPTIMIZER, MLP_HIDDEN_UNITS, MLP_DROPOUT,
    XGB_PARAMS_IVF, XGB_PARAMS_DI,
    SEED, XGB_EVAL_METRIC
)


#######################################
# 1) XGBoost 모델
#######################################
class XGBoostModel:
    def __init__(self, use_smote=False, ivf_di="ivf"):
        """
        use_smoteenn : True로 설정하면 fit()할 때 SMOTE+ENN 적용.
        """
        self.use_smote = use_smote

        # XGBoostClassifier 설정
        if ivf_di == "ivf":
            self.model = xgb.XGBClassifier(
                **XGB_PARAMS_IVF,
                eval_metric=XGB_EVAL_METRIC
            )
        else :
            self.model = xgb.XGBClassifier(
                **XGB_PARAMS_DI,
                eval_metric=XGB_EVAL_METRIC
            )
               
            
        self.best_score = 0.0
        self.best_iteration = 0
        self.best_model = None

    def fit(self, X_train, y_train, X_val, y_val):
        """
        XGBoost 모델 학습 및 Best 모델 저장 (Validation AUC 기준)
        """
        # ---------------------------
        # (1) SMOTE+ENN 적용 (선택)
        # ---------------------------
        # if self.use_smoteenn:
        #     from imblearn.combine import SMOTEENN
        #     sm = SMOTEENN(random_state=SEED)
        #     X_train, y_train = sm.fit_resample(X_train, y_train)
        #     print(f"[XGBoost] After SMOTEENN: X_train={X_train.shape}, y_train={y_train.shape}")

        if self.use_smote:
            print("[XGBoost] Applying SMOTE...")
            sm = SMOTE(sampling_strategy='auto',k_neighbors=5,random_state=SEED)  # SMOTE 적용
            X_train, y_train = sm.fit_resample(X_train, y_train)
            print(f"[XGBoost] After SMOTE: X_train={X_train.shape}, y_train={y_train.shape}")

        # ---------------------------
        # (2) 모델 학습 (Verbose=100)
        # ---------------------------
        
        
        
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            #early_stopping_rounds=10,  # Validation loss가 10번 동안 개선되지 않으면 종료
            verbose=100  # 매 100 라운드마다 AUC 로그 출력
        )

        # ---------------------------
        # (3) Best Iteration 선택
        # ---------------------------
        results = self.model.evals_result()
        val_auc_list = results["validation_1"]["auc"]  # Validation AUC 리스트

        # ✅ 최고 AUC를 기록한 index 찾기
        self.best_iteration = np.argmax(val_auc_list) + 1  # (index는 0부터 시작하므로 +1)

        # ✅ best_iteration 기반 모델 설정
        self.model.set_params(n_estimators=self.best_iteration)
        print(f"\n✅ [XGBoost] Best Iteration: {self.best_iteration}, Best Validation AUC: {max(val_auc_list):.4f}")

        # ✅ Best Score 저장
        self.best_score = max(val_auc_list)
        self.best_model = self.model.get_booster()

    def predict(self, X):
        return self.model.predict(X, iteration_range=(0, self.best_iteration))

    def predict_proba(self, X):
        return self.model.predict_proba(X, iteration_range=(0, self.best_iteration))[:, 1]


#######################################
# 2) LightGBM 모델
#######################################
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
import numpy as np

from sklearn.metrics import roc_auc_score
from collections import Counter
from .PARAMS import LGB_PARAMS_DI, LGB_PARAMS_IVF
import logging


class LightGBMModel:
    def __init__(self, use_smote=False, eval_metric="auc", ivf_di = "ivf", random_state=42):
        """
        use_smote : True 설정 시 fit()에서 SMOTE 오버샘플링 적용
        lgb_params: LightGBM 모델 파라미터(dict)
        eval_metric: 평가 지표 (e.g. 'auc', 'binary_logloss' 등)
        random_state: 난수 고정
        """
        self.use_smote = use_smote
        self.eval_metric = eval_metric
        
        if isinstance(eval_metric, str):
            self.eval_metric = [eval_metric]  # 단일 문자열이면 리스트로 변환
        else:
            self.eval_metric = eval_metric
        
        if ivf_di == "ivf":    
            self.model = lgb.LGBMClassifier(**LGB_PARAMS_IVF)
        else :
            self.model = lgb.LGBMClassifier(**LGB_PARAMS_DI)

        self.best_iteration = 0
        self.best_score = 0.0
        self.best_model = None
        
        logging.getLogger("lightgbm").setLevel(logging.CRITICAL)

    def fit(self, X_train, y_train, X_val, y_val):
        """
        LightGBM 모델을 학습하고 best_iteration, best_score를 기록.
        """
        # 1) SMOTE 적용(선택)
        if self.use_smote:
            print("[LightGBM] Applying SMOTE...")
            sm = SMOTE(random_state=42, sampling_strategy="auto")
            X_train, y_train = sm.fit_resample(X_train, y_train)
            print(f"[LightGBM] After SMOTE: X_train={X_train.shape}, y_train={Counter(y_train)}")

        # 2) 모델 학습
        #   - early_stopping_rounds: eval_metric 개선이 10번 정체되면 학습 중단
        #   - verbose: 100마다 로그
        
        cbs = [
            lgb.log_evaluation(period=100)  # 100 iteration마다 결과 로그
        ]
        
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_names=["valid"],
            eval_metric=self.eval_metric,
            # sample_weight=None, init_score=None,
            # eval_sample_weight=None, eval_class_weight=None,
            # feature_name="auto", categorical_feature="auto",
            callbacks=cbs, init_model=None,

        )

        # (3) 별도 early_stopping이 없으므로,
        #     직접 검증 세트 AUC를 구해 best_score 기록

        y_val_proba = self.model.predict_proba(X_val)[:, 1]
        auc_val = roc_auc_score(y_val, y_val_proba)
        self.best_score = auc_val

        # best_iteration_은 early_stopping 콜백을 쓰지 않으면 업데이트 안 될 수 있음
        self.best_iteration = self.model.best_iteration_
        # 그냥 참조용 -> 보통 0일 것
        self.best_model = self.model

        print(f"\n✅ [LightGBM] Final Model n_estimators={self.model.n_estimators_}, Validation {self.eval_metric}={self.best_score:.4f}")


    def predict(self, X):
        """
        LightGBM 이진 분류에서 라벨(0/1) 예측
        """
        return self.model.predict(X)

    def predict_proba(self, X):
        """
        LightGBM 이진 분류에서 양성 클래스 확률 예측
        """
        return self.model.predict_proba(X)[:, 1]


#######################################
# 3) CatBoost 모델
#######################################
class CatBoostModel:
    def __init__(self):
        """
        예시 파라미터:
        CatBoostClassifier(
            iterations=1000,
            learning_rate=0.1,
            eval_metric='AUC',
            random_seed=SEED,
            verbose=0
        )
        """
        self.model = CatBoostClassifier(
            iterations=1000,
            learning_rate=0.1,
            eval_metric='AUC',
            random_seed=SEED,
            verbose=0
        )
        self.best_score = 0.0
        self.best_model = None

    def fit(self, X_train, y_train, X_val, y_val):
        # catboost는 use_best_model=True로 설정하면 자동으로
        # best_iteration 시점의 모델을 기억
        self.model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            early_stopping_rounds=10,
            use_best_model=True
        )
        # get_best_score() => {'learn': {'AUC': ...}, 'validation': {'AUC': ...}}
        best_scores = self.model.get_best_score()
        if 'validation' in best_scores and 'AUC' in best_scores['validation']:
            self.best_score = best_scores['validation']['AUC']
        # 모델 자체가 best iteration 상태로 남아있음
        self.best_model = self.model

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]


##############################################
# 4) ElasticNet (로지스틱) - 분류 예시
##############################################
class ElasticNetModel:
    """
    sklearn LogisticRegression with penalty='elasticnet' for classification.
    """
    def __init__(self, l1_ratio=0.5, C=1.0):
        """
        l1_ratio: 0~1 사이에서 L1 가중치 비율
        C: 규제 강도 (1/alpha)
        """
        # penalty='elasticnet' 하려면 solver='saga' 필요
        self.model = LogisticRegression(
            penalty='elasticnet',
            solver='saga',
            l1_ratio=l1_ratio,
            C=C,
            max_iter=1000,
            random_state=SEED
        )
        self.best_score = 0.0
        self.best_model = None

    def fit(self, X_train, y_train, X_val, y_val):
        # 기본적으로 LogisticRegression은 fit 한 번에 수렴
        # "epoch" 개념이 없으므로 여기선 단순히 fit → val AUC 계산
        self.model.fit(X_train, y_train)

        # AUC 계산
        y_pred_prob = self.model.predict_proba(X_val)[:, 1]
        auc_val = roc_auc_score(y_val, y_pred_prob)
        self.best_score = auc_val
        # best_model = copy.deepcopy(self.model) 등으로 복사 가능
        self.best_model = self.model

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]


###########################
# 5) TensorFlow MLP 모델
###########################
class MLPModel:
    def __init__(self):
        self.learning_rate = MLP_LEARNING_RATE
        self.optimizer_name = MLP_OPTIMIZER
        self.hidden_units = MLP_HIDDEN_UNITS
        self.dropout_rate = MLP_DROPOUT
        self.epochs = EPOCHS
        self.batch_size = BATCH_SIZE

        self._build_model()
        self.best_score = 0.0
        self.best_weights = None

    def _build_model(self):
        inputs = K.Input(shape=(None,), name="features")
        x = inputs
        for units in self.hidden_units:
            x = K.layers.Dense(units, activation="relu")(x)
            x = K.layers.Dropout(self.dropout_rate)(x)
        outputs = K.layers.Dense(1, activation="sigmoid")(x)

        self.model = K.Model(inputs, outputs)

        if self.optimizer_name == "adam":
            optimizer = K.optimizers.Adam(learning_rate=self.learning_rate)
        elif self.optimizer_name == "sgd":
            optimizer = K.optimizers.SGD(learning_rate=self.learning_rate)
        else:
            optimizer = K.optimizers.Adam(learning_rate=self.learning_rate)

        self.model.compile(
            optimizer=optimizer,
            loss="binary_crossentropy",
            metrics=[]
        )

    def fit(self, X_train, y_train, X_val, y_val):
        # numpy array 변환
        X_train = np.asarray(X_train, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.float32)
        X_val = np.asarray(X_val, dtype=np.float32)
        y_val = np.asarray(y_val, dtype=np.float32)

        for epoch in range(1, self.epochs + 1):
            self.model.fit(
                X_train, y_train,
                epochs=1,
                batch_size=self.batch_size,
                validation_data=(X_val, y_val),
                verbose=0
            )
            # 매 epoch마다 AUC 계산
            y_pred_prob = self.model.predict(X_val).ravel()
            auc_val = roc_auc_score(y_val, y_pred_prob)

            print(f"[Epoch {epoch}] AUC on val: {auc_val:.4f}")

            # 최고기록 갱신 시점에 weights 저장
            if auc_val > self.best_score:
                self.best_score = auc_val
                self.best_weights = self.model.get_weights()

            # SAVE_EPOCH마다 중간 가중치 저장 (optional)
            if epoch % SAVE_EPOCH == 0:
                # 예시: results/weights/MLP_epoch{epoch}.h5 저장
                self.model.save_weights(f"results/weights/MLP_epoch{epoch}.h5")
                print(f"--- Saved weights at epoch {epoch} ---")

        # 전체 학습 후 best_weights로 복원
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        preds = self.model.predict(X).ravel()
        return (preds > 0.5).astype(int)

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        return self.model.predict(X).ravel()
