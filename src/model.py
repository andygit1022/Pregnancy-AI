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
    XGB_PARAMS, LGB_PARAMS,
    SEED, XGB_EVAL_METRIC
)


#######################################
# 1) XGBoost 모델
#######################################
class XGBoostModel:
    def __init__(self):
        """
        XGB_PARAMS는 PARAMS.py에서 예시로:
        XGB_PARAMS = {
            "n_estimators": 1000,
            "learning_rate": 0.1,
            "max_depth": 6,
            "random_state": 42,
            "eval_metric": "auc",
        }
        형태로 설정해둔다고 가정.
        """
        self.model = xgb.XGBClassifier(
            **XGB_PARAMS,
            eval_metric=XGB_EVAL_METRIC
        )
        self.best_score = 0.0
        self.best_iteration = 0
        self.best_model = None

    def fit(self, X_train, y_train, X_val, y_val):
        # XGBoost는 fit 한 번으로 자동 학습+early stopping 지원
        # iteration마다의 AUC 기록을 얻으려면 evals_result를 사용
        eval_set = [(X_train, y_train), (X_val, y_val)]
        
        early_stopping = EarlyStopping(
            rounds=10, 
            metric_name="auc", 
            data_name="validation_1",
            save_best=True
        )
        
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            #num_boost_round=XGB_PARAMS["n_estimators"],
            #early_stopping_rounds = 10,
            #callbacks=[early_stopping],  #  early_stopping을 callbacks로 전달
            verbose=True
        )
        
        
        
        # ✅ best_iteration 가져오기
        if hasattr(self.model, "best_ntree_limit"):
            self.best_iteration = self.model.best_ntree_limit
        else:
            self.best_iteration = self.model.n_estimators  # 기본값

        results = self.model.evals_result()
        train_auc = results['validation_0']['auc']
        val_auc = results['validation_1']['auc']

        print(f"Final Train AUC: {train_auc[-1]:.4f}, Validation AUC: {val_auc[-1]:.4f}")
        
        
        self.best_score = results['validation_1']['auc'][-1]  # 마지막 AUC 값 저장

        # ✅ best_model 저장
        self.best_model = self.model.get_booster()

    def predict(self, X):
        return self.model.predict(X, iteration_range=(0, self.best_iteration))

    def predict_proba(self, X):
        return self.model.predict_proba(X, iteration_range=(0, self.best_iteration))[:, 1]


#######################################
# 2) LightGBM 모델
#######################################
class LightGBMModel:
    def __init__(self):
        """
        LGB_PARAMS = {
            "n_estimators": 1000,
            "learning_rate": 0.1,
            "max_depth": -1,
            "random_state": 42
        }
        """
        self.model = lgb.LGBMClassifier(**LGB_PARAMS)
        self.best_score = 0.0
        self.best_iteration = 0
        self.best_model = None

    def fit(self, X_train, y_train, X_val, y_val):
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='auc',
            early_stopping_rounds=10,
            verbose=False
        )
        # LightGBM은 fit 후 model.best_iteration_ / model.best_score_ 존재
        self.best_iteration = self.model.best_iteration_
        # best_score_는 딕셔너리 형태
        # 예: {'training': {'auc': ...}, 'valid_1': {'auc': ...}}
        if hasattr(self.model, 'best_score_'):
            # valid_0 or valid_1 키 이름이 LightGBM 버전에 따라 다를 수 있음
            # 여기서는 valid_0일 수도 있고 valid_1일 수도 있음. 확인 필요
            best_dict = self.model.best_score_.get('valid_0', {})
            self.best_score = best_dict.get('auc', 0.0)

        # 필요한 경우, best_iteration까지의 모델 가중치를 별도로 저장
        # LightGBM은 내부적으로 best_iteration_ 사용하면 predict 시 자동으로 best만큼 사용
        # 별도 copy는 아래처럼 joblib 등으로 가능
        self.best_model = self.model  # 간단히 할당

    def predict(self, X):
        return self.model.predict(X, num_iteration=self.best_iteration)

    def predict_proba(self, X):
        return self.model.predict_proba(X, num_iteration=self.best_iteration)[:, 1]


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
