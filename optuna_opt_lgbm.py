import optuna
import lightgbm as lgb
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# 필요한 함수/상수들을 import
from src.preprocess import load_data, preprocess_data
from src.PARAMS import TRAIN_PATH, VALIDATION_SPLIT, SEED

OPTUNA_IVF_DI = 0   # 0 for di, 1 for ivf

def objective(trial):
    """
    Optuna가 반복 호출하는 최적화 대상 함수.
    trial: Optuna에서 각 하이퍼파라미터를 제안(샘플링)하는 객체
    """
    # 1) 탐색할 LightGBM 파라미터 공간 정의
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 150),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1e-8, 10.0, log=True),
        "objective": "binary",
        "metric": "auc",
        "random_state": SEED,
        "verbosity": -1,
        "early_stopping_rounds" : 100
    }

    # 2) IVF 데이터만 추출하여 전처리
    #    (DI를 하고 싶으면 "DI"로 교체)
    df_train = load_data(TRAIN_PATH)
    
    
    # OPTUNA_IVF_DI 0 for di, 1 for ivf
    if OPTUNA_IVF_DI == 1:    
        X_ivf, y_ivf = preprocess_data(df_train, is_train=True, procedure_type="IVF")
        X_train, X_val, y_train, y_val = train_test_split(
            X_ivf, y_ivf,
            test_size=VALIDATION_SPLIT,
            random_state=SEED,
            stratify=y_ivf
        )
    
    else :
        X_di, y_di = preprocess_data(df_train, is_train=True, procedure_type="DI")
        X_train, X_val, y_train, y_val = train_test_split(
            X_di, y_di,
            test_size=VALIDATION_SPLIT,
            random_state=SEED,
            stratify=y_di
        )
        

    # 3) LightGBM 모델 생성/학습
    model = lgb.LGBMClassifier(**params)
    
    callbacks = [
            lgb.early_stopping(100),  # 100번 동안 개선이 없으면 조기 중단
            lgb.log_evaluation(100)  # 100번째마다 로그 출력
    ]
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=callbacks
    )

    # 4) 검증 세트에 대해 ROC-AUC 계산
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    auc_val = roc_auc_score(y_val, y_pred_proba)

    return auc_val


if __name__ == "__main__":
    # Optuna를 통해 LightGBM 하이퍼파라미터 탐색
    study = optuna.create_study(direction="maximize")  # AUC 최대화
    study.optimize(objective, n_trials=200)             # 예: 100번 탐색

    # 결과 출력
    print("=====================================")
    print(f"Best AUC: {study.best_value:.4f}")
    print("Best Params:", study.best_params)

    # 최적 파라미터를 텍스트 파일에 저장
    with open("best_lgbm_params.txt", "w", encoding="utf-8") as f:
        f.write(f"Best AUC: {study.best_value:.4f}\n")
        for key, val in study.best_params.items():
            f.write(f"{key}: {val}\n")

    print("\n최적 파라미터가 'best_lgbm_params.txt' 에 저장되었습니다.")
