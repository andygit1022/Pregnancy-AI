#optuna

# optuna_xgb_tuning.py
import optuna
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import xgboost as xgb

# 필요한 함수/상수들을 import
# (프로젝트 구조에 맞게 import 경로를 수정하세요)
from src.preprocess import load_data, preprocess_data
from src.PARAMS import TRAIN_PATH, VALIDATION_SPLIT, SEED

def objective(trial):
    """
    Optuna가 반복 호출하는 최적화 대상 함수.
    trial: Optuna에서 각 하이퍼파라미터를 제안(샘플링)하는 객체
    """
    # 1) 탐색할 XGBoost 파라미터 공간 정의
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
        # 그 외 필요한 파라미터가 있다면 추가
        "eval_metric": "auc",     # XGBoost 평가지표
        "random_state": SEED,
        #"use_label_encoder": False
    }

    # 2) IVF 데이터만 추출하여 전처리
    #    (DI를 하고 싶으면 "DI"로 교체)
    df_train = load_data(TRAIN_PATH)

    #X_ivf, y_ivf = preprocess_data(df_train, is_train=True, procedure_type="IVF")
    

    X_di, y_di = preprocess_data(df_train, is_train=True, procedure_type="DI")

    # -------------------------------------
    # 3) Train/Val split
    # -------------------------------------
    # X_train, X_val, y_train, y_val = train_test_split(
    #     X_ivf, y_ivf,
    #     test_size=VALIDATION_SPLIT,
    #     random_state=SEED,
    #     stratify=y_ivf
    # )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_di, y_di,
        test_size=VALIDATION_SPLIT,
        random_state=SEED,
        stratify=y_di
    )

    # 4) XGBoost 모델 생성/학습
    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        #early_stopping_rounds=10,   # 10번 동안 개선 없으면 중단
        verbose=100               # 진행 로그 간소화
    )

    # 5) 검증 세트에 대해 ROC-AUC 계산
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    auc_val = roc_auc_score(y_val, y_pred_proba)

    return auc_val


if __name__ == "__main__":
    # Optuna를 통해 XGBoost 하이퍼파라미터 탐색
    study = optuna.create_study(direction="maximize")  # AUC 최대화
    study.optimize(objective, n_trials=100)             # 예: 30번 탐색

    # 결과 출력
    print("=====================================")
    print(f"Best AUC: {study.best_value:.4f}")
    print("Best Params:", study.best_params)

    # 최적 파라미터를 텍스트 파일에 저장
    with open("best_xgb_params.txt", "w", encoding="utf-8") as f:
        f.write(f"Best AUC: {study.best_value:.4f}\n")
        for key, val in study.best_params.items():
            f.write(f"{key}: {val}\n")

    print("\n최적 파라미터가 'best_xgb_params.txt' 에 저장되었습니다.")
