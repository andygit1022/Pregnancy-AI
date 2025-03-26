# main.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.preprocess import (
    load_data, preprocess_data
)
from src.model import (
    XGBoostModel, LightGBMModel, MLPModel
)
from src.PARAMS import (
    TRAIN_PATH, TEST_PATH, TARGET_COL,
    MODEL_FOR_IVF, MODEL_FOR_DI,
    FEATURES_FOR_IVF, FEATURES_FOR_DI,
    VALIDATION_SPLIT, SEED
)
from src.util import save_best_model

def main():
    # 1) 전체 Train CSV 로드
    df_train = load_data(TRAIN_PATH)

    # 2) IVF / DI 분리
    df = df_train.copy()

    # 3) 각 시술유형별 전처리 + Train/Val split
    X_ivf, y_ivf = preprocess_data(df, is_train=True, procedure_type="ALL")

    # IVF split
    X_ivf_train, X_ivf_val, y_ivf_train, y_ivf_val = train_test_split(
        X_ivf, y_ivf, test_size=VALIDATION_SPLIT, random_state=SEED, stratify=y_ivf
    )
    

    # 4) IVF / DI 각각 모델 선택
    all_model = create_model(MODEL_FOR_IVF,ivf_di = "ALL")

    # 5) 모델 학습
    print("\n=== [ALL MODEL TRAIN] ===")
    all_model.fit(X_ivf_train, y_ivf_train, X_ivf_val, y_ivf_val)
    #save_best_model(all_model.model, model_name="IVF")


    # ========== TEST ==========
    df_test = load_data(TEST_PATH)

    # 시술유형이 IVF인 Test 행 / DI인 Test 행
    df_test_all = df_test.copy()

    # 전처리
    X_test_all, _ = preprocess_data(df_test_all,  is_train=False,procedure_type="ALL")

    # 예측
    y_proba_all = all_model.predict_proba(X_test_all)

    # 예측 결과를 ID 기준으로 합치기
    # df_test_ivf["prob"] = y_proba_ivf, df_test_di["prob"] = y_proba_di
    df_test_all["probability"] = y_proba_all

    # 두 데이터프레임을 다시 합침
    # 원본 순서로 정렬하고 싶다면
    df_test_all.sort_index(inplace=True)

    # 최종 csv
    submission = df_test_all[["ID", "probability"]]
    submission.to_csv("results/csv/submission.csv", index=False)
    print("Submission saved: submission.csv")

def create_model(model_name,ivf_di="ivf"):
    """
    문자열(model_name)에 따라 다른 모델 객체 리턴
    예: "XGB" -> XGBoostModel(), "MLP" -> MLPModel(), ...
    """
    if model_name.upper() == "XGB":
        return XGBoostModel(use_smote=False,ivf_di=ivf_di)
    elif model_name.upper() == "LGB":
        return LightGBMModel(ivf_di=ivf_di)
    elif model_name.upper() == "MLP":
        return MLPModel()
    else:
        return XGBoostModel()  # default

if __name__ == "__main__":
    main()