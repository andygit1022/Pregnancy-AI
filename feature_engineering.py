import numpy as np
import pandas as pd
import gc
import time

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.preprocess import load_data, preprocess_data
from src.model import XGBoostModel, LightGBMModel, MLPModel
from src.PARAMS import (
    TRAIN_PATH, SEED, VALIDATION_SPLIT,
    MODEL_FOR_IVF, MODEL_FOR_DI,
    FEATURES_FOR_IVF, FEATURES_FOR_DI
)
from src.util import save_best_model



# 1) IVF / DI Ablation Study 실행 함수
def ablation_study(procedure_type, out_csv="results/csv/ablation_result.csv"):
    """
    IVF / DI별 Ablation Study 수행

    procedure_type : "IVF" 또는 "DI"
    out_csv       : 결과 저장 경로 (IVF / DI별로 다르게 저장됨)
    """
    if procedure_type not in ["IVF", "DI"]:
        raise ValueError(f"[ERROR] Invalid procedure_type: {procedure_type}")

    # 1) 사용될 Feature 목록 선택
    all_features = FEATURES_FOR_IVF if procedure_type == "IVF" else FEATURES_FOR_DI

    # 2) 모델 선택
    model_name = MODEL_FOR_IVF if procedure_type == "IVF" else MODEL_FOR_DI

    # 3) Train 데이터 로드 및 전처리
    df_train = load_data(TRAIN_PATH)
    X, y = preprocess_data(df_train, is_train=True, procedure_type=procedure_type)

    # 4) Train / Validation Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=VALIDATION_SPLIT,
        random_state=SEED,
        stratify=y
    )

    #  변환된 Feature 리스트 저장
    transformed_features = list(X_train.columns)

    results = []

    print(f"\n🔍 [Ablation Study 시작] {procedure_type} ({len(all_features)} features)")

    start_time = time.time()  #  총 실행 시간 측정 시작

    for f_remove in all_features:
        print(f"\n[ABALATION] Removing feature: {f_remove}")

        # 1) Feature 변환 확인 (One-Hot Encoding 포함)
        related_features = [col for col in transformed_features if col.startswith(f_remove)]

        if not related_features:
            print(f"⚠️ [Warning] Feature {f_remove}가 변환된 데이터셋에서 존재하지 않음. 스킵.")
            continue  # 변환된 Feature가 없으면 스킵

        print(f"👉 Removing related features: {related_features}")

        # 2) Feature 제외한 새로운 데이터셋 생성
        new_features = [col for col in transformed_features if col not in related_features]
        X_train_new = X_train[new_features]
        X_val_new = X_val[new_features]

        # 3) 모델 생성 및 학습
        model = create_model(model_name, procedure_type)
        model.fit(X_train_new, y_train, X_val_new, y_val)

        # 4) Validation ROC 계산
        y_val_proba = model.predict_proba(X_val_new)
        best_roc = roc_auc_score(y_val, y_val_proba)

        print(f"Feature removed: {f_remove},  Val ROC: {best_roc:.4f}")
        results.append((f_remove, best_roc))

        # 5) 메모리 정리
        del X_train_new, X_val_new, model
        gc.collect()

    end_time = time.time()  #  실행 시간 종료
    elapsed_time = end_time - start_time
    print(f"\n [완료] {procedure_type} Ablation Study 총 소요 시간: {elapsed_time:.2f} 초")

    # 6) 결과 저장
    df_result = pd.DataFrame(results, columns=["removed_feature", "best_roc"])
    df_result.to_csv(out_csv, index=False)
    print(f" Ablation result saved to {out_csv}")



# 2) IVF / DI 별 모델 생성 함수

def create_model(model_name, procedure_type):
    """
    IVF / DI별 모델을 생성하는 함수

    model_name     : 사용할 모델 (XGB, LGB, MLP 등)
    procedure_type : "IVF" 또는 "DI"
    """
    if model_name.upper() == "XGB":
        return XGBoostModel(use_smote=False, ivf_di=procedure_type.lower())
    elif model_name.upper() == "LGB":
        return LightGBMModel()
    elif model_name.upper() == "MLP":
        return MLPModel()
    else:
        raise ValueError(f"[ERROR] 지원되지 않는 모델 유형: {model_name}")



# 3) main() 함수에서 Ablation 실행

def main():
    total_start_time = time.time()  #  전체 실행 시간 측정 시작

    # IVF Ablation Study 실행
    ablation_study("IVF", out_csv="results/csv/ablation_result_ivf.csv")

    # DI Ablation Study 실행
    ablation_study("DI", out_csv="results/csv/ablation_result_di.csv")

    total_end_time = time.time()  #  전체 실행 시간 종료
    total_elapsed_time = total_end_time - total_start_time
    print(f"\n전체 Ablation Study 총 소요 시간: {total_elapsed_time:.2f} 초")


if __name__ == "__main__":
    main()
