# # main.py
# import numpy as np
# import pandas as pd
# import gc

# from sklearn.metrics import roc_auc_score
# from sklearn.model_selection import train_test_split

# from src.preprocess import (
#     get_train_data, get_test_data, load_data, preprocess_data
# )
# from src.model import (
#     XGBoostModel, LightGBMModel, MLPModel
# )
# from src.PARAMS import (
#     VALIDATION_SPLIT, SEED,
#     TEST_PATH, FEATURES as ORIGINAL_FEATURES
# )
# from src.util import save_best_model, save_training_curve

# ############################################
# # 1) Ablation Study 함수
# ############################################
# def ablation_study(model_class, all_features, out_csv="ablation_result.csv"):
#     """
#     model_class : XGBoostModel, LightGBMModel, MLPModel 중 택1 (class)
#     all_features: 전체 FEATURES 리스트
#     out_csv     : 결과 저장 경로
    
#     과정:
#      - FEATURES에서 하나씩 제외
#      - 그때마다 Train/Val 학습 → best_roc 계산
#      - CSV로 저장
#     """
#     results = []
    
#     # 원본 FEATURES 백업
#     original_features = all_features[:]
    
#     for f_remove in original_features:
#         print(f"\n[ABALTION] Removing feature: {f_remove}")
        
#         # 1) 해당 feature를 제외한 새로운 FEATURES 리스트 생성
#         new_features = [f for f in original_features if f != f_remove]
        
#         # 2) PARAMS.py 의 FEATURES 를 임시로 교체
#         #    - 여기서는 전역 상수로 정의되어 있으므로, 간단히 overwrite
#         #    - 실제로는 PARAMS에서 get/set 함수를 제공하거나, 
#         #      preprocess_data 함수에 파라미터로 넘겨주는 방식을 써도 됨
#         from src.PARAMS import FEATURES
#         old_features = FEATURES[:]            # 백업
#         FEATURES.clear()
#         FEATURES.extend(new_features)         # FEATURES 를 new_features 로 교체
        
#         # 3) Train 데이터 로드 & 전처리
#         X, y = get_train_data()
#         X_train, X_val, y_train, y_val = train_test_split(
#             X, y,
#             test_size=VALIDATION_SPLIT,
#             random_state=SEED,
#             stratify=y
#         )
        
#         # 4) 모델 학습
#         model = model_class()
#         model.fit(X_train, y_train, X_val, y_val)
        
#         # 5) Validation ROC 계산
#         #    - 각 model 클래스에서 best_score 를 저장했을 수도 있고,
#         #      여기서는 직접 predict_proba + roc_auc_score 계산 예시
#         y_val_proba = model.predict_proba(X_val)
#         best_roc = roc_auc_score(y_val, y_val_proba)
        
#         print(f"Feature removed: {f_remove},  Val ROC: {best_roc:.4f}")
#         results.append((f_remove, best_roc))
        
#         # 6) FEATURES 원복
#         FEATURES.clear()
#         FEATURES.extend(old_features)
        
#         # 7) 메모리 정리
#         del X, y, X_train, X_val, y_train, y_val, model
#         gc.collect()
    
#     # 8) 결과 저장
#     df_result = pd.DataFrame(results, columns=["removed_feature", "best_roc"])
#     df_result.to_csv(out_csv, index=False)
#     print(f"✅ Ablation result saved to {out_csv}")

# ############################################
# # 2) main() 함수 예시
# ############################################
# def main():
#     # ---------------------
#     # 보통 학습/추론 흐름
#     # ---------------------
#     # # 1) Train 데이터
#     # X, y = get_train_data()
#     # X_train, X_val, y_train, y_val = train_test_split(
#     #     X, y, test_size=VALIDATION_SPLIT, random_state=SEED, stratify=y
#     # )
#     # model = XGBoostModel()
#     # model.fit(X_train, y_train, X_val, y_val)
#     # save_best_model(model.model, model_name="XGB")
#     # ...
#     #
#     # --- (중략) ---

#     # ---------------------
#     # Ablation Study 실행
#     # ---------------------
#     from src.model import XGBoostModel
#     ablation_study(
#         model_class=XGBoostModel,
#         all_features=ORIGINAL_FEATURES,   # PARAMS.py 의 FEATURES 복사본
#         out_csv="ablation_result.csv"
#     )

# if __name__ == "__main__":
#     main()





# # src/main.py
# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split

# from src.preprocess import (
#     get_train_data, get_test_data, load_data, preprocess_data
# )
# from src.model import (
#     XGBoostModel, LightGBMModel, MLPModel
# )
# from src.PARAMS import (
#     VALIDATION_SPLIT, SEED,
#     TEST_PATH
# )
# from src.util import save_best_model, save_training_curve

# def main():
#     # ------------------------------------------------
#     # 1) Train 데이터 읽고 전처리
#     # ------------------------------------------------
    
#     X, y = get_train_data()

#     # Train/Validation split
#     X_train, X_val, y_train, y_val = train_test_split(
#         X, y,
#         test_size=VALIDATION_SPLIT,
#         random_state=SEED,
#         stratify=y
#     )

#     # ------------------------------------------------
#     # 2) 모델 선택 (예: XGBoost)
#     # ------------------------------------------------
#     model = XGBoostModel(use_smoteenn=True)
#     #  model = LightGBMModel()
#     #  model = MLPModel()

#     # ------------------------------------------------
#     # 3) 모델 학습
#     # ------------------------------------------------
#     model.fit(X_train, y_train, X_val, y_val)

#     # Best 모델 가중치 저장 (선택)
#     save_best_model(model.model, model_name="XGB")

#     # ------------------------------------------------
#     # 4) Test 데이터 불러와 확률 예측
#     # ------------------------------------------------
#     # 4-1) 원본 Test CSV 로드 (ID 열을 나중에 그대로 사용하기 위함)
#     df_test = load_data(TEST_PATH)
#     # 4-2) Test 전처리 (동일 로직)
#     X_test, _ = preprocess_data(df_test, is_train=False)

#     # 4-3) 예측 (확률)
#     y_proba = model.predict_proba(X_test)

#     # ------------------------------------------------
#     # 5) submission 파일 생성 (ID, probability)
#     # ------------------------------------------------
#     submission = pd.DataFrame({
#         "ID": df_test["ID"],
#         "probability": y_proba
#     })

#     # 예: "submission.csv"로 저장
#     submission.to_csv("submission.csv", index=False)
#     print("✅ Submission file saved: submission.csv")


# if __name__ == "__main__":
#     main()



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
    df_ivf = df_train.copy()
    df_di  = df_train.copy()

    # 3) 각 시술유형별 전처리 + Train/Val split
    X_ivf, y_ivf = preprocess_data(df_ivf, is_train=True, procedure_type="IVF")
    X_di, y_di   = preprocess_data(df_di, is_train=True, procedure_type="DI")

    # IVF split
    X_ivf_train, X_ivf_val, y_ivf_train, y_ivf_val = train_test_split(
        X_ivf, y_ivf, test_size=VALIDATION_SPLIT, random_state=SEED, stratify=y_ivf
    )
    
    # DI split
    X_di_train, X_di_val, y_di_train, y_di_val = train_test_split(
        X_di, y_di, test_size=VALIDATION_SPLIT, random_state=SEED, stratify=y_di
    )

    # 4) IVF / DI 각각 모델 선택
    ivf_model = create_model(MODEL_FOR_IVF,ivf_di = "ivf")
    di_model  = create_model(MODEL_FOR_DI,ivf_di = "di")

    # 5) 모델 학습
    print("\n=== [IVF MODEL TRAIN] ===")
    ivf_model.fit(X_ivf_train, y_ivf_train, X_ivf_val, y_ivf_val)
    save_best_model(ivf_model.model, model_name="IVF")

    print("\n=== [DI MODEL TRAIN] ===")
    di_model.fit(X_di_train, y_di_train, X_di_val, y_di_val)
    save_best_model(di_model.model, model_name="DI")

    # ========== TEST ==========
    df_test = load_data(TEST_PATH)

    # 시술유형이 IVF인 Test 행 / DI인 Test 행
    df_test_ivf = df_test[df_test["시술 유형"] == "IVF"].copy()
    df_test_di  = df_test[df_test["시술 유형"] == "DI"].copy()

    # 전처리
    X_test_ivf, _ = preprocess_data(df_test_ivf,  is_train=False,procedure_type="IVF")
    X_test_di,  _ = preprocess_data(df_test_di, is_train=False,procedure_type="DI")

    # 예측
    y_proba_ivf = ivf_model.predict_proba(X_test_ivf)
    y_proba_di  = di_model.predict_proba(X_test_di)

    # 예측 결과를 ID 기준으로 합치기
    df_test_ivf["prob"] = y_proba_ivf, df_test_di["prob"] = y_proba_di
    df_test_ivf["probability"] = y_proba_ivf
    df_test_di["probability"]  = y_proba_di

    # 두 데이터프레임을 다시 합침
    df_submit = pd.concat([df_test_ivf, df_test_di], axis=0)
    # 원본 순서로 정렬하고 싶다면
    df_submit.sort_index(inplace=True)

    # 최종 csv
    submission = df_submit[["ID", "probability"]]
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