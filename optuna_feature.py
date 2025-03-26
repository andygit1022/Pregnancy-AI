import optuna
import lightgbm as lgb
import numpy as np
import pandas as pd
import logging
import itertools
import gc
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from src.preprocess import load_data, preprocess_data
from src.PARAMS import TRAIN_PATH, VALIDATION_SPLIT, SEED, FEATURES_FOR_IVF, FEATURES_FOR_DI, FEATURES

#  로깅 설정 (optuna_feature_selection.log에 저장)
logging.basicConfig(
    filename="optuna_feature_selection.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Feature Selection 결과 저장 경로
FEATURE_SELECTION_CSV = "results/csv/feature_selection_results.csv"
i = 0

def objective(trial, X_train, y_train, X_val, y_val, feature_list, save_results=True):
    """
    Optuna 최적화 함수: Feature Subset을 선택하여 LightGBM 모델 학습 및 검증
    """
    #  1) Feature Mapping (원본 → OHE 포함 Feature 목록 매핑)
    feature_map = {}
    for feature in feature_list:
        related_features = [col for col in X_train.columns if col.startswith(feature)]
        feature_map[feature] = related_features if related_features else [feature]

    #  2) Feature Subset 랜덤 선택
    selected_original_features = [
        feature for feature in feature_list if trial.suggest_categorical(feature, [True, False])
    ]

    #  3) OHE 변환된 Feature 포함
    selected_features = list(itertools.chain(*[feature_map[f] for f in selected_original_features]))

    #  4) 최소 Feature 개수 제한 (예: 최소 5개)
    if len(selected_features) < 5:
        return 0.5  # 최소 개수보다 작으면 학습 없이 낮은 AUC 반환

    #  5) 존재하는 Feature만 유지
    valid_features = [f for f in selected_features if f in X_train.columns]

    #  6) 존재하지 않는 Feature 경고 메시지 출력
    missing_features = [f for f in selected_features if f not in X_train.columns]
    if missing_features:
        print(f"⚠️ Warning: {len(missing_features)} features not found in dataset. Skipping: {missing_features}")
        logging.warning(f"⚠️ Warning: Skipping missing features: {missing_features}")

    #  7) Feature가 너무 적으면 Skip
    if len(valid_features) < 5:
        return 0.5

    #  8) 데이터셋 구성 (선택한 Feature만 사용)
    X_train_selected = X_train[valid_features]
    X_val_selected = X_val[valid_features]

    #  9) LightGBM 하이퍼파라미터 설정
    params = {
        "n_estimators": 600, #231
        "learning_rate": 0.03076570743923803,
        "num_leaves": 67,
        "max_depth": 11,
        "min_child_samples": 48,
        "subsample": 0.8449555137871144,
        "colsample_bytree": 0.9803039748117706,
        "reg_alpha": 1.8087496933429417,
        "reg_lambda": 0.0001276697427635871,
        "min_split_gain": 0.00496364966672079,
        "min_child_weight": 0.6013014083449827,
        
        "objective": "binary",
        "metric": "auc",
        "random_state": SEED,
        "verbosity": -1,
        "early_stopping_rounds": 100
    }
    

    #  10) 모델 학습
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train_selected, y_train,
        eval_set=[(X_val_selected, y_val)],
    )

    #  11) 검증 데이터 AUC 계산
    y_val_proba = model.predict_proba(X_val_selected)[:, 1]
    auc_val = roc_auc_score(y_val, y_val_proba)

    #  12) 로그 출력 및 저장
    print(f"[Trial {trial.number}] AUC: {auc_val:.6f} | Selected Features: {len(valid_features)}")
    logging.info(f"[Trial {trial.number}] AUC: {auc_val:.6f} | Selected Features: {valid_features}")

    #  13) CSV 저장 (옵션)
    if save_results:
        df_result = pd.DataFrame({
            "trial_number": [trial.number],
            "auc": [auc_val],
            "selected_features": [", ".join(valid_features)]
        })
        df_result.to_csv(f"{FEATURE_SELECTION_CSV}{i}", mode='a', header=not pd.io.common.file_exists(FEATURE_SELECTION_CSV), index=False)

    return auc_val

def feature_selection(procedure_type="ALL", n_trials=100, save_results=False):
    """
    IVF / DI 데이터에서 Feature Subset을 최적화하는 함수
    """
    #  1) 데이터 로드 및 전처리
    df_train = load_data(TRAIN_PATH)
    if procedure_type == "ALL":
        feature_list = FEATURES
    elif procedure_type == "IVF":
        feature_list = FEATURES_FOR_IVF
    else :
        feature_list = FEATURES_FOR_DI
    
    X, y = preprocess_data(df_train, is_train=True, procedure_type=procedure_type)

    #  2) Train / Validation Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=VALIDATION_SPLIT,
        random_state=SEED,
        stratify=y
    )

    #  3) Optuna 실행
    study = optuna.create_study(direction="maximize")  # AUC 최대화
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val, feature_list, save_results),
                   n_trials=n_trials)

    #  4) 최적 Feature Set 및 성능 출력
    print("\n=====================================")
    print(f" Best AUC: {study.best_value:.6f}")
    print(" Best Feature Set:", study.best_trial.params)

    #  5) 최적 Feature Set을 최종 CSV에 저장
    if not save_results:
        df_best_result = pd.DataFrame({
            "trial_number": [study.best_trial.number],
            "auc": [study.best_value],
            "selected_features": [", ".join([f for f, use in study.best_trial.params.items() if use])]
        })
        df_best_result.to_csv(FEATURE_SELECTION_CSV, index=True)
        print(f" 최적 Feature Set이 {FEATURE_SELECTION_CSV} 에 저장되었습니다.")


def format_selected_features(csv_path, best_trial_index=None):
    """
    CSV에서 선택된 Feature Set을 Python 코드에 넣기 편하게 변환하는 함수
    """
    df = pd.read_csv(csv_path)

    #  가장 높은 AUC를 가진 Trial 찾기
    if best_trial_index is None:
        best_trial_index = df["auc"].idxmax()

    best_features = df.loc[best_trial_index, "selected_features"]
    feature_list = best_features.split(", ")

    #  Python 리스트 형식으로 변환
    formatted_str = ",\n    ".join([f'"{feature}"' + (", ##" if i % 5 == 4 else "") for i, feature in enumerate(feature_list)])
    
    #  최종 결과 출력
    formatted_str = f"selected_features = [\n    {formatted_str}\n]"
    
    return formatted_str


if __name__ == "__main__":
    #  IVF & DI 각각 Feature Selection 실행
    #feature_selection("ALL", n_trials=500, save_results=True)
    feature_selection("IVF", n_trials=1000, save_results=True)
    i+=1
    feature_selection("DI", n_trials=1000, save_results=True)

    #  최적 Feature Set을 Python 코드로 변환하여 출력
    csv_path = FEATURE_SELECTION_CSV
    formatted_feature_list = format_selected_features(csv_path)
    print(formatted_feature_list)  # Python 코드에 바로 복사 가능
