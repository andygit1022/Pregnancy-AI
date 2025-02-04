# src/preprocess.py

import pandas as pd
import numpy as np
import re

from .PARAMS import (
    TRAIN_PATH, TEST_PATH, TARGET_COL, 
    FULL_FEATURES, FEATURES
)

#####################
# 1) CSV 불러오기
#####################
def load_data(csv_path):
    df = pd.read_csv(csv_path)
    return df


###########################
# "특정 시술 유형" 파싱 함수
###########################
# 1) 슬래시('/') 개수, 2) 콜론(':') 개수,
# 3) 특정 토큰들 각각의 등장 횟수
SPECIAL_TOKENS = [
    "ICSI", "IVF", "IUI", "IVI", "ICI","Generic DI"
    "GIFT", "AH", "BLASTOCYST", "Unknown"
]

def parse_special_treatment_counts(treatment_str):
    """
    예: "ICSI / BLASTOCYST:IVF / BLASTOCYST"

    return 예시:
    {
      'slash_count': 2,
      'colon_count': 1,
      'ICSI_count': 1,
      'IVF_count': 1,
      'IUI_count': 0,
      'BLASTOCYST_count': 2,
      ...
    }
    """
    # 기본값 0
    info = {f"{t}_count": 0 for t in SPECIAL_TOKENS}
    info["slash_count"] = 0
    info["colon_count"] = 0

    if pd.isnull(treatment_str) or not isinstance(treatment_str, str):
        return info  # 전부 0

    # 슬래시, 콜론 개수
    info["slash_count"] = treatment_str.count('/')
    info["colon_count"] = treatment_str.count(':')

    # "/"와 ":"를 전부 공백 " "으로 치환 후, 스플릿
    # => 각각의 토큰을 쉽게 뽑아냄
    temp_str = treatment_str.replace('/', ' ').replace(':', ' ')
    tokens = temp_str.split()

    # tokens 내 각 단어가 SPECIAL_TOKENS 중 하나인지 확인하고
    # 등장 횟수 세기
    for tk in tokens:
        # 혹시 대소문자 일관성이 없다면 tk = tk.upper() 등 처리 필요
        # 여기서는 그대로
        if tk in SPECIAL_TOKENS:
            info[f"{tk}_count"] += 1

    return info


###########################
# 2) 실제 전처리 핵심 함수
###########################
def preprocess_data(df, is_train=True):
    # 사용할 컬럼만 추출
    use_cols = [col for col in FEATURES if col in FULL_FEATURES]
    if TARGET_COL in df.columns:
        use_cols += [TARGET_COL]

    df = df[use_cols].copy()
    out_df = pd.DataFrame(index=df.index)

    for col in df.columns:
        print(col)
        if col == TARGET_COL:
            continue

        process_type, na_strategy = FULL_FEATURES[col]
        if process_type == "drop":
            continue

        # (A) 결측치 처리
        series = apply_missing_strategy(df[col], process_type, na_strategy)

        # (B) 인코딩/타입 변환
        if process_type == "INT":
            out_df[col] = pd.to_numeric(series, errors="coerce").fillna(0).astype(int)

        elif process_type == "BINARY":
            bin_col = pd.to_numeric(series, errors="coerce").fillna(0).astype(int)
            out_df[col] = bin_col

        elif process_type == "OHE":
            dummies = pd.get_dummies(series, prefix=col)
            out_df = pd.concat([out_df, dummies], axis=1)

        elif process_type == "SPECIAL":
            # 이번엔 slash_count, colon_count 및 토큰별 "등장 횟수"를 구함
            # row별로 parse_special_treatment_counts() 호출
            slash_counts = []
            colon_counts = []
            # SPECIAL_TOKENS에 대해서만 count
            # 먼저 각 토큰별 list를 만든다
            token_cols = {t: [] for t in SPECIAL_TOKENS}

            for val in series:
                parsed = parse_special_treatment_counts(val)
                slash_counts.append(parsed["slash_count"])
                colon_counts.append(parsed["colon_count"])

                # 토큰별 count
                for t in SPECIAL_TOKENS:
                    token_cols[t].append(parsed[f"{t}_count"])

            # 이제 out_df에 컬럼으로 추가
            out_df[f"{col}_slash_count"] = slash_counts
            out_df[f"{col}_colon_count"] = colon_counts
            for t in SPECIAL_TOKENS:
                out_df[f"{col}_{t}_count"] = token_cols[t]

        else:
            out_df[col] = series

    # 타겟 분리
    y = None
    if TARGET_COL in df.columns:
        y = df[TARGET_COL].copy()
        y = pd.to_numeric(y, errors="coerce").fillna(0).astype(int)

    return out_df, y


#########################################
# 3) 결측치 처리 로직(전략별)
#########################################
def apply_missing_strategy(series, process_type, na_strategy):
    if na_strategy == "zero":
        return series.fillna(0)
    elif na_strategy == "mean":
        numeric_vals = pd.to_numeric(series, errors="coerce")
        mean_val = numeric_vals.mean()
        return numeric_vals.fillna(mean_val)
    elif na_strategy == "drop":
        return series.dropna()
    elif na_strategy == "none":
        return series.fillna("Unknown")
    elif na_strategy == "special_fill_3":
        return series.fillna(4)
    else:
        return series


########################
# 4) Train/Test 불러오기
########################
def get_train_data():
    df = load_data(TRAIN_PATH)
    X, y = preprocess_data(df, is_train=True)
    return X, y

def get_test_data():
    df = load_data(TEST_PATH)
    X, y = preprocess_data(df, is_train=False)
    return X, y
