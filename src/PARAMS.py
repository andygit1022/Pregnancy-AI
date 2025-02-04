# src/PARAMS.py

#####################
# 데이터 경로 설정  #
#####################
TRAIN_PATH = "./data/train.csv"
TEST_PATH = "./data/test.csv"

######################################
# 전처리 시 기본적으로 적용할 전략  #
######################################
DEFAULT_NUMERIC_IMPUTE = "mean"   # ["mean", "median", "zero", "drop"] 등
DEFAULT_OHE_IMPUTE = "none"       # One-Hot할 컬럼에서 결측치가 나오면 어떻게 처리할지
DEFAULT_BINARY_IMPUTE = "zero"    # 0/1 피처에서 결측치 나오면 0으로
DEFAULT_INT_IMPUTE = "zero"       # int형에서 결측치 나오면 0으로

# 라벨(임신 성공 여부) 컬럼명
TARGET_COL = "임신 성공 여부"

#####################
# 전체 피처 정의    #
#####################
#
#  key   : 실제 CSV상의 컬럼명
#  value : ( "처리방식", "결측치대체방식" )
#           처리방식 예: "OHE", "INT", "BINARY", "SPECIAL", "drop" 등
#           결측치 대체방식 예: "mean", "zero", "none", "drop" ...
#

FULL_FEATURES = {
    # 예) ID는 학습에 사용하지 않을 경우 drop
    "ID": ("drop", None),

    # 시술 시기 코드 → One-Hot, 결측치: zero(혹은 none) 처리
    "시술 시기 코드": ("OHE", DEFAULT_OHE_IMPUTE),

    # 시술 당시 나이 → One-Hot, 결측치 없음으로 가정
    "시술 당시 나이": ("OHE", "none"),

    # 임신 시도 또는 마지막 임신 경과 연수 → int, 결측 시 zero로
    "임신 시도 또는 마지막 임신 경과 연수": ("INT", "zero"),

    # 시술 유형 [DI, IVF] → One-Hot
    "시술 유형": ("OHE", "none"),

    # **특정 시술 유형**: 슬래시/콜론 파싱으로 멀티핫 & 별도 카운트
    "특정 시술 유형": ("SPECIAL", "none"),

    # 배란 자극 여부 [0,1] → BINARY, 비어있으면 0
    "배란 자극 여부": ("BINARY", "zero"),

    # 배란 유도 유형 → 사용 안 함
    "배란 유도 유형": ("drop", None),

    # 단일 배아 이식 여부 [0,1] → BINARY, 비어있으면 0
    "단일 배아 이식 여부": ("BINARY", "zero"),

    # 착상 전 유전 검사 사용 여부 [0,1]
    "착상 전 유전 검사 사용 여부": ("BINARY", "zero"),

    # 착상 전 유전 진단 사용 여부 [0,1]
    "착상 전 유전 진단 사용 여부": ("BINARY", "zero"),

    # 남성/여성/부부 주/부 불임 원인, 불명확, 불임 원인 - 난관 질환 등 → 모두 BINARY
    "남성 주 불임 원인": ("BINARY", "zero"),
    "남성 부 불임 원인": ("BINARY", "zero"),
    "여성 주 불임 원인": ("BINARY", "zero"),
    "여성 부 불임 원인": ("BINARY", "zero"),
    "부부 주 불임 원인": ("BINARY", "zero"),
    "부부 부 불임 원인": ("BINARY", "zero"),
    "불명확 불임 원인": ("BINARY", "zero"),
    "불임 원인 - 난관 질환": ("BINARY", "zero"),
    "불임 원인 - 남성 요인": ("BINARY", "zero"),
    "불임 원인 - 배란 장애": ("BINARY", "zero"),
    "불임 원인 - 여성 요인": ("BINARY", "zero"),
    "불임 원인 - 자궁경부 문제": ("BINARY", "zero"),
    "불임 원인 - 자궁내막증": ("BINARY", "zero"),
    "불임 원인 - 정자 농도": ("BINARY", "zero"),
    "불임 원인 - 정자 면역학적 요인": ("BINARY", "zero"),
    "불임 원인 - 정자 운동성": ("BINARY", "zero"),
    "불임 원인 - 정자 형태": ("BINARY", "zero"),

    # 배아 생성 주요 이유 → One-Hot
    "배아 생성 주요 이유": ("OHE", "none"),

    # 총 시술 횟수 → One-Hot
    "총 시술 횟수": ("OHE", "none"),

    # 클리닉 내 총 시술 횟수 → One-Hot
    "클리닉 내 총 시술 횟수": ("OHE", "none"),

    # IVF 시술 횟수, DI 시술 횟수, 총/IVF/DI 임신 횟수, 총/IVF/DI 출산 횟수 → 모두 One-Hot
    "IVF 시술 횟수": ("OHE", "none"),
    "DI 시술 횟수": ("OHE", "none"),
    "총 임신 횟수": ("OHE", "none"),
    "IVF 임신 횟수": ("OHE", "none"),
    "DI 임신 횟수": ("OHE", "none"),
    "총 출산 횟수": ("OHE", "none"),
    "IVF 출산 횟수": ("OHE", "none"),
    "DI 출산 횟수": ("OHE", "none"),

    # 총 생성 배아 수 등 int → 결측치 처리: mean or drop 등
    "총 생성 배아 수": ("INT", "zero"),
    "미세주입된 난자 수": ("INT", "zero"),
    "미세주입에서 생성된 배아 수": ("INT", "zero"),
    "이식된 배아 수": ("INT", "zero"),
    "미세주입 배아 이식 수": ("INT", "zero"),
    "저장된 배아 수": ("INT", "zero"),
    "미세주입 후 저장된 배아 수": ("INT", "zero"),
    "해동된 배아 수": ("INT", "zero"),
    "해동 난자 수": ("INT", "zero"),
    "수집된 신선 난자 수": ("INT", "zero"),
    "저장된 신선 난자 수": ("INT", "zero"),
    "혼합된 난자 수": ("INT", "zero"),
    "파트너 정자와 혼합된 난자 수": ("INT", "zero"),
    "기증자 정자와 혼합된 난자 수": ("INT", "zero"),

    # 난자 출처, 정자 출처, 난자/정자 기증자 나이 → One-Hot
    "난자 출처": ("OHE", "none"),  # 결측 → "알 수 없음"으로 처리해줄 수도 있음
    "정자 출처": ("OHE", "none"),
    "난자 기증자 나이": ("OHE", "none"),
    "정자 기증자 나이": ("OHE", "none"),

    # 동결/신선/기증 배아 사용 여부, 대리모, PGD/PGS 시술 여부 → [0,1], 결측 mean or zero
    "동결 배아 사용 여부": ("BINARY", "zero"),
    "신선 배아 사용 여부": ("BINARY", "zero"),
    "기증 배아 사용 여부": ("BINARY", "zero"),
    "대리모 여부": ("BINARY", "zero"),
    "PGD 시술 여부": ("BINARY", "zero"),
    "PGS 시술 여부": ("BINARY", "zero"),

    # 난자 채취 경과일, 난자 해동 경과일, 배아 해동 경과일 → 삭제
    "난자 채취 경과일": ("drop", None),
    "난자 해동 경과일": ("drop", None),
    "배아 해동 경과일": ("drop", None),

    # 난자 혼합 경과일 → 0~7, 범주화(0,1,2이상) → One-Hot
    "난자 혼합 경과일": ("OHE", "none"),

    # 배아 이식 경과일 → 0~7, 결측이면 3으로 → 그 뒤 One-Hot
    "배아 이식 경과일": ("OHE", "special_fill_3"),

    # 임신 성공 여부 [0,1] (타겟)
    "임신 성공 여부": ("drop", None),  # 실제 학습시 y로 분리하므로 drop
}

#################################
# 실제로 학습에 사용할 FEATURES #
#################################


#"배아 생성 주요 이유"
FEATURES = [
    "시술 시기 코드",
    "시술 당시 나이",
    "임신 시도 또는 마지막 임신 경과 연수",
    "시술 유형",
    "특정 시술 유형",
    "배란 자극 여부",
    "단일 배아 이식 여부",
    "착상 전 유전 검사 사용 여부",
    "착상 전 유전 진단 사용 여부",
    "남성 주 불임 원인",
    "남성 부 불임 원인",
    "여성 주 불임 원인",
    "여성 부 불임 원인",
    "부부 주 불임 원인",
    "부부 부 불임 원인",
    "불명확 불임 원인",
    "불임 원인 - 난관 진환",
    "불임 원인 - 남성 요인",
    "불임 원인 - 배란 장애",
    "불임 원인 - 여성 요인",
    "불임 원인 - 자궁 경부 문제",
    "불임 원인 - 자궁내막증",
    "불임 원인 - 정자 농도",
    "불임 원인 - 정자 면역학적 요인",
    "불임 원인 - 정자 운동성",
    "불임 원인 - 정자 형태",
    "총 시술 횟수",
    "클리닉 내 총 시술 횟수",
    "IVF 시술 횟수",
    "DI 시술 횟수",
    "총 생성 배아 수",
    "미세주입된 난자 수",
    "미세주입에서 생성된 배아 수",
    "이식된 배아 수",
    "미세주입 배아 이식 수",
    "저장된 배아 수",
    "미세주입 후 저장된 배아 수",
    "해동된 배아 수",
    "해동 난자 수",
    "수집된 신선 난자 수",
    "저장된 신선 난자 수",
    "혼합된 난자 수",
    "파트너 정자와 혼합된 난자 수",
    "기증자 정자와 혼합된 난자 수",
    "난자 출처",
    "정자 출처",
    "난자 기증자 나이",
    "정자 기증자 나이",
    "동결 배아 사용 여부",
    "신선 배아 사용 여부",
    "기증 배아 사용 여부",
    "대리모 여부",
    "PGD 시술 여부",
    "PSG 시술 여부",
    "난자 채취 경과일",
    "난자 해동 경과일",
    "난자 혼합 경과일",
    "배아 이식 경과일",
    "배아 해동 경과일"

]


########################
# 모델 학습 관련 설정  #
########################
SEED = 42
SAVE_EPOCH = 1
EPOCHS = 200    #MLP
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.1

# MLP
MLP_LEARNING_RATE = 1e-4
MLP_OPTIMIZER = "adam" 
MLP_HIDDEN_UNITS = [64, 32]
MLP_DROPOUT = 0.2

# XGBoost
XGB_PARAMS = {
    "n_estimators": 2000,
    "learning_rate": 0.01,
    "max_depth": 8,
    "random_state": SEED,
}

XGB_EVAL_METRIC = "auc"


# LightGBM
LGB_PARAMS = {
    "n_estimators": 100,
    "learning_rate": 0.1,
    "max_depth": -1,
    "random_state": SEED
}

#######################
# 결과 저장 관련 설정 #
#######################
RESULT_DIR = "results"
WEIGHTS_DIR = "results/weights"
RESULT_FIG_PREFIX = "training_curve"
