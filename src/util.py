# src/util.py
import os
import matplotlib.pyplot as plt
import numpy as np
from .PARAMS import RESULT_DIR, WEIGHTS_DIR, RESULT_FIG_PREFIX
import xgboost as xgb
import joblib

def ensure_dir_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_training_curve(epochs, auc_scores, model_name="MLP"):
    ensure_dir_exists(RESULT_DIR)
    plt.figure()
    plt.plot(range(1, epochs+1), auc_scores, label="val AUC")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.title(f"{model_name} Training AUC Curve")
    plt.legend()
    save_path = os.path.join(RESULT_DIR, f"{RESULT_FIG_PREFIX}_{model_name}.png")
    plt.savefig(save_path)
    plt.close()

def save_best_model(model, model_name="MLP"):
    """모델 가중치를 저장하거나, joblib dump 등으로 저장"""
    ensure_dir_exists(WEIGHTS_DIR)
    
    if isinstance(model, xgb.XGBClassifier) or isinstance(model, xgb.XGBRegressor):
        # ✅ XGBoost 모델 저장 방식 (pickle 사용)
        save_path = os.path.join(WEIGHTS_DIR, f"{model_name}_best.pkl")
        joblib.dump(model, save_path)
        print(f"✅ XGBoost 모델 저장 완료: {save_path}")
    
    elif hasattr(model, "save_weights"):
        # ✅ Keras 모델의 경우 .h5 형식으로 저장
        save_path = os.path.join(WEIGHTS_DIR, f"{model_name}_best.h5")
        model.save_weights(save_path)
        print(f"✅ Keras 모델 저장 완료: {save_path}")
    
    else:
        # ✅ 기본적으로 joblib 사용
        save_path = os.path.join(WEIGHTS_DIR, f"{model_name}_best.pkl")
        joblib.dump(model, save_path)
        print(f"✅ 모델 저장 완료: {save_path}")
