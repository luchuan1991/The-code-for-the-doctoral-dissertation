# -*- coding: utf-8 -*-
"""
OnekeyNet 第四十八轮调参
- 针对 Blood_stasis 和 Qi_deficiency 进行进一步调整
"""

import os
import numpy as np
import pandas as pd
import logging
from sklearn.utils.class_weight import compute_class_weight
from onekey_algo.structure_dnn.run_structure_OnekeyNet import main as onekey_main
from collections import namedtuple

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# -------------------------------------------------
# 0. 数据清洗
# -------------------------------------------------
def load_numeric_csv(path: str) -> pd.DataFrame:
    """
    加载并清洗数值型数据
    """
    df = pd.read_csv(path)
    df = df.select_dtypes(include=["number", "bool"]).copy()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.median(numeric_only=True), inplace=True)

    eps = 1e-12
    mu, std = df.mean(), df.std().replace(0, eps)
    df = (df - mu) / std

    bad = df.columns[df.isnull().any() | ~np.isfinite(df).all()]
    if len(bad):
        df = df.drop(columns=bad)
        logging.warning(f"删除列: {bad.tolist()}")

    logging.info(f"最终特征维度 = {df.shape[1]}")
    return df

# -------------------------------------------------
# 1. 输入 / 输出
# -------------------------------------------------
def setup_input_output(feature_csv: str, model_root: str) -> tuple:
    """
    设置输入和输出路径
    """
    df_feat = load_numeric_csv(feature_csv)
    input_dim = df_feat.shape[1]
    os.makedirs(model_root, exist_ok=True)
    return df_feat, input_dim

# -------------------------------------------------
# 2. 参数设置
# -------------------------------------------------
def setup_input_settings(input_dim: int) -> dict:
    """
    设置输入参数
    """
    input_settings = {
        "Blood_stasis": {
            "feature_file": FEATURE_CSV,
            "input_dim": input_dim,
            "hidden_unit": [256, 512, 256],  # 减少模型复杂度
            "dropout": 0.6,  # 增加 Dropout 比例
            "batch_norm": True,
            "activation": "leaky_relu",
        },
        "Qi_deficiency": {
            "feature_file": FEATURE_CSV,
            "input_dim": input_dim,
            "hidden_unit": [256, 512, 256],  # 减少模型复杂度
            "dropout": 0.6,  # 增加 Dropout 比例
            "batch_norm": True,
            "activation": "leaky_relu",
        },
        "Dampness_syndrome": {
            "feature_file": FEATURE_CSV,
            "input_dim": input_dim,
            "hidden_unit": [512, 1024, 512],
            "dropout": 0.3,
            "batch_norm": True,
            "activation": "relu",
        },
        "CVD": {
            "feature_file": FEATURE_CSV,
            "input_dim": input_dim,
            "hidden_unit": [512, 1024, 512],
            "dropout": 0.3,
            "batch_norm": True,
            "activation": "relu",
        },
    }
    return input_settings

def setup_task_settings() -> dict:
    """
    设置任务参数
    """
    task_settings = {
        "Blood_stasis": {
            "label_file": r"F:\graduate\tongue\xinxueguan1397\xueyu\xueyu_labelxuerad.csv",
            "type": "clf", "num_classes": 2, "weights": 2.0,
        },
        "Qi_deficiency": {
            "label_file": r"F:\graduate\tongue\xinxueguan1397\qixu\qixu_labelxuerad.csv",
            "type": "clf", "num_classes": 2, "weights": 1.5,
        },
        "Dampness_syndrome": {
            "label_file": r"F:\graduate\tongue\xinxueguan1397\shizheng\shizheng_labelxuerad.csv",
            "type": "clf", "num_classes": 2, "weights": 1.3,
        },
        "CVD": {
            "label_file": r"F:\graduate\tongue\xinxueguan1397\xinxueguan\labelxuerad.csv",
            "type": "clf", "num_classes": 2, "weights": 1.2,
        },
    }
    return task_settings

def setup_training_params() -> dict:
    """
    设置训练超参数
    """
    params = dict(
        batch_size=64,
        epochs=500,  # 增加训练轮数
        init_lr=0.00001,  # 进一步降低初始学习率
        optimizer='adamw',
        weight_decay=0.3,  # 增加重正则化
        lr_scheduler="cosine_annealing",  # 使用余弦退火学习率调度器
        lr_patience=10,
        lr_factor=0.5,
        grad_clip=1.0,  # 启用梯度裁剪
        early_stop=100,  # 调整早停耐心值
        retrain="",
        model_root=MODEL_ROOT,
        add_date=False,
        iters_start=0,
        iters_verbose=1,
        save_per_epoch=True,
        trans_dim=256,
    )
    return params

# -------------------------------------------------
# 3. 检查数据平衡性并调整类别权重
# -------------------------------------------------
def compute_class_weights(label_file: str) -> dict:
    """
    计算类别权重
    """
    labels = pd.read_csv(label_file)["label"]
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
    return {i: weight for i, weight in enumerate(class_weights)}

# -------------------------------------------------
# 5. 启动
# -------------------------------------------------
if __name__ == "__main__":
    FEATURE_CSV = r"F:\graduate\tongue\xinxueguan1397\dlfeatures\data_vgg11_xuerad\data_vgg11_xuerad_norm.csv"
    MODEL_ROOT = r"D:\OnekeyPlatfrom\onekey_comp\comp2-结构化数据\final1"
    # 设置输入和输出
    df_feat, input_dim = setup_input_output(FEATURE_CSV, MODEL_ROOT)

    # 设置输入参数
    input_settings = setup_input_settings(input_dim)

    # 设置任务参数
    task_settings = setup_task_settings()

    # 设置训练超参数
    params = setup_training_params()

    # 计算类别权重
    for task in task_settings:
        task_settings[task]["class_weights"] = compute_class_weights(task_settings[task]["label_file"])
        logging.info(f"{task} 类别权重: {task_settings[task]['class_weights']}")

    # 合并所有参数
    params.update({
        "input_settings": input_settings,
        "task_settings": task_settings,
    })

    # 启动训练
    Args = namedtuple("Args", params)
    onekey_main(Args(**params))
