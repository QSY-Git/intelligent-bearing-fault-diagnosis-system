"""
数据加载模块
功能：读取 CWRU 轴承数据（.mat.txt 格式，实为二进制 MAT 文件），
      遍历目录构建带标签的样本集。

迁移来源: utils/data_loader.py (LABEL_MAP, load_mat_txt, build_dataset)
迁移日期: 2025-05-28
改动说明:
  - LABEL_MAP、load_mat_txt、build_dataset 从原文件完整迁移
  - sliding_window 已独立为 core/signal/transform.py，此处从新路径导入
  - 算法逻辑零改动
"""

import sys
import os
import numpy as np
import scipy.io as sio
from glob import glob

# 确保项目根目录在 sys.path 中，支持直接运行此文件
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from core.signal.transform import sliding_window

# ============================================================
# 故障类型与目录映射
# ============================================================
# 目录名 -> (标签ID, 中文名称)
LABEL_MAP = {
    "正常":   (0, "正常"),
    "内圈故障": (1, "内圈故障"),
    "外圈故障": (2, "外圈故障"),
    "滚动体故障": (3, "滚动体故障"),
}


def load_mat_txt(filepath):
    """
    读取 .mat.txt 文件（本质是MATLAB v5二进制文件）。
    参数:
        filepath: 文件路径
    返回:
        data: 1D NumPy数组，振动信号
    """
    mat_data = sio.loadmat(filepath)
    # MAT文件中通常有多个键，过滤掉MATLAB元数据键（以'__'开头）
    data_keys = [k for k in mat_data.keys() if not k.startswith("__")]
    if not data_keys:
        raise ValueError(f"未找到有效数据键: {filepath}")
    # 取第一个有效键对应的数据，并展平为1D数组
    data = mat_data[data_keys[0]].flatten().astype(np.float64)
    return data


def build_dataset(data_root="data", window_size=1024, step=512):
    """
    遍历data目录下所有类别文件夹，加载全部.mat.txt文件，
    执行滑动窗口切片，构建完整数据集。
    参数:
        data_root:    数据根目录
        window_size:  窗口大小
        step:         滑动步长
    返回:
        X: shape (n_samples, window_size) 的NumPy数组
        y: shape (n_samples,) 的标签数组
        label_names:  list of str, 标签名称列表
    """
    X_list, y_list = [], []

    for folder_name, (label_id, label_name) in LABEL_MAP.items():
        folder_path = os.path.join(data_root, folder_name)
        if not os.path.isdir(folder_path):
            print(f"  [警告] 目录不存在，跳过: {folder_path}")
            continue

        mat_files = glob(os.path.join(folder_path, "*.mat.txt"))
        print(f"  类别 [{label_name}]: 找到 {len(mat_files)} 个文件")

        for fpath in mat_files:
            raw = load_mat_txt(fpath)
            segs = sliding_window(raw, window_size, step)
            X_list.append(segs)
            y_list.append(np.full(segs.shape[0], label_id))

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    label_names = [LABEL_MAP[k][1] for k in LABEL_MAP.keys()]
    print(f"  总样本数: {X.shape[0]}, 特征维度: {X.shape[1]}")
    return X, y, label_names


if __name__ == "__main__":
    print("=" * 50)
    print("数据加载模块迁移验证")
    print("=" * 50)
    X, y, names = build_dataset()
    for i, name in enumerate(names):
        count = np.sum(y == i)
        print(f"  {name}: {count} 个样本")
