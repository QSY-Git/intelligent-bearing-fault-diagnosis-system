"""
传统机器学习模型：PCA + SVM 故障诊断
训练流程：数据加载 -> 特征提取 -> 标准化 -> 分类器训练 -> 模型保存

⚠️ DEPRECATED: 此脚本存在数据泄漏问题（先切窗后随机划分）。
   请使用 tools/run_experiments.py 进行可信实验。
   此脚本仅保留用于快速原型验证。
"""

import os
import sys
import warnings
warnings.warn(
    "models/svm_model.py is deprecated due to data leakage. "
    "Use tools/run_experiments.py for trustworthy experiments.",
    DeprecationWarning, stacklevel=2
)
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 将项目根目录加入路径，便于直接运行此文件时导入utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader import build_dataset
from utils.features import extract_all_features


def train_svm_model(data_root="data", window_size=1024, step=512,
                    save_dir="models/saved"):
    """
    训练SVM分类器并保存模型。
    参数:
        data_root:    数据目录路径
        window_size:  滑动窗口大小
        step:         窗口滑动步长
        save_dir:     模型保存目录
    返回:
        svm:          训练好的SVM模型
        scaler:       标准化器
        pca:          PCA降维器
        accuracy:     测试集准确率
        history:      训练历史（SVM无迭代过程，此处仅保存最终指标）
    """
    print("=" * 60)
    print("阶段1: 加载数据并提取特征")
    print("=" * 60)
    X_raw, y, label_names = build_dataset(data_root, window_size, step)
    X_feat, feat_names = extract_all_features(X_raw)

    print(f"\n特征维度: {X_feat.shape}")
    print(f"标签分布: {np.bincount(y)}")

    # ----------------------------------------------------------
    # 划分训练集和测试集（stratify保证类别均衡）
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("阶段2: 训练/测试集划分")
    print("=" * 60)
    X_train, X_test, y_train, y_test = train_test_split(
        X_feat, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"  训练集: {X_train.shape[0]} 样本")
    print(f"  测试集: {X_test.shape[0]} 样本")

    # ----------------------------------------------------------
    # 标准化 + PCA（保留所有主成分用于分类，min(样本数, 特征数)）
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("阶段3: 特征标准化 + PCA")
    print("=" * 60)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    n_pca = min(X_train_scaled.shape[1], X_train_scaled.shape[0])
    pca = PCA(n_components=n_pca)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    print(f"  PCA保留成分数: {n_pca}")
    print(f"  累积解释方差: {np.sum(pca.explained_variance_ratio_):.4f}")

    # ----------------------------------------------------------
    # SVM训练（RBF核，自动调优gamma）
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("阶段4: SVM分类器训练")
    print("=" * 60)
    svm = SVC(kernel="rbf", C=10, gamma="scale", probability=True,
              random_state=42)
    svm.fit(X_train_pca, y_train)

    # 评估
    y_pred = svm.predict(X_test_pca)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n  测试集准确率: {acc:.4f} ({acc*100:.2f}%)")
    print("\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=label_names))
    print("混淆矩阵:")
    print(confusion_matrix(y_test, y_pred))

    # ----------------------------------------------------------
    # 保存模型及预处理组件
    # ----------------------------------------------------------
    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(svm,    os.path.join(save_dir, "svm_model.pkl"))
    joblib.dump(scaler, os.path.join(save_dir, "svm_scaler.pkl"))
    joblib.dump(pca,    os.path.join(save_dir, "svm_pca.pkl"))
    # 保存PCA降维至2D的模型（用于可视化）
    pca_2d = PCA(n_components=2)
    pca_2d.fit(X_train_scaled)
    joblib.dump(pca_2d, os.path.join(save_dir, "svm_pca_2d.pkl"))
    print(f"\n模型已保存至: {os.path.abspath(save_dir)}")

    return svm, scaler, pca, acc


if __name__ == "__main__":
    train_svm_model()
