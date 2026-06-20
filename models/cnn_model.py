"""
深度学习模型：1D-CNN 故障诊断
网络结构：Conv1D -> BN -> ReLU -> MaxPool -> Conv1D -> BN -> ReLU ->
          MaxPool -> Flatten -> FC -> Dropout -> Softmax

⚠️ DEPRECATED: 此脚本存在数据泄漏问题（先切窗后随机划分 + 全量数据fit scaler）。
   请使用 tools/run_experiments.py 进行可信实验。
   此脚本仅保留用于快速原型验证。
"""

import os
import sys
import warnings
warnings.warn(
    "models/cnn_model.py is deprecated due to data leakage. "
    "Use tools/run_experiments.py for trustworthy experiments.",
    DeprecationWarning, stacklevel=2
)
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader import build_dataset


# ================================================================
# 设备检测
# ================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {DEVICE}")


# ================================================================
# 1D-CNN 网络定义
# ================================================================
class FaultCNN(nn.Module):
    """
    一维卷积神经网络，用于轴承振动信号分类。
    输入: (batch, 1, signal_length)
    输出: (batch, num_classes)
    """
    def __init__(self, input_length=1024, num_classes=4):
        super(FaultCNN, self).__init__()

        # ---- 卷积层1 ----
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16,
                               kernel_size=64, stride=2, padding=31)
        self.bn1   = nn.BatchNorm1d(16)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        # ---- 卷积层2 ----
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32,
                               kernel_size=32, stride=2, padding=15)
        self.bn2   = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        # ---- 全连接层 ----
        # 动态计算展平后的特征维度
        self._to_linear = None
        self._compute_linear_dim(input_length)
        self.dropout = nn.Dropout(0.5)
        self.fc1     = nn.Linear(self._to_linear, 128)
        self.fc2     = nn.Linear(128, num_classes)

    def _compute_linear_dim(self, input_length):
        """通过前向推算计算展平后的维度，避免硬编码。"""
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_length)
            x = self.pool1(torch.relu(self.bn1(self.conv1(dummy))))
            x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
            self._to_linear = x.view(1, -1).size(1)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)        # 展平
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)                  # 输出logits（CrossEntropyLoss内置Softmax）
        return x


# ================================================================
# 训练函数
# ================================================================
def train_cnn(data_root="data", window_size=1024, step=512,
              epochs=50, batch_size=32, lr=0.001,
              save_dir="models/saved"):
    """
    训练1D-CNN模型并保存。
    返回:
        model:      训练好的PyTorch模型
        scaler:     标准化器
        history:    {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    """
    print("=" * 60)
    print("阶段1: 加载数据")
    print("=" * 60)
    X_raw, y, label_names = build_dataset(data_root, window_size, step)
    num_classes = len(label_names)

    # ----------------------------------------------------------
    # 标准化（全局逐样本标准化）
    # ----------------------------------------------------------
    print("\n阶段2: 数据预处理")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)  # 逐样本标准化

    # 转换为 (n_samples, 1, window_size) 供CNN输入
    X_tensor = torch.FloatTensor(X_scaled).unsqueeze(1)
    y_tensor = torch.LongTensor(y)

    # 划分训练/验证集
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tensor, y_tensor, test_size=0.3, random_state=42, stratify=y
    )
    train_loader = DataLoader(TensorDataset(X_tr, y_tr),
                              batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val, y_val),
                              batch_size=batch_size, shuffle=False)
    print(f"  训练集: {len(X_tr)} 样本, 验证集: {len(X_val)} 样本")

    # ----------------------------------------------------------
    # 模型、损失函数、优化器
    # ----------------------------------------------------------
    print("\n阶段3: 构建CNN模型")
    model = FaultCNN(input_length=window_size, num_classes=num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    print(model)

    # ----------------------------------------------------------
    # 训练循环
    # ----------------------------------------------------------
    print("\n阶段4: 开始训练")
    print("-" * 60)
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(epochs):
        # ---- 训练阶段 ----
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        scheduler.step()
        epoch_loss = train_loss / total
        epoch_acc  = correct / total
        history["train_loss"].append(epoch_loss)
        history["train_acc"].append(epoch_acc)

        # ---- 验证阶段 ----
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        epoch_val_loss = val_loss / total
        epoch_val_acc  = correct / total
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | "
                  f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}")

    final_val_acc = history["val_acc"][-1]
    print(f"\n  最终验证集准确率: {final_val_acc:.4f} ({final_val_acc*100:.2f}%)")

    # ----------------------------------------------------------
    # 保存模型
    # ----------------------------------------------------------
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, "cnn_model.pt"))
    joblib.dump(scaler, os.path.join(save_dir, "cnn_scaler.pkl"))
    # 保存训练历史用于Streamlit绘图
    joblib.dump(history, os.path.join(save_dir, "cnn_history.pkl"))
    # 保存模型参数以便Streamlit重建
    torch.save({
        "input_length": window_size,
        "num_classes":  num_classes,
    }, os.path.join(save_dir, "cnn_config.pt"))
    print(f"模型已保存至: {os.path.abspath(save_dir)}")

    return model, scaler, history


if __name__ == "__main__":
    train_cnn(epochs=50)
