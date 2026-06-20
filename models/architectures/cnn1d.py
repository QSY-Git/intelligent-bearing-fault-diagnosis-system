"""
1D-CNN 网络架构定义
网络结构：Conv1D -> BN -> ReLU -> MaxPool -> Conv1D -> BN -> ReLU ->
          MaxPool -> Flatten -> FC -> Dropout -> Linear

迁移来源: models/cnn_model.py (FaultCNN 类 + DEVICE 常量, L26-L79)
迁移日期: 2025-05-28
改动说明: 零逻辑改动。仅将 FaultCNN 类定义和 DEVICE 常量独立为架构模块。
         训练逻辑仍保留在 models/cnn_model.py 的 train_cnn() 中。
"""

import torch
import torch.nn as nn

# ================================================================
# 设备检测
# ================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
# 自测
# ================================================================
if __name__ == "__main__":
    print(f"设备: {DEVICE}")

    # 测试默认参数实例化
    model = FaultCNN(input_length=1024, num_classes=4)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 测试前向传播
    dummy_input = torch.randn(4, 1, 1024)  # batch=4, channels=1, length=1024
    with torch.no_grad():
        output = model(dummy_input)
    print(f"输入形状: {dummy_input.shape}")
    print(f"输出形状: {output.shape}")     # 期望 (4, 4)
    print("FaultCNN 架构模块迁移验证通过。")
