"""
推理引擎

模型加载 + 特征提取 + 推理，返回结构化诊断结果。

迁移来源: stream/stream_processor.py (StreamProcessor 类的模型加载与推理逻辑, L128-L251)
迁移日期: 2025-05-28
改动说明:
  - 将模型加载、特征提取、推理逻辑从 StreamProcessor 中独立出来
  - RingBuffer 和报警逻辑不在此模块中（已分别迁移至 ring_buffer.py 和 alerts/manager.py）
  - predict() 接受单段 1D 信号，返回纯净 dict
  - 算法逻辑零改动
"""

import sys
import os
import numpy as np

# 确保项目根目录在 sys.path 中，支持直接运行此文件
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from core.signal.features import extract_all_features
from core.signal.loader import LABEL_MAP


class InferenceEngine:
    """
    故障诊断推理引擎。

    使用:
        engine = InferenceEngine(model_dir="models/saved")
        engine.load_model("svm")   # 或 "cnn"

        signal = ...               # 1D numpy array, shape (window_size,)
        result = engine.predict(signal)
        # => {"fault_type": "内圈故障", "confidence": 0.9876,
        #     "probabilities": [0.001, 0.9876, 0.008, 0.0034]}

    支持的模型类型:
        - "svm": PCA + SVM (scikit-learn, joblib)
        - "cnn": 1D-CNN (PyTorch)
    """

    def __init__(self, model_dir="models/saved"):
        self.model_dir = model_dir

        # 标签映射: {label_id: label_name}
        self._label_dict = {v[0]: v[1] for v in LABEL_MAP.values()}

        # 模型类型
        self._model_type = None

        # SVM 组件
        self._svm = None
        self._svm_scaler = None
        self._svm_pca = None

        # CNN 组件
        self._cnn = None
        self._cnn_scaler = None

    # ═══════════════════════════════════════════════
    # 模型加载
    # ═══════════════════════════════════════════════
    def load_model(self, model_type: str):
        """
        加载预训练模型。

        参数:
            model_type: "svm" 或 "cnn"
        """
        import joblib
        import torch
        from models.architectures.cnn1d import FaultCNN, DEVICE

        self._model_type = model_type

        if model_type == "svm":
            self._svm = joblib.load(os.path.join(self.model_dir, "svm_model.pkl"))
            self._svm_scaler = joblib.load(os.path.join(self.model_dir, "svm_scaler.pkl"))
            self._svm_pca = joblib.load(os.path.join(self.model_dir, "svm_pca.pkl"))

        elif model_type == "cnn":
            config = torch.load(
                os.path.join(self.model_dir, "cnn_config.pt"),
                map_location=DEVICE, weights_only=False
            )
            self._cnn = FaultCNN(
                input_length=config["input_length"],
                num_classes=config["num_classes"]
            ).to(DEVICE)
            self._cnn.load_state_dict(
                torch.load(os.path.join(self.model_dir, "cnn_model.pt"),
                           map_location=DEVICE, weights_only=True)
            )
            self._cnn.eval()
            self._cnn_scaler = joblib.load(os.path.join(self.model_dir, "cnn_scaler.pkl"))

        else:
            raise ValueError(f"未知模型类型: {model_type}，支持 'svm' 或 'cnn'")

    @property
    def model_type(self):
        return self._model_type

    @property
    def labels(self):
        """返回标签映射 {id: name}。"""
        return dict(self._label_dict)

    # ═══════════════════════════════════════════════
    # 推理
    # ═══════════════════════════════════════════════
    def predict(self, signal: np.ndarray) -> dict:
        """
        对单段振动信号进行故障诊断。

        参数:
            signal: 1D numpy array, shape (window_size,)
                    或 2D (1, window_size)，会自动 reshape

        返回:
            {
                "fault_type":   str,    # 故障类型名称，如 "内圈故障"
                "confidence":   float,  # 置信度 0.0 ~ 1.0
                "probabilities": list,  # 各类别概率 [p0, p1, p2, p3]
            }
        """
        if self._model_type is None:
            raise RuntimeError("模型未加载。请先调用 load_model('svm') 或 load_model('cnn')。")

        import torch
        from models.architectures.cnn1d import DEVICE

        # 严格检查输入长度
        if signal.ndim != 1:
            raise ValueError(f"signal 必须是一维数组，当前维度: {signal.ndim}")
        if len(signal) != 1024:
            raise ValueError(
                f"signal 长度必须为 1024，当前长度: {len(signal)}。"
                f"模型训练时使用 window_size=1024，不支持动态长度输入。"
            )

        # 确保输入为 2D (1, window_size)
        signal_2d = signal.reshape(1, -1)

        # ---- 特征提取 ----
        feats, _ = extract_all_features(signal_2d)

        # ---- 模型推理 ----
        if self._model_type == "svm":
            scaled = self._svm_scaler.transform(feats)
            reduced = self._svm_pca.transform(scaled)
            pred = self._svm.predict(reduced)[0]
            probs = self._svm.predict_proba(reduced)[0]

        elif self._model_type == "cnn":
            scaled = self._cnn_scaler.transform(signal_2d)
            tensor = torch.FloatTensor(scaled).unsqueeze(1).to(DEVICE)
            with torch.no_grad():
                outputs = self._cnn(tensor)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
                pred = int(torch.argmax(outputs, dim=1).cpu().numpy()[0])

        return {
            "fault_type": self._label_dict[pred],
            "confidence": float(probs[pred]),
            "probabilities": [float(p) for p in probs],
        }


# ================================================================
# 自测
# ================================================================
if __name__ == "__main__":
    from core.signal.loader import build_dataset

    print("=" * 60)
    print("InferenceEngine 迁移验证")
    print("=" * 60)

    # 加载一批测试数据
    X, y, names = build_dataset(window_size=1024, step=512)
    print(f"  测试样本: {X.shape[0]}")

    # 每类取 3 个样本
    test_samples = []
    for label in range(4):
        idx = np.where(y == label)[0][:3]
        for i in idx:
            test_samples.append((X[i], label))

    # ---- 测试 SVM ----
    print("\n--- SVM 推理 ---")
    engine_svm = InferenceEngine()
    engine_svm.load_model("svm")
    svm_correct = 0
    for signal, true_label in test_samples:
        result = engine_svm.predict(signal)
        pred_name = result["fault_type"]
        expected_name = names[true_label]
        if pred_name == expected_name:
            svm_correct += 1
        print(f"  真实: {expected_name} | 预测: {pred_name} | "
              f"置信度: {result['confidence']:.4f} | "
              f"概率: {[f'{p:.4f}' for p in result['probabilities']]}")
    print(f"  SVM 准确率: {svm_correct}/{len(test_samples)}")

    # ---- 测试 CNN ----
    print("\n--- CNN 推理 ---")
    engine_cnn = InferenceEngine()
    engine_cnn.load_model("cnn")
    cnn_correct = 0
    for signal, true_label in test_samples:
        result = engine_cnn.predict(signal)
        pred_name = result["fault_type"]
        expected_name = names[true_label]
        if pred_name == expected_name:
            cnn_correct += 1
        print(f"  真实: {expected_name} | 预测: {pred_name} | "
              f"置信度: {result['confidence']:.4f} | "
              f"概率: {[f'{p:.4f}' for p in result['probabilities']]}")
    print(f"  CNN 准确率: {cnn_correct}/{len(test_samples)}")

    # ---- API 字段验证 ----
    sample = X[0]
    result = engine_svm.predict(sample)
    assert "fault_type" in result
    assert "confidence" in result
    assert "probabilities" in result
    assert isinstance(result["fault_type"], str)
    assert 0.0 <= result["confidence"] <= 1.0
    assert len(result["probabilities"]) == 4
    assert abs(sum(result["probabilities"]) - 1.0) < 1e-6

    print(f"\n  API 字段验证通过。迁移成功。")
