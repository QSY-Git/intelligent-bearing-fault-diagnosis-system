"""
实时推理引擎
环形缓冲区 + 滑动窗口提取 + 在线推理。
复用现有特征提取和模型，只改变数据流入方式。
"""

import time
from dataclasses import dataclass, asdict
from typing import Optional
import numpy as np

from utils.features import extract_all_features, compute_fft_spectrum


@dataclass
class DiagnosisRecord:
    """单次诊断结果记录。"""
    timestamp: float           # Unix 时间戳
    predicted_label: int       # 预测标签 ID
    predicted_name: str        # 预测标签名称
    confidence: float          # 置信度 (0~1)
    all_probs: list            # 各类别概率分布
    mean_val: float            # 时域均值
    rms: float                 # 均方根
    kurtosis: float            # 峭度
    skewness: float            # 偏度
    dom_freq: float            # 主频 (Hz)
    spectral_energy: float     # 频谱能量
    alarm: bool = False        # 是否触发报警


class RingBuffer:
    """
    环形缓冲区。

    在固定大小的数组上循环写入，支持提取最新的 window_size 个点。

    使用:
        buf = RingBuffer(capacity=4096)
        buf.push(new_data)                    # 追加数据
        window = buf.latest(1024)             # 取最新 1024 点
    """

    def __init__(self, capacity=4096):
        self.capacity = capacity
        self._buffer = np.zeros(capacity, dtype=np.float64)
        self._write_ptr = 0         # 下一个写入位置
        self._total_written = 0     # 累计写入总数
        self._ready = False         # 是否已积累足够数据

    def push(self, data: np.ndarray):
        """向缓冲区追加数据。"""
        n = len(data)
        if n > self.capacity:
            # 数据块超过容量，只保留最后 capacity 个点
            data = data[-self.capacity:]
            n = self.capacity

        write_end = self._write_ptr + n
        if write_end <= self.capacity:
            self._buffer[self._write_ptr:write_end] = data
        else:
            # 环绕写入
            first_part = self.capacity - self._write_ptr
            self._buffer[self._write_ptr:] = data[:first_part]
            self._buffer[:write_end - self.capacity] = data[first_part:]

        self._write_ptr = write_end % self.capacity
        self._total_written += n

    @property
    def ready(self):
        """缓冲区是否已积累至少一个窗口的数据。"""
        return self._total_written >= self.capacity

    @property
    def total_written(self):
        return self._total_written

    def latest(self, window_size: int) -> Optional[np.ndarray]:
        """
        提取缓冲区中最新 window_size 个点（连续信号段）。
        返回 None 如果数据不足。
        """
        if self._total_written < window_size:
            return None

        if window_size > self.capacity:
            window_size = self.capacity

        # 最新 window_size 个连续点在缓冲区中的位置
        start = (self._write_ptr - window_size) % self.capacity
        end = self._write_ptr

        if start < end:
            return self._buffer[start:end].copy()
        else:
            # 环绕
            return np.concatenate([
                self._buffer[start:],
                self._buffer[:end]
            ])

    def get_buffer(self) -> np.ndarray:
        """返回当前缓冲区内容（用于波形显示，按时间顺序排列）。"""
        if self._total_written < self.capacity:
            return self._buffer[:self._write_ptr].copy()
        return np.concatenate([
            self._buffer[self._write_ptr:],
            self._buffer[:self._write_ptr]
        ])


class StreamProcessor:
    """
    流式推理处理器。

    使用:
        proc = StreamProcessor(window_size=1024, buffer_capacity=4096)
        proc.load_model("svm")   # 或 "cnn"

        for chunk in simulator.stream():
            record = proc.process_chunk(chunk)
            if record:
                print(f"诊断: {record.predicted_name}, 置信度: {record.confidence:.2%}")
    """

    def __init__(self, window_size=1024, buffer_capacity=4096, fs=12000,
                 model_dir="models/saved"):
        self.window_size = window_size
        self.fs = fs
        self.model_dir = model_dir
        self.buffer = RingBuffer(capacity=buffer_capacity)

        # 模型组件（延迟加载）
        self._model_type = None
        self._svm = None
        self._svm_scaler = None
        self._svm_pca = None
        self._cnn = None
        self._cnn_scaler = None

        # 报警状态
        self._alarm_counter = {}      # label -> 连续出现次数
        self._alarm_threshold = 3     # 连续 N 次触发报警

    # ---- 模型加载 ----
    def load_model(self, model_type: str):
        """
        加载预训练模型。

        参数:
            model_type: "svm" 或 "cnn"
        """
        import joblib
        import torch
        import os
        from models.cnn_model import FaultCNN, DEVICE

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
            raise ValueError(f"未知模型类型: {model_type}")

    @property
    def model_type(self):
        return self._model_type

    # ---- 核心推理 ----
    def process_chunk(self, chunk: np.ndarray) -> Optional[DiagnosisRecord]:
        """
        处理一个数据块：推入缓冲区 → 提取窗口 → 推理。

        参数:
            chunk: 新到达的数据 (shape: (n,))

        返回:
            DiagnosisRecord 如果缓冲区已积累足够数据进行一次推理
            None 如果数据还不够一个窗口
        """
        import torch
        from models.cnn_model import DEVICE

        self.buffer.push(chunk)

        window = self.buffer.latest(self.window_size)
        if window is None:
            return None

        # 特征提取（用于记录和 SVM）
        feats, feat_names = extract_all_features(window.reshape(1, -1), fs=self.fs)
        td_feats = feats[0]

        # 模型推理
        if self._model_type == "svm":
            scaled = self._svm_scaler.transform(feats)
            reduced = self._svm_pca.transform(scaled)
            pred = self._svm.predict(reduced)[0]
            probs = self._svm.predict_proba(reduced)[0]
        elif self._model_type == "cnn":
            scaled = self._cnn_scaler.transform(window.reshape(1, -1))
            tensor = torch.FloatTensor(scaled).unsqueeze(1).to(DEVICE)
            with torch.no_grad():
                outputs = self._cnn(tensor)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
                pred = int(torch.argmax(outputs, dim=1).cpu().numpy()[0])

        # 标签名称
        from utils.data_loader import LABEL_MAP
        label_names = {v[0]: v[1] for v in LABEL_MAP.values()}
        pred_name = label_names.get(pred, f"未知({pred})")

        confidence = float(probs[pred])

        # 报警判断
        alarm = self._check_alarm(pred)

        record = DiagnosisRecord(
            timestamp=time.time(),
            predicted_label=int(pred),
            predicted_name=pred_name,
            confidence=confidence,
            all_probs=[float(p) for p in probs],
            mean_val=float(td_feats[0]),
            rms=float(td_feats[1]),
            kurtosis=float(td_feats[2]),
            skewness=float(td_feats[3]),
            dom_freq=float(td_feats[4]),
            spectral_energy=float(td_feats[5]),
            alarm=alarm,
        )
        return record

    def _check_alarm(self, label: int) -> bool:
        """检查是否触发报警（连续 N 次同一故障）。"""
        if label == 0:  # 正常状态重置
            self._alarm_counter.clear()
            return False

        self._alarm_counter[label] = self._alarm_counter.get(label, 0) + 1
        # 清除其他标签的计数
        for k in list(self._alarm_counter.keys()):
            if k != label:
                self._alarm_counter[k] = 0

        return self._alarm_counter[label] >= self._alarm_threshold

    def set_alarm_threshold(self, n: int):
        """设置报警连续次数阈值。"""
        self._alarm_threshold = n
        self._alarm_counter.clear()

    def get_alarm_count(self, label: int) -> int:
        return self._alarm_counter.get(label, 0)
