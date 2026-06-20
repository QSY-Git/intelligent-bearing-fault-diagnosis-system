"""
数据流模拟器
模拟工业现场传感器持续上报振动数据的场景。
从 CWRU 数据文件（.mat.txt 或 demo .npz）中按 chunk 迭代读取，
支持循环播放、故障切换、噪声注入。
"""

import numpy as np
from glob import glob
import os


class StreamSimulator:
    """
    振动数据流模拟器。

    使用方式:
        # 方式 1: 从 demo .npz 信号直接构造（Streamlit Cloud 推荐）
        from core.signal.demo_loader import load_demo_signal
        sig = load_demo_signal("内圈故障")
        sim = StreamSimulator.from_signal(sig, chunk_size=512)

        # 方式 2: 从 data/ 目录 .mat.txt 文件构造
        sim = StreamSimulator("data", fault_type="内圈故障", chunk_size=512)

        for chunk in sim.stream():
            process(chunk)

    参数:
        data_root:   CWRU 数据根目录
        fault_type:  模拟的故障类型（"正常"/"内圈故障"/"外圈故障"/"滚动体故障"）
        chunk_size:  每次推送的采样点数
        interval:    推送间隔（秒），仅用于标注，实际 sleep 由调用方控制
        noise_std:   高斯噪声标准差（0 表示不加噪），模拟传感器噪声
        loop:        是否循环播放
    """

    def __init__(
        self,
        data_root="data",
        fault_type="正常",
        chunk_size=512,
        interval=0.5,
        noise_std=0.0,
        loop=True,
    ):
        self.data_root = data_root
        self.fault_type = fault_type
        self.chunk_size = chunk_size
        self.interval = interval
        self.noise_std = noise_std
        self.loop = loop

        # 内部状态
        self._signal = None          # 当前加载的完整信号
        self._signal_length = 0
        self._cursor = 0             # 当前读取位置
        self._running = False
        self._total_pushed = 0       # 累计已推送点数

        # 初始加载
        self._load_signal()

    @classmethod
    def from_signal(cls, signal: np.ndarray, chunk_size=512,
                    interval=0.5, noise_std=0.0, loop=True):
        """从已有 numpy 信号数组直接构造模拟器（无需读取磁盘文件）。"""
        sim = cls.__new__(cls)
        sim.data_root = None
        sim.fault_type = "demo"
        sim.chunk_size = chunk_size
        sim.interval = interval
        sim.noise_std = noise_std
        sim.loop = loop

        sim._signal = signal.astype(np.float64)
        sim._signal_length = len(sim._signal)
        sim._cursor = 0
        sim._running = False
        sim._total_pushed = 0
        return sim

    # ---- 公开属性 ----
    @property
    def signal_length(self):
        return self._signal_length

    @property
    def progress(self):
        """返回当前播放进度 0.0~1.0"""
        if self._signal_length == 0:
            return 0.0
        return min(self._cursor / self._signal_length, 1.0)

    @property
    def total_pushed(self):
        return self._total_pushed

    @property
    def running(self):
        return self._running

    # ---- 核心方法 ----
    def _load_signal(self):
        """从磁盘加载指定故障类型的原始信号。"""
        from utils.data_loader import load_mat_txt, LABEL_MAP

        folder = None
        for fname, (lid, lname) in LABEL_MAP.items():
            if lname == self.fault_type:
                folder = fname
                break
        if folder is None:
            raise ValueError(f"未知故障类型: {self.fault_type}")

        folder_path = os.path.join(self.data_root, folder)
        files = glob(os.path.join(folder_path, "*.mat.txt"))
        if not files:
            raise FileNotFoundError(f"目录 {folder_path} 下未找到 .mat.txt 文件")

        # 加载第一个文件的信号作为主信号，其余文件拼接
        signals = []
        for fp in sorted(files):
            signals.append(load_mat_txt(fp))
        self._signal = np.concatenate(signals).astype(np.float64)
        self._signal_length = len(self._signal)
        self._cursor = 0
        self._total_pushed = 0

    def switch_fault(self, fault_type):
        """运行时切换故障类型，信号从头开始播放。"""
        old = self.fault_type
        self.fault_type = fault_type
        self._load_signal()
        return old

    def reset(self):
        """从头开始播放当前信号。"""
        self._cursor = 0

    def stream(self):
        """
        生成器函数，每次 yield 一块数据。

        使用:
            for chunk in sim.stream():
                ...
        或手动迭代:
            gen = sim.stream()
            chunk = next(gen)
        """
        self._running = True
        while self._running:
            if self._cursor + self.chunk_size > self._signal_length:
                if self.loop:
                    # 先产出尾部 + 头部拼接，保证 chunk 长度一致
                    tail = self._signal[self._cursor:]
                    wrap_len = self.chunk_size - len(tail)
                    head = self._signal[:wrap_len]
                    chunk = np.concatenate([tail, head])
                    self._cursor = wrap_len
                else:
                    # 不循环，产出剩余部分，长度可能不足 chunk_size
                    chunk = self._signal[self._cursor:]
                    self._running = False
            else:
                chunk = self._signal[self._cursor:self._cursor + self.chunk_size].copy()
                self._cursor += self.chunk_size

            # 可选加噪
            if self.noise_std > 0:
                chunk = chunk + np.random.normal(0, self.noise_std, size=chunk.shape)

            self._total_pushed += len(chunk)
            yield chunk

    def stop(self):
        """停止流。"""
        self._running = False

    def to_dict(self):
        """导出可序列化的状态，供 Streamlit session_state 使用。"""
        return {
            "fault_type": self.fault_type,
            "chunk_size": self.chunk_size,
            "interval": self.interval,
            "noise_std": self.noise_std,
            "cursor": self._cursor,
            "total_pushed": self._total_pushed,
            "running": self._running,
            "signal_length": self._signal_length,
            "loop": self.loop,
        }

    @classmethod
    def from_dict(cls, state, data_root="data"):
        """从字典恢复模拟器状态。"""
        sim = cls(
            data_root=data_root,
            fault_type=state["fault_type"],
            chunk_size=state["chunk_size"],
            interval=state["interval"],
            noise_std=state["noise_std"],
            loop=state.get("loop", True),
        )
        sim._cursor = state["cursor"]
        sim._total_pushed = state["total_pushed"]
        sim._running = state["running"]
        return sim
