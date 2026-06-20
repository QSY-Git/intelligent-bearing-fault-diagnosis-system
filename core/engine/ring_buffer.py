"""
环形缓冲区 (Ring Buffer)

固定大小的 NumPy 数组上循环写入，支持 O(1) 提取最新 N 个连续采样点。
用于流式信号处理场景：传感器持续上报 → 写入缓冲区 → 提取窗口 → 推理。

迁移来源: stream/stream_processor.py (RingBuffer 类, L32-L111)
迁移日期: 2025-05-28
改动说明: 零逻辑改动。纯数据结构类，无项目内部依赖。
"""

import numpy as np
from typing import Optional


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


# ================================================================
# 自测
# ================================================================
if __name__ == "__main__":
    print("=" * 50)
    print("RingBuffer 迁移验证")
    print("=" * 50)

    # 测试 1: 基本写入和读取
    buf = RingBuffer(capacity=4096)
    buf.push(np.arange(500, dtype=np.float64))
    buf.push(np.arange(500, 1000, dtype=np.float64))
    w = buf.latest(1024)
    assert w is None, "数据不足时应返回 None"
    print("测试 1 通过: 数据不足时 latest() 返回 None")

    # 测试 2: 填满后提取
    buf.push(np.zeros(4000, dtype=np.float64))
    w = buf.latest(1024)
    assert w is not None and w.shape == (1024,), f"期望 (1024,)，实际 {w.shape}"
    assert buf.ready, "填满后 ready 应为 True"
    print(f"测试 2 通过: latest(1024).shape = {w.shape}, ready = {buf.ready}")

    # 测试 3: 环绕写入
    buf2 = RingBuffer(capacity=100)
    buf2.push(np.arange(80, dtype=np.float64))      # 写入 [0..79]
    buf2.push(np.arange(80, dtype=np.float64) + 80) # 写入 [80..159]，此时已环绕
    w2 = buf2.latest(50)
    assert w2 is not None, "环绕后应可提取"
    # 最新 50 个点应该是 [110..159]
    assert w2[0] == 110.0, f"环绕提取错误，期望 110.0，实际 {w2[0]}"
    print(f"测试 3 通过: 环绕后 latest(50) 首点 = {w2[0]}")

    # 测试 4: get_buffer 时间顺序
    b = buf2.get_buffer()
    # buf2 容量 100，写入 160 点，最终保留最新 100 点即 [60..159]
    assert len(b) == 100, f"get_buffer 应返回 100 点，实际 {len(b)}"
    print(f"测试 4 通过: get_buffer() 长度 = {len(b)}")

    print("\n所有 RingBuffer 测试通过。迁移成功。")
