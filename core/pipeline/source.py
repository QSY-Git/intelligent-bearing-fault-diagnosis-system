"""
数据源抽象基类

定义工业数据接入的统一接口。所有数据源（文件、MQTT、OPC-UA、WebSocket 等）
必须实现此接口，确保上层调用方无需关心底层传输协议。

使用示例:
    from core.pipeline.sources.mqtt_client import MQTTDataSource

    source = MQTTDataSource(host="localhost", port=1883, topic="bearing/vibration")
    source.connect()
    while True:
        chunk = source.read()      # 阻塞直到收到数据
        engine.predict(chunk)      # 送入推理引擎
    source.disconnect()
"""

from abc import ABC, abstractmethod
import numpy as np


class DataSource(ABC):
    """
    工业数据源抽象基类。

    子类必须实现:
        connect()    — 建立连接
        disconnect() — 断开连接
        read()       — 读取一个数据块 (返回 np.ndarray)

    可选覆盖:
        is_connected — 连接状态属性
    """

    @abstractmethod
    def connect(self) -> bool:
        """
        建立与数据源的连接。

        返回:
            True 如果连接成功，False 如果失败
        """
        ...

    @abstractmethod
    def disconnect(self) -> None:
        """
        断开与数据源的连接，释放资源。
        """
        ...

    @abstractmethod
    def read(self) -> np.ndarray:
        """
        从数据源读取一个数据块。

        返回:
            1D numpy float64 数组，振动信号采样点
            如果连接断开或无可读数据，可抛出 RuntimeError
        """
        ...

    @property
    def is_connected(self) -> bool:
        """是否处于连接状态。子类应覆盖此属性。"""
        return False

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
        return False
