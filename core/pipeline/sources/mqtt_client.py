"""
MQTT 数据源实现

通过 MQTT 协议订阅工业传感器振动数据，实现 DataSource 接口。
支持 JSON 格式消息解析、自动重连、阻塞读取。

消息格式:
    {
        "timestamp": 1716883200.123,
        "device_id": "bearing_001",
        "signal": [0.012, -0.003, ...]
    }

使用:
    source = MQTTDataSource(host="localhost", port=1883, topic="bearing/vibration")
    source.connect()
    chunk = source.read()   # 阻塞等待下一条消息
    source.disconnect()
"""

import sys
import os
import json
import time
import queue
import threading
import numpy as np

# 确保项目根目录在 sys.path 中
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from core.pipeline.source import DataSource

try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False


class MQTTDataSource(DataSource):
    """
    MQTT 振动数据源。

    订阅指定 topic，接收 JSON 格式的振动信号数组，
    通过 read() 方法阻塞返回。

    参数:
        host:       MQTT Broker 地址
        port:       MQTT Broker 端口
        topic:      订阅主题
        client_id:  客户端 ID（可选）
        keepalive:  心跳间隔（秒）
        qos:        消息质量 0/1/2
        timeout:    read() 超时时间（秒），None 表示无限等待
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 1883,
        topic: str = "bearing/vibration",
        client_id: str = "diagnosis_agent",
        keepalive: int = 60,
        qos: int = 1,
        timeout: float = None,
    ):
        if not MQTT_AVAILABLE:
            raise ImportError(
                "paho-mqtt 未安装。请运行: pip install paho-mqtt"
            )

        self.host = host
        self.port = port
        self.topic = topic
        self.client_id = client_id
        self.keepalive = keepalive
        self.qos = qos
        self.timeout = timeout

        # 内部状态
        self._client: mqtt.Client = None
        self._connected = False
        self._message_queue = queue.Queue(maxsize=1000)
        self._lock = threading.Lock()

    # ═══════════════════════════════════════════════
    # DataSource 接口实现
    # ═══════════════════════════════════════════════

    def connect(self) -> bool:
        """
        连接到 MQTT Broker 并订阅主题。

        返回:
            True 连接成功，False 失败
        """
        if self._connected:
            return True

        try:
            self._client = mqtt.Client(
                client_id=self.client_id,
                protocol=mqtt.MQTTv311,
            )
            self._client.on_connect = self._on_connect
            self._client.on_message = self._on_message
            self._client.on_disconnect = self._on_disconnect

            self._client.connect(self.host, self.port, self.keepalive)
            # 在后台线程启动网络循环
            self._client.loop_start()

            # 等待连接建立（最多 5 秒）
            waited = 0
            while not self._connected and waited < 5:
                time.sleep(0.1)
                waited += 0.1

            if not self._connected:
                self._client.loop_stop()
                return False

            return True

        except Exception as e:
            self._connected = False
            raise ConnectionError(
                f"无法连接到 MQTT Broker {self.host}:{self.port}: {e}"
            )

    def disconnect(self) -> None:
        """断开 MQTT 连接并释放资源。"""
        self._connected = False
        if self._client is not None:
            try:
                self._client.unsubscribe(self.topic)
                self._client.loop_stop()
                self._client.disconnect()
            except Exception:
                pass
            self._client = None

    def read(self) -> np.ndarray:
        """
        阻塞读取一条振动信号数据。

        返回:
            1D numpy float64 数组，signal 字段的值

        异常:
            RuntimeError: 如果未连接
            queue.Empty: 如果超时
        """
        if not self._connected:
            raise RuntimeError("MQTT 未连接，请先调用 connect()")

        try:
            msg = self._message_queue.get(block=True, timeout=self.timeout)
        except queue.Empty:
            raise TimeoutError(
                f"等待 MQTT 消息超时 ({self.timeout}秒)，topic={self.topic}"
            )

        signal = np.array(msg.get("signal", []), dtype=np.float64)

        if len(signal) == 0:
            raise ValueError("MQTT 消息中 signal 字段为空")

        return signal

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def queue_size(self) -> int:
        """当前消息队列中的待消费消息数。"""
        return self._message_queue.qsize()

    # ═══════════════════════════════════════════════
    # MQTT 回调
    # ═══════════════════════════════════════════════

    def _on_connect(self, client, userdata, flags, rc):
        """连接成功回调。"""
        if rc == 0:
            self._connected = True
            client.subscribe(self.topic, qos=self.qos)
            print(f"[MQTT] 已连接 {self.host}:{self.port}，订阅 {self.topic}")
        else:
            self._connected = False
            print(f"[MQTT] 连接失败，返回码: {rc}")

    def _on_message(self, client, userdata, msg):
        """收到消息回调：解析 JSON 并入队。"""
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
            # 防止队列满时阻塞 — 丢弃最旧消息
            if self._message_queue.full():
                try:
                    self._message_queue.get_nowait()
                except queue.Empty:
                    pass
            self._message_queue.put_nowait(payload)
        except json.JSONDecodeError as e:
            print(f"[MQTT] JSON 解析失败: {e}")
        except queue.Full:
            print("[MQTT] 消息队列已满，丢弃消息")

    def _on_disconnect(self, client, userdata, rc):
        """断开连接回调。"""
        self._connected = False
        if rc != 0:
            print(f"[MQTT] 异常断开 (rc={rc})，将尝试自动重连")
        else:
            print("[MQTT] 已断开连接")


# ================================================================
# 自测（需要本地 MQTT Broker）
# ================================================================
if __name__ == "__main__":
    print("MQTTDataSource 模块加载成功。")
    print("完整测试请配合 MQTT Broker 和 mqtt_simulator 一起运行。")
    print()
    print("测试步骤:")
    print("  1. 启动 MQTT Broker:  mosquitto -v")
    print("  2. 启动模拟器:      python tools/mqtt_simulator.py")
    print("  3. 运行本测试:      python core/pipeline/sources/mqtt_client.py")
