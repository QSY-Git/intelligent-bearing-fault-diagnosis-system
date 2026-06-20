"""
报警数据模型

定义报警等级枚举和报警记录数据结构。
"""

from dataclasses import dataclass, field
from enum import Enum
import uuid


class AlarmLevel(str, Enum):
    """报警等级。"""
    WARNING = "WARNING"      # 警告：需要关注，可继续运行
    CRITICAL = "CRITICAL"    # 严重：建议立即停机检查


@dataclass
class AlarmRecord:
    """
    报警记录。

    字段:
        id:           唯一标识 (UUID)
        timestamp:    报警触发时间 (Unix 秒)
        device_id:    设备标识
        fault_type:   故障类型名称
        confidence:   诊断置信度
        alarm_level:  报警等级 (WARNING / CRITICAL)
        message:      报警描述信息
    """
    timestamp: float
    device_id: str
    fault_type: str
    confidence: float
    alarm_level: AlarmLevel
    message: str
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "device_id": self.device_id,
            "fault_type": self.fault_type,
            "confidence": self.confidence,
            "alarm_level": self.alarm_level.value,
            "message": self.message,
        }
