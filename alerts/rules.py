"""
报警规则引擎

定义报警规则的抽象基类和具体实现。
每条规则接收诊断上下文，返回是否需要触发报警及报警等级。

规则设计:
    Rule 1: 连续 ≥3 次同类型故障 → WARNING
    Rule 2: 连续 ≥5 次同类型故障 → CRITICAL
    Rule 3: 置信度 > 0.95 → 立即 WARNING
    Rule 4: 置信度 > 0.99 → 立即 CRITICAL

优先级: CRITICAL > WARNING (多条触发时取最高等级)
"""

from abc import ABC, abstractmethod
from typing import Optional
from alerts.models import AlarmLevel


class AlarmRule(ABC):
    """报警规则抽象基类。"""

    @abstractmethod
    def evaluate(self, ctx: "RuleContext") -> Optional[tuple[AlarmLevel, str]]:
        """
        评估规则是否触发。

        参数:
            ctx: 规则上下文，包含当前诊断结果和历史状态

        返回:
            (AlarmLevel, message) 如果触发报警
            None 如果不触发
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """规则名称。"""
        ...


class RuleContext:
    """
    规则评估上下文。

    由 AlarmManager 在每次收到诊断结果时构建并传递给各规则。

    包含当前诊断信息 + 设备历史状态（连续故障计数等）。
    """

    def __init__(
        self,
        device_id: str,
        fault_type: str,
        confidence: float,
        is_fault: bool,                  # 是否故障（非正常）
        consecutive_count: int = 0,      # 当前连续同类型故障次数
        total_window_count: int = 0,     # 累计诊断次数
    ):
        self.device_id = device_id
        self.fault_type = fault_type
        self.confidence = confidence
        self.is_fault = is_fault
        self.consecutive_count = consecutive_count
        self.total_window_count = total_window_count


# ═══════════════════════════════════════════════
# 规则实现
# ═══════════════════════════════════════════════

class ConsecutiveFaultRule(AlarmRule):
    """
    连续故障计数规则。

    连续 ≥N_warning 次 → WARNING
    连续 ≥N_critical 次 → CRITICAL
    """

    def __init__(self, warning_count: int = 3, critical_count: int = 5):
        self._warning_count = warning_count
        self._critical_count = critical_count

    @property
    def name(self) -> str:
        return f"连续故障 ≥{self._warning_count}/{self._critical_count}"

    def evaluate(self, ctx: RuleContext) -> Optional[tuple[AlarmLevel, str]]:
        if not ctx.is_fault:
            return None

        if ctx.consecutive_count >= self._critical_count:
            return (
                AlarmLevel.CRITICAL,
                f"连续 {ctx.consecutive_count} 次检测到 {ctx.fault_type}，触发严重报警"
            )
        elif ctx.consecutive_count >= self._warning_count:
            return (
                AlarmLevel.WARNING,
                f"连续 {ctx.consecutive_count} 次检测到 {ctx.fault_type}，触发警告"
            )
        return None


class HighConfidenceRule(AlarmRule):
    """
    高置信度即时报警规则。

    置信度 > warning_threshold   → 立即 WARNING
    置信度 > critical_threshold  → 立即 CRITICAL
    """

    def __init__(self, warning_threshold: float = 0.95, critical_threshold: float = 0.99):
        self._warning_threshold = warning_threshold
        self._critical_threshold = critical_threshold

    @property
    def name(self) -> str:
        return f"高置信度 >{self._warning_threshold}/{self._critical_threshold}"

    def evaluate(self, ctx: RuleContext) -> Optional[tuple[AlarmLevel, str]]:
        if not ctx.is_fault:
            return None

        if ctx.confidence > self._critical_threshold:
            return (
                AlarmLevel.CRITICAL,
                f"{ctx.fault_type} 置信度 {ctx.confidence:.2%} > {self._critical_threshold}，触发严重报警"
            )
        elif ctx.confidence > self._warning_threshold:
            return (
                AlarmLevel.WARNING,
                f"{ctx.fault_type} 置信度 {ctx.confidence:.2%} > {self._warning_threshold}，触发警告"
            )
        return None
