"""
报警管理器

接收诊断结果 → 维护设备状态 → 评估规则 → 生成报警记录 → SQLite 持久化。

设计原则:
    - 独立于 InferenceEngine，通过依赖注入集成
    - 规则可插拔：在 __init__ 中注册 AlarmRule 列表
    - 设备状态维护：记录每台设备的连续故障计数
    - 冷却期：同一故障类型在冷却期内不重复报警

使用:
    from alerts.manager import AlarmManager
    from alerts.rules import ConsecutiveFaultRule, HighConfidenceRule

    mgr = AlarmManager(
        device_id="bearing_001",
        rules=[ConsecutiveFaultRule(3, 5), HighConfidenceRule(0.95, 0.99)],
        db_path="alarms.db",
        cooldown_seconds=60,
    )

    # 收到诊断结果后调用
    alarm = mgr.evaluate(
        fault_type="内圈故障",
        confidence=0.987,
        is_fault=True,
    )
    if alarm:
        print(f"[{alarm.alarm_level.value}] {alarm.message}")
"""

import sys
import os
import time
import sqlite3
import json
from typing import List, Optional
from dataclasses import asdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from alerts.models import AlarmRecord, AlarmLevel
from alerts.rules import AlarmRule, RuleContext

# ═══════════════════════════════════════════════
# 默认规则集
# ═══════════════════════════════════════════════
from alerts.rules import ConsecutiveFaultRule, HighConfidenceRule

DEFAULT_RULES = [
    ConsecutiveFaultRule(warning_count=3, critical_count=5),
    HighConfidenceRule(warning_threshold=0.95, critical_threshold=0.99),
]


class AlarmManager:
    """
    报警管理器。

    参数:
        device_id:        当前监控设备 ID
        rules:            报警规则列表 (AlarmRule 实例)
        db_path:          报警 SQLite 数据库路径
        cooldown_seconds: 同故障类型报警冷却时间（秒）
    """

    def __init__(
        self,
        device_id: str = "bearing_001",
        rules: List[AlarmRule] = None,
        db_path: str = "alarms.db",
        cooldown_seconds: float = 60.0,
    ):
        self.device_id = device_id
        self.rules = rules if rules is not None else DEFAULT_RULES
        self.db_path = db_path
        self.cooldown_seconds = cooldown_seconds

        # 设备状态
        self._consecutive_count = 0         # 同类型故障连续次数
        self._last_fault_type = None        # 上一个故障类型
        self._total_windows = 0             # 累计诊断窗口数
        self._last_alarm_time: dict = {}    # fault_type → 上次报警时间

        self._init_db()

    # ═══════════════════════════════════════════════
    # 数据库
    # ═══════════════════════════════════════════════

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS alarms (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                alarm_id TEXT NOT NULL,
                timestamp REAL NOT NULL,
                device_id TEXT NOT NULL,
                fault_type TEXT NOT NULL,
                confidence REAL NOT NULL,
                alarm_level TEXT NOT NULL,
                message TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_alarm_time ON alarms(timestamp)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_alarm_device ON alarms(device_id)
        """)
        conn.commit()
        conn.close()

    def _persist(self, record: AlarmRecord):
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """INSERT INTO alarms
               (alarm_id, timestamp, device_id, fault_type, confidence, alarm_level, message)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                record.id,
                record.timestamp,
                record.device_id,
                record.fault_type,
                record.confidence,
                record.alarm_level.value,
                record.message,
            ),
        )
        conn.commit()
        conn.close()

    # ═══════════════════════════════════════════════
    # 核心逻辑
    # ═══════════════════════════════════════════════

    def evaluate(
        self,
        fault_type: str,
        confidence: float,
        is_fault: bool,
    ) -> Optional[AlarmRecord]:
        """
        接收一次诊断结果，评估是否触发报警。

        参数:
            fault_type: 诊断故障类型名称
            confidence: 置信度
            is_fault:   是否故障（正常状态为 False）

        返回:
            AlarmRecord 如果触发报警
            None 如果不需要报警
        """
        self._total_windows += 1

        # 更新连续故障计数
        if is_fault and fault_type == self._last_fault_type:
            self._consecutive_count += 1
        elif is_fault:
            self._consecutive_count = 1
            self._last_fault_type = fault_type
        else:
            self._consecutive_count = 0
            self._last_fault_type = None

        # 构建规则上下文
        ctx = RuleContext(
            device_id=self.device_id,
            fault_type=fault_type,
            confidence=confidence,
            is_fault=is_fault,
            consecutive_count=self._consecutive_count,
            total_window_count=self._total_windows,
        )

        # 评估所有规则，取最高等级
        best_level = None
        best_message = ""

        for rule in self.rules:
            result = rule.evaluate(ctx)
            if result is not None:
                level, msg = result
                if best_level is None or self._level_gt(level, best_level):
                    best_level = level
                    best_message = f"[{rule.name}] {msg}"

        if best_level is None:
            return None

        # 冷却检查
        now = time.time()
        last_time = self._last_alarm_time.get(fault_type, 0)
        if now - last_time < self.cooldown_seconds:
            return None

        self._last_alarm_time[fault_type] = now

        # 生成报警记录
        record = AlarmRecord(
            timestamp=now,
            device_id=self.device_id,
            fault_type=fault_type,
            confidence=confidence,
            alarm_level=best_level,
            message=best_message,
        )
        self._persist(record)
        return record

    def _level_gt(self, a: AlarmLevel, b: AlarmLevel) -> bool:
        """a 的严重程度是否高于 b。"""
        order = {AlarmLevel.WARNING: 1, AlarmLevel.CRITICAL: 2}
        return order.get(a, 0) > order.get(b, 0)

    # ═══════════════════════════════════════════════
    # 查询
    # ═══════════════════════════════════════════════

    def query(
        self,
        time_range: Optional[tuple] = None,
        level: Optional[AlarmLevel] = None,
        limit: int = 500,
    ) -> List[dict]:
        """查询历史报警。"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row

        conditions = []
        params = []

        if time_range:
            conditions.append("timestamp BETWEEN ? AND ?")
            params.extend(time_range)
        if level:
            conditions.append("alarm_level = ?")
            params.append(level.value)

        where = " AND ".join(conditions) if conditions else "1=1"
        rows = conn.execute(
            f"SELECT * FROM alarms WHERE {where} ORDER BY timestamp DESC LIMIT ?",
            params + [limit],
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def stats(self) -> dict:
        """报警统计。"""
        conn = sqlite3.connect(self.db_path)
        total = conn.execute("SELECT COUNT(*) FROM alarms").fetchone()[0]
        warnings = conn.execute(
            "SELECT COUNT(*) FROM alarms WHERE alarm_level='WARNING'"
        ).fetchone()[0]
        criticals = conn.execute(
            "SELECT COUNT(*) FROM alarms WHERE alarm_level='CRITICAL'"
        ).fetchone()[0]
        dist = conn.execute(
            "SELECT fault_type, COUNT(*) as cnt FROM alarms GROUP BY fault_type"
        ).fetchall()
        conn.close()
        return {
            "total_alarms": total,
            "warnings": warnings,
            "criticals": criticals,
            "by_fault_type": {row[0]: row[1] for row in dist},
        }

    def clear(self):
        """清空所有报警数据。"""
        self._consecutive_count = 0
        self._last_fault_type = None
        self._last_alarm_time.clear()
        conn = sqlite3.connect(self.db_path)
        conn.execute("DELETE FROM alarms")
        conn.commit()
        conn.close()

    # ═══════════════════════════════════════════════
    # 属性
    # ═══════════════════════════════════════════════

    @property
    def consecutive_count(self) -> int:
        return self._consecutive_count

    @property
    def total_windows(self) -> int:
        return self._total_windows


# ================================================================
# 自测
# ================================================================
if __name__ == "__main__":
    import os

    TEST_DB = "_test_alarms.db"
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)

    print("=" * 60)
    print("AlarmManager 单元测试")
    print("=" * 60)

    mgr = AlarmManager(device_id="test_bearing", db_path=TEST_DB, cooldown_seconds=0)

    # 场景 1: 正常状态不应触发报警
    alarm = mgr.evaluate("正常", 0.99, is_fault=False)
    assert alarm is None, "正常不应报警"
    print("  [PASS] 场景1: 正常状态无报警")

    # 场景 2: 连续 3 次同故障 → WARNING
    for i in range(3):
        alarm = mgr.evaluate("内圈故障", 0.90, is_fault=True)
    assert alarm is not None, "连续3次应触发报警"
    assert alarm.alarm_level == AlarmLevel.WARNING, f"应为 WARNING，实际 {alarm.alarm_level}"
    print(f"  [PASS] 场景2: 连续3次 → {alarm.alarm_level.value} | {alarm.message[:40]}...")

    # 场景 3: 继续到 5 次 → CRITICAL
    for i in range(2):
        alarm = mgr.evaluate("内圈故障", 0.90, is_fault=True)
    assert alarm is not None and alarm.alarm_level == AlarmLevel.CRITICAL, \
        f"应为 CRITICAL，实际 {alarm.alarm_level if alarm else None}"
    print(f"  [PASS] 场景3: 连续5次 → {alarm.alarm_level.value}")

    # 场景 4: 切换故障类型 → 重置计数
    alarm = mgr.evaluate("外圈故障", 0.88, is_fault=True)
    assert mgr.consecutive_count == 1, f"切换故障应重置为1，实际 {mgr.consecutive_count}"
    print(f"  [PASS] 场景4: 故障类型切换，计数重置 = {mgr.consecutive_count}")

    # 场景 5: 高置信度即时 WARNING
    mgr._consecutive_count = 0  # 重置
    alarm = mgr.evaluate("外圈故障", 0.97, is_fault=True)
    assert alarm is not None and alarm.alarm_level == AlarmLevel.WARNING, \
        f"置信度0.97应触发WARNING，实际 {alarm.alarm_level if alarm else None}"
    print(f"  [PASS] 场景5: 置信度0.97 → {alarm.alarm_level.value}")

    # 场景 6: 高置信度即时 CRITICAL
    alarm = mgr.evaluate("外圈故障", 0.995, is_fault=True)
    assert alarm is not None and alarm.alarm_level == AlarmLevel.CRITICAL, \
        f"置信度0.995应触发CRITICAL，实际 {alarm.alarm_level if alarm else None}"
    print(f"  [PASS] 场景6: 置信度0.995 → {alarm.alarm_level.value}")

    # 统计验证
    st = mgr.stats()
    print(f"  [INFO] 统计: WARNING={st['warnings']}, CRITICAL={st['criticals']}, "
          f"分布={st['by_fault_type']}")

    # 查询
    records = mgr.query(limit=10)
    print(f"  [PASS] 查询: {len(records)} 条报警记录")

    mgr.clear()
    os.remove(TEST_DB)
    print(f"\n  全部 6 个场景通过。")
