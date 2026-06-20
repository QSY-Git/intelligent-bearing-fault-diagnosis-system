"""
诊断历史存储

内存 deque 供实时仪表盘毫秒级查询 + SQLite 持久化供历史趋势分析。

迁移来源: stream/diagnostics_store.py (DiagnosticsStore 类, 全文件)
迁移日期: 2025-05-28
改动说明: 零逻辑改动。DiagnosisRecord 已独立为 core/persistence/models.py，
          store 通过 asdict() 序列化，无需直接依赖具体 dataclass 类型。
"""

import sys
import os
import time
import json
import sqlite3
from collections import deque
from typing import List, Optional
from dataclasses import asdict

# 确保项目根目录在 sys.path 中，支持直接运行此文件
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


class DiagnosticsStore:
    """
    诊断历史存储。

    双层存储:
    - 内存: deque(maxlen) — 仪表盘毫秒级查询
    - 磁盘: SQLite — 持久化，支持历史回查

    使用:
        store = DiagnosticsStore(memory_size=5000, db_path="diagnostics.db")
        store.append(record)                        # record 是任意 dataclass
        recent = store.get_recent(100)
        history = store.query(time_range=(t1, t2))
    """

    def __init__(self, memory_size=5000, db_path="diagnostics.db"):
        self.db_path = db_path
        self._recent = deque(maxlen=memory_size)
        self._alarms = deque(maxlen=200)
        self._init_db()

    def _init_db(self):
        """初始化 SQLite 表。"""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS diagnoses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                predicted_label INTEGER NOT NULL,
                predicted_name TEXT NOT NULL,
                confidence REAL NOT NULL,
                all_probs TEXT NOT NULL,
                mean_val REAL,
                rms REAL,
                kurtosis REAL,
                skewness REAL,
                dom_freq REAL,
                spectral_energy REAL,
                alarm INTEGER DEFAULT 0
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp ON diagnoses(timestamp)
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS alarms (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                fault_type TEXT NOT NULL,
                confidence REAL NOT NULL,
                message TEXT
            )
        """)
        conn.commit()
        conn.close()

    def append(self, record):
        """
        保存一条诊断记录。

        参数:
            record: DiagnosisRecord 实例（或任意含有对应字段的 dataclass）
        """
        d = asdict(record)
        self._recent.append(d)

        # 持久化到 SQLite
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """INSERT INTO diagnoses
               (timestamp, predicted_label, predicted_name, confidence,
                all_probs, mean_val, rms, kurtosis, skewness,
                dom_freq, spectral_energy, alarm)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                d["timestamp"], d["predicted_label"], d["predicted_name"],
                d["confidence"], json.dumps(d["all_probs"]),
                d["mean_val"], d["rms"], d["kurtosis"], d["skewness"],
                d["dom_freq"], d["spectral_energy"], int(d["alarm"])
            )
        )

        # 报警持久化
        if d["alarm"]:
            self._alarms.append(d)
            conn.execute(
                """INSERT INTO alarms (timestamp, fault_type, confidence, message)
                   VALUES (?, ?, ?, ?)""",
                (
                    d["timestamp"], d["predicted_name"], d["confidence"],
                    f"连续检测到{d['predicted_name']}，置信度 {d['confidence']:.1%}"
                )
            )

        conn.commit()
        conn.close()

    def get_recent(self, n: int = 100) -> List[dict]:
        """获取最近 n 条记录（从内存读取）。"""
        items = list(self._recent)[-n:]
        return items

    def get_recent_alarms(self, n: int = 50) -> List[dict]:
        """获取最近 n 条报警。"""
        return list(self._alarms)[-n:]

    def query(self, time_range: Optional[tuple] = None, limit: int = 5000) -> List[dict]:
        """
        从 SQLite 查询历史记录。

        参数:
            time_range: (start_ts, end_ts) 时间范围，None 表示不限
            limit:      最大返回条数
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        if time_range:
            rows = conn.execute(
                "SELECT * FROM diagnoses WHERE timestamp BETWEEN ? AND ? "
                "ORDER BY timestamp DESC LIMIT ?",
                (time_range[0], time_range[1], limit)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM diagnoses ORDER BY timestamp DESC LIMIT ?",
                (limit,)
            ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def query_alarms(self, time_range: Optional[tuple] = None, limit: int = 500) -> List[dict]:
        """从 SQLite 查询历史报警。"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        if time_range:
            rows = conn.execute(
                "SELECT * FROM alarms WHERE timestamp BETWEEN ? AND ? "
                "ORDER BY timestamp DESC LIMIT ?",
                (time_range[0], time_range[1], limit)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM alarms ORDER BY timestamp DESC LIMIT ?",
                (limit,)
            ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def stats(self) -> dict:
        """统计信息：各类别数量、总记录数。"""
        conn = sqlite3.connect(self.db_path)
        cur = conn.execute(
            "SELECT predicted_name, COUNT(*) as cnt FROM diagnoses GROUP BY predicted_name"
        )
        dist = {row[0]: row[1] for row in cur.fetchall()}
        total = conn.execute("SELECT COUNT(*) FROM diagnoses").fetchone()[0]
        alarm_total = conn.execute("SELECT COUNT(*) FROM alarms").fetchone()[0]
        conn.close()
        return {"total_records": total, "total_alarms": alarm_total, "distribution": dist}

    def clear(self):
        """清空所有数据（用于重置）。"""
        self._recent.clear()
        self._alarms.clear()
        conn = sqlite3.connect(self.db_path)
        conn.execute("DELETE FROM diagnoses")
        conn.execute("DELETE FROM alarms")
        conn.commit()
        conn.close()


if __name__ == "__main__":
    from core.persistence.models import DiagnosisRecord

    print("=" * 50)
    print("DiagnosticsStore 迁移验证")
    print("=" * 50)

    TEST_DB = "_test_migration.db"

    # 清理旧数据
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)

    store = DiagnosticsStore(memory_size=100, db_path=TEST_DB)

    # 写入测试记录
    records = []
    for i in range(15):
        label = i % 4
        names = {0: "正常", 1: "内圈故障", 2: "外圈故障", 3: "滚动体故障"}
        probs = [0.0] * 4
        probs[label] = 0.9 + i * 0.005
        rec = DiagnosisRecord(
            timestamp=1716883200.0 + i * 60,
            predicted_label=label,
            predicted_name=names[label],
            confidence=probs[label],
            all_probs=probs,
            mean_val=0.01 * i,
            rms=0.2 + i * 0.01,
            kurtosis=3.0 + i * 0.1,
            skewness=-0.1 * i,
            dom_freq=300.0 + i * 10,
            spectral_energy=10000.0 + i * 100,
            alarm=(label != 0 and i >= 3),
        )
        records.append(rec)
        store.append(rec)

    # 测试 1: 内存查询
    recent = store.get_recent(10)
    assert len(recent) == 10, f"期望 10 条，实际 {len(recent)}"
    assert recent[-1]["predicted_label"] == records[-1].predicted_label
    print(f"  测试 1 通过: get_recent(10) 返回 {len(recent)} 条")

    # 测试 2: 报警查询
    alarms = store.get_recent_alarms(50)
    expected_alarms = sum(1 for r in records if r.alarm)
    assert len(alarms) == expected_alarms, f"期望 {expected_alarms} 条报警，实际 {len(alarms)}"
    print(f"  测试 2 通过: 报警 {len(alarms)} 条")

    # 测试 3: SQLite 查询
    all_rows = store.query(limit=100)
    assert len(all_rows) == 15, f"期望 15 条，实际 {len(all_rows)}"
    print(f"  测试 3 通过: query() 返回 {len(all_rows)} 条")

    # 测试 4: 时间范围查询
    t_start = 1716883200.0
    t_end = 1716883200.0 + 6 * 60
    ranged = store.query(time_range=(t_start, t_end), limit=100)
    assert len(ranged) == 7, f"时间范围查询期望 7 条，实际 {len(ranged)}"
    print(f"  测试 4 通过: 时间范围查询返回 {len(ranged)} 条")

    # 测试 5: 统计信息
    st = store.stats()
    assert st["total_records"] == 15
    assert st["total_alarms"] == expected_alarms
    assert len(st["distribution"]) == 4
    print(f"  测试 5 通过: stats = {st}")

    # 测试 6: 清除
    store.clear()
    assert len(store.get_recent(100)) == 0
    assert len(store.query(limit=100)) == 0
    assert store.stats()["total_records"] == 0
    print(f"  测试 6 通过: clear() 后数据为空")

    # 清理
    os.remove(TEST_DB)
    print(f"\n  全部 6 项测试通过。迁移成功。")
