"""
诊断历史存储
内存 deque 供实时仪表盘查询 + SQLite 持久化供历史趋势分析。
"""

import os
import time
import json
import sqlite3
from collections import deque
from typing import List, Optional
from dataclasses import asdict


class DiagnosticsStore:
    """
    诊断历史存储。

    双层存储:
    - 内存: deque(maxlen) — 仪表盘毫秒级查询
    - 磁盘: SQLite — 持久化，支持历史回查

    使用:
        store = DiagnosticsStore(memory_size=5000, db_path="diagnostics.db")
        store.append(record)
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
            record: DiagnosisRecord 实例
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
