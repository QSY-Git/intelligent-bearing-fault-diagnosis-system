"""
诊断数据模型

定义诊断记录、报警记录等核心数据结构。

迁移来源: stream/stream_processor.py (DiagnosisRecord dataclass, L15-L29)
迁移日期: 2025-05-28
改动说明: 零逻辑改动。纯 dataclass 定义，无项目内部依赖。
"""

from dataclasses import dataclass


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


if __name__ == "__main__":
    print("=" * 50)
    print("DiagnosisRecord 迁移验证")
    print("=" * 50)

    # 构造一条记录
    record = DiagnosisRecord(
        timestamp=1716883200.0,
        predicted_label=1,
        predicted_name="内圈故障",
        confidence=0.9876,
        all_probs=[0.001, 0.9876, 0.008, 0.0034],
        mean_val=0.012,
        rms=0.234,
        kurtosis=4.56,
        skewness=-0.12,
        dom_freq=345.6,
        spectral_energy=12345.0,
        alarm=True,
    )

    # 验证字段
    assert record.predicted_label == 1
    assert record.predicted_name == "内圈故障"
    assert abs(record.confidence - 0.9876) < 1e-6
    assert len(record.all_probs) == 4
    assert record.alarm is True

    # 验证 asdict 可序列化
    from dataclasses import asdict
    d = asdict(record)
    assert d["timestamp"] == 1716883200.0
    assert d["rms"] == 0.234
    assert d["kurtosis"] == 4.56
    assert d["alarm"] is True

    print(f"  标签: {record.predicted_name}")
    print(f"  置信度: {record.confidence:.2%}")
    print(f"  报警: {record.alarm}")
    print(f"  asdict 字段数: {len(d)}")
    print("  所有断言通过。迁移成功。")
