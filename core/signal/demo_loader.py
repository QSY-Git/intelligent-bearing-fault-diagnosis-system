"""
demo_loader.py — 演示数据统一加载器

为 Streamlit Cloud 在线演示提供轻量级数据加载入口。
所有路径均基于项目根目录计算，兼容 Windows 和 Linux。

使用:
    from core.signal.demo_loader import (
        load_demo_signal,
        build_demo_dataset,
        load_dataset_stats,
        FAULT_TYPES_CN,
        SAMPLE_RATE,
    )

    signal = load_demo_signal("内圈故障")          # → shape (DEMO_LENGTH,)
    X, y, names = build_demo_dataset(1024, 512)     # → (n_samples, window_size), (n_samples,), [str]
    stats = load_dataset_stats()                     # → dict
"""

import os
import sys
import json
import numpy as np

# ── 项目根目录（始终基于本文件位置计算） ──────────
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, _PROJECT_ROOT)
_DEMO_DIR = os.path.join(_PROJECT_ROOT, "demo_data")

SAMPLE_RATE = 12000

# 故障类型 → .npz 文件名映射
_FAULT_FILE_MAP = {
    "正常":     "normal.npz",
    "内圈故障":  "inner_race.npz",
    "外圈故障":  "outer_race.npz",
    "滚动体故障": "ball.npz",
}

# 标签 ID → 名称
_ID_TO_NAME = {0: "正常", 1: "内圈故障", 2: "外圈故障", 3: "滚动体故障"}
_NAME_TO_ID = {v: k for k, v in _ID_TO_NAME.items()}

FAULT_TYPES_CN = list(_FAULT_FILE_MAP.keys())


def _ensure_demo_dir():
    if not os.path.isdir(_DEMO_DIR):
        raise FileNotFoundError(
            f"演示数据目录不存在: {_DEMO_DIR}\n"
            "请先运行: python tools/build_demo_assets.py"
        )


def load_demo_signal(fault_type: str) -> np.ndarray:
    """
    加载指定故障类型的演示信号（原始 1D 振动数据）。

    参数:
        fault_type: "正常" | "内圈故障" | "外圈故障" | "滚动体故障"

    返回:
        signal: 1D numpy array, shape (DEMO_LENGTH,)
    """
    _ensure_demo_dir()

    if fault_type not in _FAULT_FILE_MAP:
        raise ValueError(
            f"未知故障类型: '{fault_type}'，"
            f"支持: {list(_FAULT_FILE_MAP.keys())}"
        )

    fname = _FAULT_FILE_MAP[fault_type]
    fpath = os.path.join(_DEMO_DIR, fname)
    if not os.path.isfile(fpath):
        raise FileNotFoundError(f"演示数据文件缺失: {fpath}")

    data = np.load(fpath, allow_pickle=False)
    signal = data["signal"]
    return signal.copy()


def build_demo_dataset(window_size: int = 1024, step: int = 512):
    """
    从演示 .npz 文件构建窗口切片数据集。

    参数:
        window_size: 滑动窗口大小
        step:        滑动步长

    返回:
        X:           shape (n_samples, window_size)
        y:           shape (n_samples,)
        label_names: list of str
    """
    from core.signal.transform import sliding_window

    X_list, y_list = [], []

    for fault_type, fname in _FAULT_FILE_MAP.items():
        fpath = os.path.join(_DEMO_DIR, fname)
        if not os.path.isfile(fpath):
            continue

        data = np.load(fpath, allow_pickle=False)
        signal = data["signal"]
        label_id = int(data["label_id"])

        segs = sliding_window(signal, window_size, step)
        X_list.append(segs)
        y_list.append(np.full(segs.shape[0], label_id))

    if not X_list:
        raise RuntimeError("演示数据集为空，请先运行 build_demo_assets.py")

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    label_names = list(_FAULT_FILE_MAP.keys())

    return X, y, label_names


def load_dataset_stats() -> dict:
    """
    加载演示数据集统计信息。

    返回:
        dict: 与 dataset_stats.json 内容一致
    """
    _ensure_demo_dir()
    stats_path = os.path.join(_DEMO_DIR, "dataset_stats.json")
    if not os.path.isfile(stats_path):
        raise FileNotFoundError(f"统计文件缺失: {stats_path}")
    with open(stats_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ═══════════════════════════════════════════════════
# 自测
# ═══════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 50)
    print("demo_loader 自测")
    print("=" * 50)

    # 测试 1: 加载单类信号
    sig = load_demo_signal("内圈故障")
    print(f"  load_demo_signal('内圈故障'): shape={sig.shape}, "
          f"dtype={sig.dtype}, range=[{sig.min():.4f}, {sig.max():.4f}]")
    assert sig.ndim == 1
    assert len(sig) > 0

    # 测试 2: 构建数据集
    X, y, names = build_demo_dataset(window_size=1024, step=512)
    print(f"  build_demo_dataset: X.shape={X.shape}, y.shape={y.shape}, "
          f"classes={names}")
    assert X.shape[0] > 0
    assert X.shape[1] == 1024
    assert len(names) == 4

    # 测试 3: 读取统计
    stats = load_dataset_stats()
    print(f"  load_dataset_stats: sample_rate={stats['sample_rate']}, "
          f"num_classes={stats['num_classes']}")
    assert stats["num_classes"] == 4

    print("\n[OK] 全部自测通过。")
