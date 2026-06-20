"""
build_demo_assets.py

从项目 data/ 目录读取真实 CWRU 振动信号，每类截取 65536 个连续采样点，
输出到 demo_data/ 目录供 Streamlit Cloud 在线演示使用。

严禁使用随机数据或人工正弦波。
如果无法读取真实数据，直接报错，不得伪造。

输出:
    demo_data/normal.npz
    demo_data/inner_race.npz
    demo_data/outer_race.npz
    demo_data/ball.npz
    demo_data/dataset_stats.json
"""

import os
import sys
import json
import io
import numpy as np
import scipy.io as sio
from glob import glob
from collections import OrderedDict

# -- Windows GBK 终端兼容 -----------------------------------------
if sys.stdout.encoding and sys.stdout.encoding.lower() in ('gbk', 'cp936'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8',
                                  errors='replace', line_buffering=True)

# -- 确保项目根在 path --------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# -- 故障类别映射 ------------------------------------------------
LABEL_MAP = OrderedDict([
    ("正常",     (0, "正常")),
    ("内圈故障",  (1, "内圈故障")),
    ("外圈故障",  (2, "外圈故障")),
    ("滚动体故障", (3, "滚动体故障")),
])

# 输出文件名（英文，兼容各种文件系统）
FAULT_FILES = {
    "正常":     "normal.npz",
    "内圈故障":  "inner_race.npz",
    "外圈故障":  "outer_race.npz",
    "滚动体故障": "ball.npz",
}

DEMO_LENGTH = 65536   # 每类保留的采样点数
SAMPLE_RATE = 12000   # CWRU 12k Drive End


def load_mat_txt(filepath):
    """读取 .mat.txt 文件（本质是 MATLAB v5 二进制 MAT 文件）。"""
    try:
        mat_data = sio.loadmat(filepath)
    except Exception as e:
        raise IOError(f"无法读取文件 {filepath}: {e}")

    data_keys = [k for k in mat_data.keys() if not k.startswith("__")]
    if not data_keys:
        raise ValueError(f"未找到有效数据键: {filepath}")
    data = mat_data[data_keys[0]].flatten().astype(np.float64)
    return data


def build_demo_assets(data_root="data", output_dir="demo_data"):
    """主流程：读取真实数据 -> 截取 -> 保存。"""

    data_root = os.path.join(PROJECT_ROOT, data_root)
    output_dir = os.path.join(PROJECT_ROOT, output_dir)

    if not os.path.isdir(data_root):
        raise FileNotFoundError(
            f"数据目录不存在: {data_root}\n"
            "请确保原始 CWRU 数据已放置在 data/ 下。"
        )

    os.makedirs(output_dir, exist_ok=True)

    stats = {
        "sample_rate": SAMPLE_RATE,
        "demo_length_per_class": DEMO_LENGTH,
        "num_classes": 4,
        "class_names": ["正常", "内圈故障", "外圈故障", "滚动体故障"],
        "class_names_en": ["Normal", "Inner Race", "Outer Race", "Ball"],
        "files": {},
    }

    for folder_name, (label_id, label_name) in LABEL_MAP.items():
        folder_path = os.path.join(data_root, folder_name)
        if not os.path.isdir(folder_path):
            raise FileNotFoundError(
                f"数据子目录不存在: {folder_path}\n"
                "无法读取真实数据，请检查 CWRU 数据集完整性。"
            )

        mat_files = sorted(glob(os.path.join(folder_path, "*.mat.txt")))
        if not mat_files:
            raise FileNotFoundError(
                f"目录 {folder_path} 下未找到 .mat.txt 文件。\n"
                "无法读取真实数据，请检查 CWRU 数据集完整性。"
            )

        print(f"[{label_name}] 找到 {len(mat_files)} 个文件，正在读取...")

        # 加载第一个文件的信号
        first_signal = load_mat_txt(mat_files[0])
        print(f"  |-- 主信号长度: {len(first_signal):,} 点")

        if len(first_signal) < DEMO_LENGTH:
            # 第一个文件不够长，尝试拼接后续文件
            print(f"    信号不足 {DEMO_LENGTH} 点，正在拼接后续文件...")
            parts = [first_signal]
            for fp in mat_files[1:]:
                parts.append(load_mat_txt(fp))
                total = sum(len(p) for p in parts)
                if total >= DEMO_LENGTH:
                    break
            signal = np.concatenate(parts)
            if len(signal) < DEMO_LENGTH:
                raise ValueError(
                    f"[{label_name}] 所有文件拼接后仅 {len(signal)} 点，"
                    f"不足 {DEMO_LENGTH} 点。无法生成演示数据。"
                )
            print(f"    拼接完成: {len(signal):,} 点")
        else:
            signal = first_signal

        # 截取前 DEMO_LENGTH 个连续采样点（从开头取，保证是真实连续信号）
        signal = signal[:DEMO_LENGTH].astype(np.float64)

        # 输出文件名
        out_name = FAULT_FILES[folder_name]
        out_path = os.path.join(output_dir, out_name)
        np.savez_compressed(
            out_path,
            signal=signal,
            label_id=label_id,
            label_name=label_name,
            sample_rate=SAMPLE_RATE,
            source_file=os.path.basename(mat_files[0]),
        )
        file_size_kb = os.path.getsize(out_path) / 1024
        print(f"  [OK] 已保存: {out_name} ({file_size_kb:.1f} KB)")

        stats["files"][folder_name] = {
            "file": out_name,
            "signal_length": len(signal),
            "label_id": label_id,
            "label_name": label_name,
            "source_file": os.path.basename(mat_files[0]),
        }

    # 写入统计 JSON
    stats_path = os.path.join(output_dir, "dataset_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] 统计文件已保存: dataset_stats.json")

    # 最终验证
    print("\n" + "=" * 50)
    print("验证输出的演示数据文件")
    print("=" * 50)
    for folder_name, fname in FAULT_FILES.items():
        fpath = os.path.join(output_dir, fname)
        data = np.load(fpath, allow_pickle=False)
        sig = data["signal"]
        lid = int(data["label_id"])
        lname = str(data["label_name"])
        sr = int(data["sample_rate"])
        src = str(data["source_file"])

        # 检查信号真实性
        if np.all(sig == 0):
            raise ValueError(f"{fname}: 信号全为零，数据异常！")
        if np.std(sig) < 1e-12:
            raise ValueError(f"{fname}: 信号方差接近零，疑似非真实数据！")

        print(f"  {fname}: len={len(sig)}, label={lid}({lname}), "
              f"sr={sr}, src={src}, "
              f"range=[{sig.min():.4f}, {sig.max():.4f}], "
              f"std={np.std(sig):.4f}")

    print("\n[OK] 所有演示数据验证通过。")
    print(f"[OK] 输出目录: {output_dir}")
    return stats


if __name__ == "__main__":
    build_demo_assets()
