"""
无数据泄漏的数据加载模块

与旧版 build_dataset() 的关键区别:
1. 保留每个窗口的 group_id（源文件+工况），支持 Group-based 划分
2. 返回元数据 DataFrame，包含 source_file, fault_type, load, fault_size
3. 窗口创建和数据集划分严格分离
4. 支持按负载筛选（用于跨工况实验）

使用示例:
    loader = LeakageFreeLoader(data_root="data", window_size=1024, step=512)
    X, y, meta = loader.load_all()  # meta 包含 group_id 等信息
    # 然后用 GroupShuffleSplit 按 group_id 划分
"""

import os
import sys
import hashlib
import numpy as np
import scipy.io as sio
from glob import glob
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from core.signal.transform import sliding_window


# ============================================================
# CWRU 12k Drive End 标准文件→工况映射
# ============================================================
CWRU_FILE_METADATA = {
    # ---- 正常基线 ----
    97:  {"fault_type": "normal",     "fault_size": 0.000, "load": 0},
    98:  {"fault_type": "normal",     "fault_size": 0.000, "load": 1},
    99:  {"fault_type": "normal",     "fault_size": 0.000, "load": 2},
    100: {"fault_type": "normal",     "fault_size": 0.000, "load": 3},
    # ---- 内圈故障 0.007" ----
    105: {"fault_type": "inner_race", "fault_size": 0.007, "load": 0},
    106: {"fault_type": "inner_race", "fault_size": 0.007, "load": 1},
    107: {"fault_type": "inner_race", "fault_size": 0.007, "load": 2},
    108: {"fault_type": "inner_race", "fault_size": 0.007, "load": 3},
    # ---- 内圈故障 0.014" ----
    169: {"fault_type": "inner_race", "fault_size": 0.014, "load": 0},
    170: {"fault_type": "inner_race", "fault_size": 0.014, "load": 1},
    171: {"fault_type": "inner_race", "fault_size": 0.014, "load": 2},
    172: {"fault_type": "inner_race", "fault_size": 0.014, "load": 3},
    # ---- 内圈故障 0.021" ----
    209: {"fault_type": "inner_race", "fault_size": 0.021, "load": 0},
    210: {"fault_type": "inner_race", "fault_size": 0.021, "load": 1},
    211: {"fault_type": "inner_race", "fault_size": 0.021, "load": 2},
    212: {"fault_type": "inner_race", "fault_size": 0.021, "load": 3},
    # ---- 外圈故障 0.007" @6:00 ----
    130: {"fault_type": "outer_race", "fault_size": 0.007, "load": 0},
    131: {"fault_type": "outer_race", "fault_size": 0.007, "load": 1},
    132: {"fault_type": "outer_race", "fault_size": 0.007, "load": 2},
    133: {"fault_type": "outer_race", "fault_size": 0.007, "load": 3},
    # ---- 外圈故障 0.014" ----
    197: {"fault_type": "outer_race", "fault_size": 0.014, "load": 0},
    198: {"fault_type": "outer_race", "fault_size": 0.014, "load": 1},
    199: {"fault_type": "outer_race", "fault_size": 0.014, "load": 2},
    200: {"fault_type": "outer_race", "fault_size": 0.014, "load": 3},
    # ---- 外圈故障 0.021" ----
    234: {"fault_type": "outer_race", "fault_size": 0.021, "load": 0},
    235: {"fault_type": "outer_race", "fault_size": 0.021, "load": 1},
    236: {"fault_type": "outer_race", "fault_size": 0.021, "load": 2},
    237: {"fault_type": "outer_race", "fault_size": 0.021, "load": 3},
    # ---- 滚动体故障 0.007" ----
    118: {"fault_type": "ball", "fault_size": 0.007, "load": 0},
    119: {"fault_type": "ball", "fault_size": 0.007, "load": 1},
    120: {"fault_type": "ball", "fault_size": 0.007, "load": 2},
    121: {"fault_type": "ball", "fault_size": 0.007, "load": 3},
    # ---- 滚动体故障 0.014" ----
    185: {"fault_type": "ball", "fault_size": 0.014, "load": 0},
    186: {"fault_type": "ball", "fault_size": 0.014, "load": 1},
    187: {"fault_type": "ball", "fault_size": 0.014, "load": 2},
    188: {"fault_type": "ball", "fault_size": 0.014, "load": 3},
    # ---- 滚动体故障 0.021" ----
    222: {"fault_type": "ball", "fault_size": 0.021, "load": 0},
    223: {"fault_type": "ball", "fault_size": 0.021, "load": 1},
    224: {"fault_type": "ball", "fault_size": 0.021, "load": 2},
    225: {"fault_type": "ball", "fault_size": 0.021, "load": 3},
}

# 故障类型英文→(label_id, 中文名) 映射
FAULT_TYPE_TO_LABEL = {
    "normal":     0,
    "inner_race": 1,
    "outer_race": 2,
    "ball":       3,
}
LABEL_ID_TO_CN = {0: "正常", 1: "内圈故障", 2: "外圈故障", 3: "滚动体故障"}
LABEL_ID_TO_EN = {0: "Normal", 1: "Inner Race", 2: "Outer Race", 3: "Ball"}


def _extract_file_number(filepath: str) -> int:
    """从文件名提取数字，如 'data/normal/97.mat.txt' -> 97"""
    basename = os.path.basename(filepath)
    # 移除 .mat.txt 后缀
    for suffix in [".mat.txt", ".mat", ".txt"]:
        if basename.endswith(suffix):
            basename = basename[: -len(suffix)]
            break
    return int(basename)


def _compute_window_hash(window: np.ndarray) -> str:
    """计算单个窗口的 SHA256 哈希（用于重复检测）。"""
    return hashlib.sha256(window.tobytes()).hexdigest()[:16]


class LeakageFreeLoader:
    """
    无数据泄漏的数据加载器。

    与旧版 build_dataset() 的区别:
    - 不执行 np.concatenate 后丢失文件身份
    - 返回每个窗口的完整元数据
    - group_id = source_file（用于 Group-based 划分）
    """

    def __init__(self, data_root: str = "data",
                 window_size: int = 1024, step: int = 512):
        self.data_root = data_root
        self.window_size = window_size
        self.step = step

        # 内部类别标签映射
        self.fault_type_to_label = FAULT_TYPE_TO_LABEL
        self.label_names = [LABEL_ID_TO_CN[i] for i in range(4)]
        self.label_names_en = [LABEL_ID_TO_EN[i] for i in range(4)]

    def load_all(self) -> tuple:
        """
        加载全部数据，返回窗口和元数据。

        返回:
            X:        np.ndarray, shape (n_windows, window_size)
            y:        np.ndarray, shape (n_windows,)
            meta:     dict, 包含:
                - group_ids:      np.ndarray, shape (n_windows,), 每个窗口的源文件编号
                - source_files:   list[str], 每个窗口的源文件路径
                - fault_types:    list[str], 故障类型字符串
                - fault_sizes:    np.ndarray, 故障尺寸
                - loads:          np.ndarray, 负载(HP)
                - window_hashes:  list[str], 窗口内容哈希
                - window_starts:  np.ndarray, 窗口起始位置
            label_names: list[str]
        """
        X_list = []
        y_list = []
        group_ids = []
        source_files = []
        fault_types = []
        fault_sizes = []
        loads = []
        window_hashes = []
        window_starts = []

        # 直接扫描 data_root 下所有子目录和 .mat.txt 文件
        # 不依赖目录名的中英文匹配，完全根据 CWRU 文件编号确定标签
        class_file_counts = defaultdict(int)

        for subdir in sorted(os.listdir(self.data_root)):
            subdir_path = os.path.join(self.data_root, subdir)
            if not os.path.isdir(subdir_path):
                continue

            mat_files = sorted(glob(os.path.join(subdir_path, "*.mat.txt")))
            if not mat_files:
                continue

            for fpath in mat_files:
                file_num = _extract_file_number(fpath)
                metadata = CWRU_FILE_METADATA.get(file_num, {})
                if not metadata:
                    print(f"  [警告] 文件 {fpath} 无元数据，跳过")
                    continue

                ft = metadata["fault_type"]
                fs = metadata["fault_size"]
                ld = metadata["load"]
                label_id = FAULT_TYPE_TO_LABEL[ft]

                # 加载原始信号
                mat_data = sio.loadmat(fpath)
                data_keys = [k for k in mat_data.keys()
                             if not k.startswith("__")]
                if not data_keys:
                    raise ValueError(f"未找到有效数据键: {fpath}")
                raw = mat_data[data_keys[0]].flatten().astype(np.float64)

                # 滑动窗口
                segs = sliding_window(raw, self.window_size, self.step)
                n_segs = segs.shape[0]

                # 记录元数据
                for i in range(n_segs):
                    X_list.append(segs[i])
                    y_list.append(label_id)
                    group_ids.append(file_num)
                    source_files.append(fpath)
                    fault_types.append(ft)
                    fault_sizes.append(fs)
                    loads.append(ld)
                    window_hashes.append(_compute_window_hash(segs[i]))
                    window_starts.append(i * self.step)

                class_file_counts[ft] += 1

        for ft, cnt in class_file_counts.items():
            print(f"  类别 [{LABEL_ID_TO_CN[FAULT_TYPE_TO_LABEL[ft]]}]: "
                  f"{cnt} 个文件")

        X = np.array(X_list, dtype=np.float64)
        y = np.array(y_list, dtype=np.int64)
        meta = {
            "group_ids": np.array(group_ids, dtype=np.int64),
            "source_files": source_files,
            "fault_types": fault_types,
            "fault_sizes": np.array(fault_sizes, dtype=np.float64),
            "loads": np.array(loads, dtype=np.int64),
            "window_hashes": window_hashes,
            "window_starts": np.array(window_starts, dtype=np.int64),
        }

        print(f"  总样本数: {X.shape[0]}, 窗口长度: {X.shape[1]}")
        return X, y, meta, self.label_names

    def load_by_loads(self, target_loads: list) -> tuple:
        """
        只加载指定负载的数据（用于跨工况实验）。

        参数:
            target_loads: list[int], 要加载的负载列表，如 [0, 1, 2]

        返回:
            X, y, meta, label_names
        """
        X, y, meta, label_names = self.load_all()
        mask = np.isin(meta["loads"], target_loads)
        if not mask.any():
            raise ValueError(f"没有找到负载为 {target_loads} 的数据")
        X_f = X[mask]
        y_f = y[mask]
        meta_f = {k: (v[mask] if isinstance(v, np.ndarray)
                       else [v[i] for i in range(len(v)) if mask[i]])
                   for k, v in meta.items()}
        print(f"  按负载 {target_loads} 筛选: {X_f.shape[0]} 样本 "
              f"(从 {X.shape[0]} 中)")
        return X_f, y_f, meta_f, label_names

    def check_duplicates(self, meta: dict) -> dict:
        """检查窗口是否有重复内容。"""
        hashes = meta["window_hashes"]
        unique = set(hashes)
        n_dup = len(hashes) - len(unique)
        return {
            "total_windows": len(hashes),
            "unique_windows": len(unique),
            "duplicate_windows": n_dup,
            "duplicate_ratio": n_dup / len(hashes) if hashes else 0,
        }

    def check_split_leakage(self, train_meta: dict, test_meta: dict) -> dict:
        """
        检查 train/test 之间是否有泄漏。
        返回泄漏检测结果字典。如果发现泄漏，result['clean'] = False。
        """
        train_files = set(train_meta["group_ids"])
        test_files = set(test_meta["group_ids"])
        file_overlap = train_files & test_files

        train_hashes = set(train_meta["window_hashes"])
        test_hashes = set(test_meta["window_hashes"])
        hash_overlap = train_hashes & test_hashes

        return {
            "clean": len(file_overlap) == 0 and len(hash_overlap) == 0,
            "file_overlap": sorted(file_overlap),
            "hash_overlap_count": len(hash_overlap),
            "train_files": len(train_files),
            "test_files": len(test_files),
            "train_windows": len(train_meta["window_hashes"]),
            "test_windows": len(test_meta["window_hashes"]),
        }
