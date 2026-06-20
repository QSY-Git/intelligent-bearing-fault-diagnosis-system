"""
信号变换模块
功能：滑动窗口切片等数组变换操作。

迁移来源: utils/data_loader.py (sliding_window 函数, L42-L57)
迁移日期: 2025-05-28
改动说明: 零逻辑改动。纯数组变换，无项目内部依赖。
"""

import numpy as np


def sliding_window(data, window_size=1024, step=512):
    """
    滑动窗口切片，将长信号切分为等长样本段。
    参数:
        data:      1D NumPy数组，原始振动信号
        window_size: 窗口大小（采样点数）
        step:       滑动步长（采样点数）
    返回:
        segments:   shape (n_samples, window_size) 的2D数组
    """
    segments = []
    start = 0
    while start + window_size <= len(data):
        segments.append(data[start:start + window_size])
        start += step
    return np.array(segments)


if __name__ == "__main__":
    print("=" * 50)
    print("sliding_window 迁移验证")
    print("=" * 50)

    data = np.arange(5000, dtype=np.float64)
    segs = sliding_window(data, window_size=1024, step=512)

    expected_n = (5000 - 1024) // 512 + 1  # = 8
    assert segs.shape[0] == expected_n, f"期望 {expected_n} 段，实际 {segs.shape[0]}"
    assert segs.shape[1] == 1024, f"期望窗口 1024，实际 {segs.shape[1]}"
    assert segs[0, 0] == 0.0, f"第一段起点应为 0.0"
    assert segs[0, -1] == 1023.0, f"第一段终点应为 1023.0"
    assert segs[-1, 0] == 3584.0, f"最后一段起点应为 3584.0 (step=512, 第8段)"

    print(f"  输入长度: {len(data)}")
    print(f"  窗口大小: 1024, 步长: 512")
    print(f"  生成段数: {segs.shape[0]}")
    print(f"  输出形状: {segs.shape}")
    print("  所有断言通过。")
