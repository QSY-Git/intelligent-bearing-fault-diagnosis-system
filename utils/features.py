"""
特征提取与降维模块
功能：提取时域/频域特征，执行PCA降维用于可视化。
"""

import numpy as np
from scipy.fft import fft
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def extract_time_domain(segments):
    """
    提取时域特征：均值、均方根(RMS)、峭度(Kurtosis)、偏度(Skewness)。
    参数:
        segments: shape (n_samples, window_size)
    返回:
        features: shape (n_samples, 4)
    """
    mean_vals   = np.mean(segments, axis=1)                       # 均值
    rms_vals    = np.sqrt(np.mean(segments ** 2, axis=1))         # 均方根
    # 峭度 = E[(x-μ)^4] / σ^4
    centered    = segments - mean_vals[:, np.newaxis]
    std_vals    = np.std(segments, axis=1, ddof=0) + 1e-10        # 防止除零
    kurt_vals   = np.mean(centered ** 4, axis=1) / (std_vals ** 4)
    # 偏度 = E[(x-μ)^3] / σ^3
    skew_vals   = np.mean(centered ** 3, axis=1) / (std_vals ** 3)

    return np.column_stack([mean_vals, rms_vals, kurt_vals, skew_vals])


def extract_frequency_domain(segments, fs=12000):
    """
    提取频域特征：主频（幅值最大的频率）、频谱能量。
    参数:
        segments: shape (n_samples, window_size)
        fs:       采样频率，CWRU默认12kHz
    返回:
        features: shape (n_samples, 2)
    """
    n = segments.shape[1]
    freq = np.fft.fftfreq(n, d=1/fs)
    half_n = n // 2

    dom_freqs = []
    energies = []

    for seg in segments:
        spectrum = np.abs(fft(seg))[:half_n]
        dom_freqs.append(freq[np.argmax(spectrum)])
        energies.append(np.sum(spectrum ** 2))

    return np.column_stack([dom_freqs, energies])


def extract_all_features(segments, fs=12000):
    """
    合并时域和频域特征。
    返回:
        features: shape (n_samples, 6)
        feature_names: 各特征名称列表
    """
    td = extract_time_domain(segments)
    fd = extract_frequency_domain(segments, fs)
    feature_names = ["均值", "均方根(RMS)", "峭度", "偏度", "主频(Hz)", "频谱能量"]
    return np.concatenate([td, fd], axis=1), feature_names


def compute_fft_spectrum(signal, fs=12000):
    """
    计算单段信号的FFT频谱，供可视化使用。
    参数:
        signal: 1D NumPy数组
        fs:     采样频率
    返回:
        freq:   频率轴（取正半轴）
        amplitude: 对应幅值
    """
    n = len(signal)
    spectrum = np.abs(fft(signal))
    half_n = n // 2
    freq = np.fft.fftfreq(n, d=1/fs)[:half_n]
    return freq, spectrum[:half_n]


def perform_pca(features, n_components=2):
    """
    对特征矩阵进行标准化 + PCA降维。
    参数:
        features:    shape (n_samples, n_features)
        n_components: 目标维度（默认2，用于可视化）
    返回:
        pca_result:  shape (n_samples, n_components) 降维结果
        pca_model:   训练好的PCA模型（可用于新数据变换）
        scaler:      训练好的StandardScaler（可用于新数据变换）
    """
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    pca = PCA(n_components=n_components)
    result = pca.fit_transform(scaled)
    return result, pca, scaler


if __name__ == "__main__":
    # 测试特征提取
    from data_loader import build_dataset
    print("=" * 50)
    print("特征提取测试")
    print("=" * 50)
    X, y, names = build_dataset()
    feats, feat_names = extract_all_features(X)
    print(f"  提取特征形状: {feats.shape}")
    print(f"  特征名称: {feat_names}")

    pca_res, pca_model, scaler = perform_pca(feats)
    print(f"  PCA降维后形状: {pca_res.shape}")
    print(f"  PCA解释方差比: {pca_model.explained_variance_ratio_}")
