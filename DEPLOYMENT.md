# 工业轴承故障诊断系统 — Streamlit Cloud 部署指南

## 项目说明

本项目是 **工业轴承故障诊断与状态监测系统** 的在线演示版本，专为招聘展示设计。

- **数据**: 真实 CWRU 12k Drive End 振动信号子集 (`demo_data/`)
- **模型**: 已训练好的 PCA+SVM (scikit-learn) 和 1D-CNN (PyTorch) — 跨负载泛化模型
- **入口**: `app.py`
- **前端**: Streamlit 仪表盘（工业暗色主题）

⚠️ **本版本不是工业生产部署**。在线记录在应用重启后重置（Streamlit Cloud 为无状态容器）。

---

## 本地启动

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

如果系统缺少中文字体（Linux），额外安装：

```bash
# Debian/Ubuntu (Streamlit Cloud 使用 packages.txt 自动安装)
sudo apt-get install fonts-noto-cjk
```

### 2. 生成演示数据

```bash
python tools/build_demo_assets.py
```

此脚本从 `data/` 目录读取真实 CWRU 数据，截取 65536 点/类，输出到 `demo_data/`。

### 3. 启动应用

```bash
streamlit run app.py
```

浏览器访问 `http://localhost:8501`。

---

## GitHub 上传

确保以下文件/目录已提交：

```
app.py                       # 首页入口
pages/                       # Streamlit 子页面
demo_data/                   # 演示数据（4 个 .npz + dataset_stats.json）
models/saved/                # 训练好的模型文件
results/                     # 实验结果报告
core/                        # 核心模块
stream/                      # 流处理模块
alerts/                      # 报警模块
tools/build_demo_assets.py   # 演示数据生成脚本
requirements.txt             # Python 依赖
packages.txt                 # 系统级依赖（中文字体）
.streamlit/config.toml       # Streamlit 配置
DEPLOYMENT.md                # 本文件
```

**不要上传**：`data/`（完整原始数据）、`*.db`（运行时数据库）、`__pycache__/`。

---

## Streamlit Community Cloud 部署

1. 将项目推送到 **公开** GitHub 仓库。
2. 在 [share.streamlit.io](https://share.streamlit.io) 点击 **New app**。
3. 选择仓库、分支，**Main file path** 设为 `app.py`。
4. **Python version** 选择 **3.12**。
5. 点击 **Deploy!**

首次部署时 Streamlit Cloud 会自动：
- 安装 `requirements.txt` 中的 Python 包
- 安装 `packages.txt` 中的 `fonts-noto-cjk`（中文字体）

---

## 页面功能

| 页面 | 功能 |
|------|------|
| 首页 | 系统概览、指标卡、架构图 |
| 实时诊断 | 模拟传感器推流 + SVM/CNN 在线推理 + 波形/FFT/报警 |
| 历史趋势 | 诊断时间线、故障分布饼图、PCA 轨迹、报警日志 |
| 报警历史 | 分级报警查询（WARNING/CRITICAL）、时间筛选 |
| 报警统计 | 报警等级分布、故障类型占比、事件时间线 |

---

## 技术栈

- **信号处理**: NumPy, SciPy
- **机器学习**: scikit-learn (PCA + SVM), joblib
- **深度学习**: PyTorch (1D-CNN)
- **可视化**: Matplotlib, Seaborn
- **仪表盘**: Streamlit
- **存储**: SQLite (运行时，重启后重置)
