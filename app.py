"""
工业轴承故障智能诊断系统 — 首页
"""

import os, sys, numpy as np, matplotlib.pyplot as plt, matplotlib.font_manager as fm
import seaborn as sns, streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(page_title="轴承故障智能诊断", page_icon="⚙️", layout="wide",
                   initial_sidebar_state="collapsed")

# ═══════════════════════════════════════════════
# 工业暗色主题 CSS
# ═══════════════════════════════════════════════
st.markdown("""
<style>
:root {
    --bg-primary: #0b1120;
    --bg-card: #131c33;
    --bg-card-hover: #182544;
    --border: rgba(56,189,248,0.12);
    --accent: #38bdf8;
    --accent2: #818cf8;
    --green: #10b981;
    --amber: #f59e0b;
    --red: #ef4444;
    --text-primary: #e2e8f0;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
}
.stApp { background: #0b1120; }
.main .block-container { padding-top: 1.2rem; max-width: 1400px; }

/* 卡片 */
.card {
    background: #131c33;
    border: 1px solid rgba(56,189,248,0.10);
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    transition: all 0.25s;
    height: 100%;
}
.card:hover {
    border-color: rgba(56,189,248,0.30);
    background: #182544;
    box-shadow: 0 0 24px rgba(56,189,248,0.06);
}
.card-icon { font-size: 2.4rem; margin-bottom: 0.6rem; }
.card-title { font-size: 1.05rem; font-weight: 700; color: #e2e8f0; margin-bottom: 0.4rem; }
.card-desc { font-size: 0.82rem; color: #94a3b8; line-height: 1.5; }
.card-tag {
    display: inline-block; margin-top: 0.8rem; padding: 0.25rem 0.75rem;
    border-radius: 20px; font-size: 0.75rem; font-weight: 600;
    background: rgba(56,189,248,0.1); color: #38bdf8; border: 1px solid rgba(56,189,248,0.2);
}

/* 指标卡 */
.metric-card {
    background: linear-gradient(135deg, #131c33 0%, #1a2747 100%);
    border: 1px solid rgba(56,189,248,0.10);
    border-radius: 12px; padding: 1.1rem 1.4rem; text-align: center;
}
.metric-value {
    font-size: 2.2rem; font-weight: 800; color: #e2e8f0;
    font-family: 'Consolas', 'JetBrains Mono', monospace;
}
.metric-label { font-size: 0.78rem; color: #64748b; margin-top: 0.2rem; letter-spacing: 0.5px; }
.metric-accent { color: #38bdf8; }
.metric-green { color: #10b981; }
.metric-amber { color: #f59e0b; }

/* 标题 */
.hero {
    font-size: 2.2rem; font-weight: 800; color: #e2e8f0;
    letter-spacing: -0.5px; margin-bottom: 0.1rem;
}
.hero-accent { background: linear-gradient(135deg, #38bdf8, #818cf8);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.subtitle { font-size: 0.9rem; color: #64748b; letter-spacing: 0.3px; }

/* 分隔线 */
.divider {
    height: 1px; margin: 1.5rem 0;
    background: linear-gradient(90deg, transparent, rgba(56,189,248,0.15), transparent);
}

/* 架构节点 */
.arch-node {
    text-align: center; padding: 1.2rem 0.6rem;
    background: rgba(19,28,51,0.7); border-radius: 10px;
    border: 1px solid rgba(56,189,248,0.08);
}
.arch-icon { font-size: 2rem; margin-bottom: 0.4rem; }
.arch-title { font-weight: 700; color: #cbd5e1; font-size: 0.9rem; margin-bottom: 0.3rem; }
.arch-desc { font-size: 0.72rem; color: #64748b; line-height: 1.4; }

/* 按钮 */
.stButton > button {
    background: linear-gradient(135deg, rgba(56,189,248,0.15), rgba(129,140,248,0.15));
    border: 1px solid rgba(56,189,248,0.25); border-radius: 10px;
    color: #38bdf8; font-weight: 600; padding: 0.5rem 1.2rem;
    transition: all 0.25s;
}
.stButton > button:hover {
    background: linear-gradient(135deg, rgba(56,189,248,0.25), rgba(129,140,248,0.25));
    border-color: rgba(56,189,248,0.5); color: #e2e8f0;
}

/* 页脚 */
.footer { text-align: center; padding: 0.5rem 0; font-size: 0.72rem; color: #475569; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════
# 中文字体（优先级: Noto Sans CJK SC > Microsoft YaHei > SimHei > DejaVu Sans）
# ═══════════════════════════════════════════════
_af = {f.name for f in fm.fontManager.ttflist}
for _f in ["Noto Sans CJK SC","Microsoft YaHei","SimHei","KaiTi","SimSun"]:
    if _f in _af:
        _p = next((x.fname for x in fm.fontManager.ttflist if x.name == _f), None)
        if _p: fm.fontManager.addfont(_p)
sns.set_style("darkgrid")
plt.rcParams.update({
    "axes.unicode_minus": False,
    "font.sans-serif": ["Noto Sans CJK SC","Microsoft YaHei","SimHei","DejaVu Sans"],
    "font.family": "sans-serif",
    "figure.facecolor": "#0b1120",
    "axes.facecolor": "#111827",
    "axes.edgecolor": "#1e293b",
    "axes.labelcolor": "#94a3b8",
    "text.color": "#cbd5e1",
    "xtick.color": "#64748b",
    "ytick.color": "#64748b",
    "grid.color": "#1e293b",
    "grid.alpha": 0.6,
})

# ═══════════════════════════════════════════════
# 数据（从 demo_data/ 加载演示统计）
# ═══════════════════════════════════════════════
from core.signal.demo_loader import load_dataset_stats, build_demo_dataset
from core.signal.loader import LABEL_MAP
from core.persistence.store import DiagnosticsStore

LABEL_DICT = {v[0]: v[1] for v in LABEL_MAP.values()}
_demo_stats = load_dataset_stats()
_demo_samples = 0
for _cls_info in _demo_stats.get("files", {}).values():
    _demo_len = _cls_info.get("signal_length", 0)
    _demo_samples += (_demo_len - 1024) // 512 + 1

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diagnostics.db")
store = DiagnosticsStore(memory_size=5000, db_path=DB_PATH)
stats = store.stats()

# ═══════════════════════════════════════════════
# 页面渲染
# ═══════════════════════════════════════════════
st.markdown('<p class="hero"><span class="hero-accent">⚙️ 工业轴承故障智能诊断系统</span></p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">CWRU DATASET &nbsp;·&nbsp; PCA + SVM &nbsp;·&nbsp; 1D-CNN &nbsp;·&nbsp; 实时流式推理 & 历史趋势分析</p>', unsafe_allow_html=True)
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# 指标卡
mc = st.columns(5)
mc[0].markdown(f'<div class="metric-card"><div class="metric-value metric-accent">{_demo_samples:,}</div><div class="metric-label">演示窗口样本</div></div>', unsafe_allow_html=True)
mc[1].markdown(f'<div class="metric-card"><div class="metric-value">4</div><div class="metric-label">故障类别</div></div>', unsafe_allow_html=True)
mc[2].markdown(f'<div class="metric-card"><div class="metric-value">2</div><div class="metric-label">诊断算法</div></div>', unsafe_allow_html=True)
mc[3].markdown(f'<div class="metric-card"><div class="metric-value metric-green">{stats["total_records"]:,}</div><div class="metric-label">历史诊断次数</div></div>', unsafe_allow_html=True)
mc[4].markdown(f'<div class="metric-card"><div class="metric-value metric-amber">{stats["total_alarms"]:,}</div><div class="metric-label">历史报警次数</div></div>', unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# 导航
st.markdown('<p style="font-size:1.1rem;font-weight:700;color:#cbd5e1;margin-bottom:0.8rem;">📋 功能导航</p>', unsafe_allow_html=True)
nav = st.columns(2)
with nav[0]:
    st.markdown('<div class="card"><div class="card-icon">📡</div><div class="card-title">实时故障诊断</div><div class="card-desc">模拟工业传感器持续上报振动数据，基于环形缓冲区实现流式滑动窗口在线推理。支持 SVM / 1D-CNN 模型实时切换、传感器噪声模拟、连续故障报警检测。</div><div class="card-tag">实时推流 · 波形监控 · 置信度趋势</div></div>', unsafe_allow_html=True)
    st.page_link("pages/1_📈_实时诊断.py", label="进入实时诊断 →", use_container_width=True)
with nav[1]:
    st.markdown('<div class="card"><div class="card-icon">📊</div><div class="card-title">历史趋势分析</div><div class="card-desc">查询诊断历史数据库，分析故障类型分布比例、PCA 特征空间漂移轨迹、报警事件时间线。支持 SQLite 持久化存储与自定义时间段筛选。</div><div class="card-tag">PCA 轨迹 · 故障分布 · 报警日志</div></div>', unsafe_allow_html=True)
    st.page_link("pages/2_📊_历史趋势.py", label="进入历史趋势 →", use_container_width=True)

nav2 = st.columns(2)
with nav2[0]:
    st.markdown('<div class="card"><div class="card-icon">🚨</div><div class="card-title">报警历史查询</div><div class="card-desc">浏览所有历史报警记录，按时间和等级筛选。支持 WARNING / CRITICAL 分级展示，每条报警包含触发原因和详细说明。</div><div class="card-tag">分级报警 · 时间筛选 · 详情查看</div></div>', unsafe_allow_html=True)
    st.page_link("pages/3_🚨_报警历史.py", label="进入报警历史 →", use_container_width=True)
with nav2[1]:
    st.markdown('<div class="card"><div class="card-icon">📊</div><div class="card-title">报警统计分析</div><div class="card-desc">可视化报警统计概览：等级分布、故障类型占比、时间线趋势。帮助运维人员快速掌握设备健康态势。</div><div class="card-tag">等级分布 · 故障占比 · 时间线</div></div>', unsafe_allow_html=True)
    st.page_link("pages/4_📊_报警统计.py", label="进入报警统计 →", use_container_width=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# 架构
st.markdown('<p style="font-size:1.1rem;font-weight:700;color:#cbd5e1;margin-bottom:0.8rem;">🏗 系统架构</p>', unsafe_allow_html=True)
ac = st.columns(4)
ac[0].markdown('<div class="arch-node"><div class="arch-icon">📂</div><div class="arch-title">数据层</div><div class="arch-desc">CWRU 数据集<br>.mat.txt 文件<br>StreamSimulator</div></div>', unsafe_allow_html=True)
ac[1].markdown('<div class="arch-node"><div class="arch-icon">⚙️</div><div class="arch-title">处理层</div><div class="arch-desc">RingBuffer<br>滑动窗口切片<br>时频域特征提取</div></div>', unsafe_allow_html=True)
ac[2].markdown('<div class="arch-node"><div class="arch-icon">🧠</div><div class="arch-title">推理层</div><div class="arch-desc">PCA + SVM<br>1D-CNN 深度学习<br>连续报警判定</div></div>', unsafe_allow_html=True)
ac[3].markdown('<div class="arch-node"><div class="arch-icon">📊</div><div class="arch-title">展示层</div><div class="arch-desc">Streamlit 仪表盘<br>实时监控面板<br>历史趋势分析</div></div>', unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown('<p class="footer">INDUSTRIAL BEARING FAULT DIAGNOSIS SYSTEM &nbsp;·&nbsp; CWRU DATASET &nbsp;·&nbsp; PCA + SVM & 1D-CNN</p>', unsafe_allow_html=True)
st.caption("💡 在线演示版 | 数据为 CWRU 真实信号子集 | 诊断记录可能在应用重启后重置")
