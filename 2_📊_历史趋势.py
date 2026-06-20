"""
历史趋势与报警分析 — 工业暗色风格。
"""

import os, sys, time, numpy as np, pandas as pd
import matplotlib.pyplot as plt, matplotlib.font_manager as fm, seaborn as sns
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.persistence.store import DiagnosticsStore
from core.signal.loader import LABEL_MAP
from core.signal.demo_loader import build_demo_dataset
from core.signal.features import extract_all_features, perform_pca

st.set_page_config(page_title="历史趋势", page_icon="📊", layout="wide")

# ═══════════════════════════════════════════════
# 暗色 CSS
# ═══════════════════════════════════════════════
st.markdown("""
<style>
.stApp { background: #0b1120; }
[data-testid="stSidebar"] { background: #0f172a; border-right: 1px solid rgba(56,189,248,0.08); }
[data-testid="stSidebar"] label, [data-testid="stSidebar"] .stMarkdown { color: #94a3b8 !important; }
[data-testid="stSidebar"] h3 { color: #38bdf8 !important; font-size: 0.95rem; letter-spacing: 1px; }
[data-testid="stSidebar"] hr { border-color: rgba(56,189,248,0.08); }
.metric-card { background: #131c33; border: 1px solid rgba(56,189,248,0.08); border-radius: 10px; padding: 0.9rem 1.2rem; text-align: center; }
.metric-value { font-size: 1.8rem; font-weight: 800; color: #e2e8f0; font-family: 'Consolas','JetBrains Mono',monospace; }
.metric-label { font-size: 0.72rem; color: #64748b; letter-spacing: 0.5px; }
.metric-accent { color: #38bdf8; }
.metric-green { color: #10b981; }
.metric-red { color: #ef4444; }
.metric-amber { color: #f59e0b; }
.alarm-row { background: rgba(239,68,68,0.06); border-left: 3px solid #ef4444; border-radius: 6px; padding: 0.5rem 0.8rem; margin: 0.25rem 0; font-size: 0.82rem; color: #cbd5e1; }
.stButton > button { background: rgba(56,189,248,0.12); border: 1px solid rgba(56,189,248,0.2); border-radius: 8px; color: #38bdf8; font-weight: 600; }
.stButton > button:hover { background: rgba(56,189,248,0.22); border-color: rgba(56,189,248,0.5); color: #e2e8f0; }
[data-testid="stDataFrame"] { background: #131c33 !important; border: 1px solid rgba(56,189,248,0.06); }
[data-testid="stDataFrame"] th { background: #1a2747 !important; color: #94a3b8 !important; }
[data-testid="stDataFrame"] td { color: #cbd5e1 !important; }
.footer { text-align: center; padding: 0.3rem 0; font-size: 0.68rem; color: #475569; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════
# 字体（优先级: Noto Sans CJK SC > Microsoft YaHei > SimHei > DejaVu Sans）
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
    "grid.alpha": 0.4,
})

LABEL_DICT = {v[0]: v[1] for v in LABEL_MAP.values()}
COLORS = ["#10b981", "#ef4444", "#f59e0b", "#8b5cf6"]
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "diagnostics.db")

# ═══════════════════════════════════════════════
# 侧边栏
# ═══════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div style="text-align:center;padding:0.5rem 0 0.8rem 0;"><div style="font-size:2rem;">📊</div><div style="font-size:1rem;font-weight:700;color:#e2e8f0;letter-spacing:1px;">历史趋势分析</div></div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### ⏱ 数据筛选")
    tf = st.radio("时间范围", ["全部","最近 1 分钟","最近 5 分钟","最近 30 分钟"], label_visibility="collapsed")
    st.markdown("---")
    if st.button("🔄 刷新数据", use_container_width=True): st.rerun()
    if st.button("🗑 清除历史", use_container_width=True):
        DiagnosticsStore(memory_size=5000, db_path=DB_PATH).clear(); st.success("已清除。"); st.rerun()

now = time.time()
tr_map = {"全部":None,"最近 1 分钟":(now-60,now),"最近 5 分钟":(now-300,now),"最近 30 分钟":(now-1800,now)}
tr = tr_map[tf]

store = DiagnosticsStore(memory_size=5000, db_path=DB_PATH)
records = store.query(time_range=tr, limit=5000)
alarms = store.query_alarms(time_range=tr, limit=500)
stats = store.stats()

# ═══════════════════════════════════════════════
# 主区域
# ═══════════════════════════════════════════════
st.markdown('<p style="font-size:1.3rem;font-weight:700;color:#e2e8f0;margin-bottom:0;">📊 历史趋势与报警分析</p>', unsafe_allow_html=True)
st.markdown(f'<p style="font-size:0.78rem;color:#64748b;margin-top:0;">时间范围: {tf} &nbsp;|&nbsp; 诊断记录: {len(records)} &nbsp;|&nbsp; 报警: {len(alarms)}</p>', unsafe_allow_html=True)

# 统计卡片
mc = st.columns(4)
mc[0].markdown(f'<div class="metric-card"><div class="metric-value metric-accent">{stats["total_records"]:,}</div><div class="metric-label">历史总诊断</div></div>', unsafe_allow_html=True)
mc[1].markdown(f'<div class="metric-card"><div class="metric-value metric-red">{stats["total_alarms"]:,}</div><div class="metric-label">总报警次数</div></div>', unsafe_allow_html=True)
mc[2].markdown(f'<div class="metric-card"><div class="metric-value metric-amber">{stats["total_alarms"]/max(stats["total_records"],1):.1%}</div><div class="metric-label">报警率</div></div>', unsafe_allow_html=True)
ago = now - records[0]["timestamp"] if records else 0
mc[3].markdown(f'<div class="metric-card"><div class="metric-value metric-green">{f"{ago:.0f}秒前" if ago<120 else f"{ago/60:.0f}分钟前" if records else "N/A"}</div><div class="metric-label">最近诊断</div></div>', unsafe_allow_html=True)

if not records:
    st.info("暂无诊断历史。请先在「实时诊断」页面启动推流。")
    st.stop()

# ═══════════════════════════════════════════════
# 第一行: 时间线 + 故障分布
# ═══════════════════════════════════════════════
col1, col2 = st.columns(2)
with col1:
    st.markdown('<p style="font-weight:700;color:#cbd5e1;margin-bottom:0.3rem;">📋 诊断时间线</p>', unsafe_allow_html=True)
    df = pd.DataFrame(records)
    df["t"] = pd.to_datetime(df["timestamp"],unit="s").dt.strftime("%H:%M:%S")
    step = max(1,len(df)//300)
    ds = df.iloc[::step]
    fig, ax = plt.subplots(figsize=(7,3.2),dpi=100)
    ax.scatter(range(len(ds)), ds["confidence"], c=[COLORS[l] if l<4 else "#666" for l in ds["predicted_label"]], s=3, alpha=0.7)
    ax.set_xlabel("诊断序号",fontsize=7); ax.set_ylabel("置信度",fontsize=7); ax.set_ylim(0,1.02)
    ax.set_title("诊断置信度历史",fontsize=9,fontweight=600,color="#cbd5e1"); ax.tick_params(labelsize=6)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=c,label=n,alpha=0.7) for n,c in zip(LABEL_DICT.values(),COLORS)], fontsize=6, loc="upper right", ncol=2, framealpha=0.3)
    fig.tight_layout(pad=1); st.pyplot(fig); plt.close(fig)

with col2:
    st.markdown('<p style="font-weight:700;color:#cbd5e1;margin-bottom:0.3rem;">🥧 故障类型分布</p>', unsafe_allow_html=True)
    dist = stats.get("distribution",{})
    if dist:
        labels, vals, cs = [], [], []
        for i,n in LABEL_DICT.items():
            cnt = dist.get(n,0); labels.append(f"{n} ({cnt})"); vals.append(cnt); cs.append(COLORS[i])
        fig, ax = plt.subplots(figsize=(5.5,3.2),dpi=100)
        w,_,at = ax.pie(vals,labels=None,colors=cs,autopct="%1.1f%%",startangle=90,wedgeprops={"edgecolor":"#0b1120","linewidth":2})
        for t in at: t.set_fontsize(8); t.set_fontweight("bold")
        ax.legend(w,labels,fontsize=6,loc="lower center",ncol=2,framealpha=0.3)
        ax.set_title("故障类型比例分布",fontsize=9,fontweight=600,color="#cbd5e1")
        fig.tight_layout(pad=1.2); st.pyplot(fig); plt.close(fig)

# ═══════════════════════════════════════════════
# 第二行: PCA 轨迹 + 报警日志
# ═══════════════════════════════════════════════
col3, col4 = st.columns(2)
with col3:
    st.markdown('<p style="font-weight:700;color:#cbd5e1;margin-bottom:0.3rem;">🗺️ PCA 特征轨迹</p>', unsafe_allow_html=True)
    st.caption("PCA 背景基于演示数据集 (demo_data/) 有限窗口计算")
    if len(records) >= 10:
        try:
            X_demo, y_demo, _ = build_demo_dataset(window_size=1024, step=512)
            feats_demo, _ = extract_all_features(X_demo)
            _, pca_m, scl = perform_pca(feats_demo)

            hf = np.array([[r.get("mean_val",0),r.get("rms",0),r.get("kurtosis",0),r.get("skewness",0),r.get("dom_freq",0),r.get("spectral_energy",0)] for r in records[-500:]])
            hp = pca_m.transform(scl.transform(hf))

            fig, ax = plt.subplots(figsize=(6.5,4.2),dpi=100)
            dp = pca_m.transform(scl.transform(feats_demo))
            for i,(n,c) in enumerate(zip(LABEL_DICT.values(),COLORS)):
                m = y_demo==i; ax.scatter(dp[m,0],dp[m,1],c=c,label=n,alpha=0.08,s=4,edgecolors="none")
            ax.plot(hp[:,0],hp[:,1],color="#94a3b8",linewidth=0.6,alpha=0.5,zorder=5)
            ax.scatter(hp[-1,0],hp[-1,1],c="#ef4444",s=70,marker="X",zorder=10,edgecolors="white",linewidth=1,label="当前位置")
            ax.scatter(hp[0,0],hp[0,1],c="#38bdf8",s=50,marker="o",zorder=10,edgecolors="white",linewidth=1,label="起始位置")
            ax.set_xlabel(f"PC 1 ({pca_m.explained_variance_ratio_[0]:.1%})",fontsize=7)
            ax.set_ylabel(f"PC 2 ({pca_m.explained_variance_ratio_[1]:.1%})",fontsize=7)
            ax.set_title("PCA 特征空间轨迹",fontsize=9,fontweight=600,color="#cbd5e1")
            ax.legend(fontsize=6,markerscale=1.2,framealpha=0.3,loc="upper right"); ax.tick_params(labelsize=6)
            fig.tight_layout(pad=1.2); st.pyplot(fig); plt.close(fig)
            st.caption("蓝色圆 = 起始 · 红色 X = 当前 · 灰线 = 移动轨迹")
        except Exception as e: st.warning(f"PCA 失败: {e}")
    else: st.info("需要至少 10 条记录。")

with col4:
    st.markdown('<p style="font-weight:700;color:#cbd5e1;margin-bottom:0.3rem;">🚨 报警日志</p>', unsafe_allow_html=True)
    if alarms:
        for r in alarms[:20]:
            ts = pd.to_datetime(r["timestamp"],unit="s").strftime("%m-%d %H:%M:%S")
            st.markdown(f'<div class="alarm-row"><span style="color:#fca5a5;font-weight:600;">🚨 {ts}</span> &nbsp;{r["fault_type"]}&nbsp; <span style="color:#94a3b8;">置信度 {r["confidence"]:.1%}</span></div>', unsafe_allow_html=True)
    else: st.info("暂无报警记录。")

    st.markdown('<p style="font-weight:700;color:#cbd5e1;margin-top:1rem;margin-bottom:0.3rem;">📝 最近诊断记录</p>', unsafe_allow_html=True)
    df2 = pd.DataFrame(records[:15])
    dc = {"timestamp":"时间","predicted_name":"诊断结果","confidence":"置信度","rms":"RMS","alarm":"报警"}
    df2 = df2[list(dc.keys())].rename(columns=dc)
    df2["时间"] = pd.to_datetime(df2["时间"],unit="s").dt.strftime("%H:%M:%S")
    df2["置信度"] = df2["置信度"].apply(lambda x: f"{x:.1%}")
    df2["RMS"] = df2["RMS"].apply(lambda x: f"{x:.4f}")
    df2["报警"] = df2["报警"].apply(lambda x: "🚨" if x else "")
    st.dataframe(df2, use_container_width=True, height=340, hide_index=True)

st.markdown('<p class="footer">数据源: diagnostics.db (demo_data/) &nbsp;|&nbsp; 时间范围: {} &nbsp;|&nbsp; {} 条记录 &nbsp;|&nbsp; {} 条报警</p>'.format(tf, len(records), len(alarms)), unsafe_allow_html=True)
st.caption("💡 在线演示版 | 诊断记录可能在应用重启后重置")
