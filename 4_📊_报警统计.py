"""
报警统计页面 — 可视化报警趋势、分布和概览。
"""

import os, sys, time, numpy as np, pandas as pd
import matplotlib.pyplot as plt, matplotlib.font_manager as fm, seaborn as sns
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alerts.manager import AlarmManager
from alerts.models import AlarmLevel

st.set_page_config(page_title="报警统计", page_icon="📊", layout="wide")

# CSS
st.markdown("""
<style>
.stApp { background: #0b1120; }
.metric-card { background: #131c33; border: 1px solid rgba(56,189,248,0.08); border-radius: 10px; padding: 0.9rem 1.2rem; text-align: center; }
.metric-value { font-size: 1.8rem; font-weight: 800; color: #e2e8f0; font-family: 'Consolas','JetBrains Mono',monospace; }
.metric-label { font-size: 0.72rem; color: #64748b; letter-spacing: 0.5px; }
[data-testid="stDataFrame"] { background: #131c33 !important; }
[data-testid="stDataFrame"] th { background: #1a2747 !important; color: #94a3b8 !important; }
.footer { text-align: center; padding: 0.3rem 0; font-size: 0.68rem; color: #475569; }
</style>
""", unsafe_allow_html=True)

# 字体（优先级: Noto Sans CJK SC > Microsoft YaHei > SimHei > DejaVu Sans）
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
})

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "alarms.db")
mgr = AlarmManager(device_id="bearing_001", db_path=DB_PATH)
stats = mgr.stats()
records = mgr.query(limit=5000)

st.markdown('<p style="font-size:1.3rem;font-weight:700;color:#e2e8f0;margin-bottom:0;">📊 报警统计概览</p>', unsafe_allow_html=True)

# 统计卡
mc = st.columns(4)
mc[0].markdown(f'<div class="metric-card"><div class="metric-value" style="color:#f59e0b;">{stats["warnings"]}</div><div class="metric-label">WARNING</div></div>', unsafe_allow_html=True)
mc[1].markdown(f'<div class="metric-card"><div class="metric-value" style="color:#ef4444;">{stats["criticals"]}</div><div class="metric-label">CRITICAL</div></div>', unsafe_allow_html=True)
mc[2].markdown(f'<div class="metric-card"><div class="metric-value" style="color:#38bdf8;">{stats["total_alarms"]}</div><div class="metric-label">总报警</div></div>', unsafe_allow_html=True)
cr = stats["criticals"] / max(stats["total_alarms"], 1)
mc[3].markdown(f'<div class="metric-card"><div class="metric-value" style="color:#ef4444;">{cr:.0%}</div><div class="metric-label">严重率</div></div>', unsafe_allow_html=True)

if not records:
    st.info("暂无报警数据。请在「实时诊断」页面启动推流，连续检测到故障时将自动生成报警记录。")
    st.caption("💡 在线演示版 | 报警记录可能在应用重启后重置")
    st.stop()

st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 🥧 故障类型分布")
    dist = stats.get("by_fault_type", {})
    if dist:
        labels = list(dist.keys())
        vals = list(dist.values())
        colors = ["#ef4444","#f59e0b","#8b5cf6","#ec4899"]
        fig, ax = plt.subplots(figsize=(5.5,3.5),dpi=100)
        w,_,at = ax.pie(vals,labels=None,colors=colors[:len(labels)],autopct="%1.1f%%",startangle=90,wedgeprops={"edgecolor":"#0b1120","linewidth":2})
        for t in at: t.set_fontsize(8); t.set_fontweight("bold")
        ax.legend(w,[f"{l} ({v})" for l,v in zip(labels,vals)],fontsize=7,loc="lower center",ncol=2,framealpha=0.3)
        ax.set_title("报警故障类型分布",fontsize=9,fontweight=600,color="#cbd5e1")
        fig.tight_layout(pad=1.2); st.pyplot(fig); plt.close(fig)

with col2:
    st.markdown("#### 📊 等级分布")
    lvl_labels = ["WARNING","CRITICAL"]
    lvl_vals = [stats["warnings"],stats["criticals"]]
    lvl_colors = ["#f59e0b","#ef4444"]
    fig, ax = plt.subplots(figsize=(5.5,3.5),dpi=100)
    bars = ax.bar(lvl_labels,lvl_vals,color=lvl_colors,width=0.4,edgecolor="white",linewidth=0.5)
    for bar, val in zip(bars,lvl_vals):
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+max(lvl_vals)*0.02,str(val),ha="center",fontsize=10,fontweight=700,color="#e2e8f0")
    ax.set_ylabel("次数",fontsize=7); ax.tick_params(labelsize=7)
    ax.set_title("报警等级分布",fontsize=9,fontweight=600,color="#cbd5e1")
    fig.tight_layout(pad=1.2); st.pyplot(fig); plt.close(fig)

st.markdown("---")

if len(records) >= 2:
    st.markdown("#### 📈 报警时间线")
    df = pd.DataFrame(records)
    df["time_str"] = pd.to_datetime(df["timestamp"],unit="s")
    df = df.sort_values("timestamp")

    fig, ax = plt.subplots(figsize=(12,2.5),dpi=100)
    colors_map = {"WARNING":"#f59e0b","CRITICAL":"#ef4444"}
    for lvl in ["WARNING","CRITICAL"]:
        sub = df[df["alarm_level"]==lvl]
        if len(sub)>0:
            ax.scatter(pd.to_datetime(sub["timestamp"],unit="s"),[1]*len(sub),
                      c=colors_map[lvl],label=lvl,s=50,alpha=0.8,marker="s")
    ax.set_ylim(0.5,1.5); ax.set_yticks([])
    ax.set_xlabel("时间",fontsize=7); ax.legend(fontsize=7,loc="upper right",framealpha=0.3)
    ax.set_title("报警事件时间线",fontsize=9,fontweight=600,color="#cbd5e1"); ax.tick_params(labelsize=6)
    fig.tight_layout(pad=1); st.pyplot(fig); plt.close(fig)

st.markdown('<p class="footer">报警数据库: alarms.db | 统计时间: 全部历史 | 在线演示版 · 记录可能在应用重启后重置</p>', unsafe_allow_html=True)
