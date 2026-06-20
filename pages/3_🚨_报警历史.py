"""
报警历史页面 — 查询和浏览历史报警记录。
"""

import os, sys, time, pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alerts.manager import AlarmManager
from alerts.models import AlarmLevel

st.set_page_config(page_title="报警历史", page_icon="🚨", layout="wide")

# CSS
st.markdown("""
<style>
.stApp { background: #0b1120; }
[data-testid="stSidebar"] { background: #0f172a; border-right: 1px solid rgba(56,189,248,0.08); }
[data-testid="stSidebar"] label, [data-testid="stSidebar"] .stMarkdown { color: #94a3b8 !important; }
[data-testid="stSidebar"] h3 { color: #38bdf8 !important; }
.metric-card { background: #131c33; border: 1px solid rgba(56,189,248,0.08); border-radius: 10px; padding: 0.9rem 1.2rem; text-align: center; }
.metric-value { font-size: 1.8rem; font-weight: 800; color: #e2e8f0; font-family: 'Consolas','JetBrains Mono',monospace; }
.metric-label { font-size: 0.72rem; color: #64748b; letter-spacing: 0.5px; }
.critical-row { background: rgba(239,68,68,0.08); border-left: 3px solid #ef4444; border-radius: 6px; padding: 0.5rem 0.8rem; margin: 0.25rem 0; }
.warning-row { background: rgba(245,158,11,0.08); border-left: 3px solid #f59e0b; border-radius: 6px; padding: 0.5rem 0.8rem; margin: 0.25rem 0; }
.stButton > button { background: rgba(56,189,248,0.12); border: 1px solid rgba(56,189,248,0.2); border-radius: 8px; color: #38bdf8; font-weight: 600; }
[data-testid="stDataFrame"] { background: #131c33 !important; }
[data-testid="stDataFrame"] th { background: #1a2747 !important; color: #94a3b8 !important; }
[data-testid="stDataFrame"] td { color: #cbd5e1 !important; }
.footer { text-align: center; padding: 0.3rem 0; font-size: 0.68rem; color: #475569; }
</style>
""", unsafe_allow_html=True)

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "alarms.db")

# 侧边栏
with st.sidebar:
    st.markdown('<div style="text-align:center;padding:0.5rem 0 0.8rem 0;"><div style="font-size:2rem;">🚨</div><div style="font-size:1rem;font-weight:700;color:#e2e8f0;letter-spacing:1px;">报警历史查询</div></div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### ⏱ 筛选条件")
    tf = st.radio("时间范围", ["全部","最近 1 分钟","最近 5 分钟","最近 30 分钟"], label_visibility="collapsed")
    level_filter = st.selectbox("报警等级", ["全部", "WARNING", "CRITICAL"], label_visibility="collapsed")
    st.markdown("---")
    if st.button("🔄 刷新", use_container_width=True): st.rerun()

mgr = AlarmManager(device_id="bearing_001", db_path=DB_PATH)
now = time.time()
tr_map = {"全部":None,"最近 1 分钟":(now-60,now),"最近 5 分钟":(now-300,now),"最近 30 分钟":(now-1800,now)}
tr = tr_map[tf]
lv = AlarmLevel(level_filter) if level_filter != "全部" else None

records = mgr.query(time_range=tr, level=lv, limit=1000)
stats = mgr.stats()

# 主区域
st.markdown('<p style="font-size:1.3rem;font-weight:700;color:#e2e8f0;margin-bottom:0;">🚨 报警历史记录</p>', unsafe_allow_html=True)
st.markdown(f'<p style="font-size:0.78rem;color:#64748b;margin-top:0;">WARNING: {stats["warnings"]} | CRITICAL: {stats["criticals"]} | 总计: {stats["total_alarms"]}</p>', unsafe_allow_html=True)

# 统计卡
mc = st.columns(3)
mc[0].markdown(f'<div class="metric-card"><div class="metric-value" style="color:#f59e0b;">{stats["warnings"]}</div><div class="metric-label">WARNING</div></div>', unsafe_allow_html=True)
mc[1].markdown(f'<div class="metric-card"><div class="metric-value" style="color:#ef4444;">{stats["criticals"]}</div><div class="metric-label">CRITICAL</div></div>', unsafe_allow_html=True)
mc[2].markdown(f'<div class="metric-card"><div class="metric-value" style="color:#38bdf8;">{stats["total_alarms"]}</div><div class="metric-label">总报警</div></div>', unsafe_allow_html=True)

st.markdown("---")

if not records:
    st.info("暂无报警记录。请在「实时诊断」页面启动推流，连续检测到故障时将自动生成报警。")
    st.caption("💡 在线演示版 | 报警记录可能在应用重启后重置")
else:
    st.markdown("#### 📋 报警列表")
    for r in records[:50]:
        is_crit = r["alarm_level"] == "CRITICAL"
        cls = "critical-row" if is_crit else "warning-row"
        ts = pd.to_datetime(r["timestamp"], unit="s").strftime("%m-%d %H:%M:%S")
        badge = "🔴 CRITICAL" if is_crit else "🟡 WARNING"
        st.markdown(f'<div class="{cls}"><span style="font-weight:600;">{badge}</span> &nbsp; <span style="color:#94a3b8;">{ts}</span> &nbsp; <b>{r["fault_type"]}</b> &nbsp; <span style="color:#64748b;">置信度 {r["confidence"]:.1%}</span><br><span style="font-size:0.8rem;color:#6b7280;">{r["message"][:100]}</span></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 📝 数据表")
    df = pd.DataFrame(records)
    dc = {"timestamp":"时间","device_id":"设备","fault_type":"故障类型","confidence":"置信度","alarm_level":"等级","message":"说明"}
    df = df[list(dc.keys())].rename(columns=dc)
    df["时间"] = pd.to_datetime(df["时间"],unit="s").dt.strftime("%H:%M:%S")
    df["置信度"] = df["置信度"].apply(lambda x: f"{x:.1%}")
    st.dataframe(df, use_container_width=True, height=400, hide_index=True)

st.markdown('<p class="footer">报警数据库: alarms.db &nbsp;|&nbsp; 在线演示版 · 记录可能在应用重启后重置</p>', unsafe_allow_html=True)
