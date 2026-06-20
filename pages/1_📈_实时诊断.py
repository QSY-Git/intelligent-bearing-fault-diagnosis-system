"""
实时故障诊断仪表盘 — 工业 SCADA 风格，模拟传感器持续推流 + 在线推理。
"""

import os, sys, time, numpy as np, pandas as pd
from dataclasses import asdict
import matplotlib.pyplot as plt, matplotlib.font_manager as fm, seaborn as sns
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stream.stream_simulator import StreamSimulator
from core.signal.demo_loader import load_demo_signal, FAULT_TYPES_CN
from core.engine.ring_buffer import RingBuffer
from core.engine.inference import InferenceEngine
from core.persistence.store import DiagnosticsStore
from core.persistence.models import DiagnosisRecord
from core.signal.loader import LABEL_MAP
from core.signal.features import compute_fft_spectrum, extract_all_features
from alerts.manager import AlarmManager

st.set_page_config(page_title="实时诊断", page_icon="📡", layout="wide")

# ═══════════════════════════════════════════════
# 工业暗色 CSS
# ═══════════════════════════════════════════════
st.markdown("""
<style>
.stApp { background: #0b1120; }
[data-testid="stSidebar"] { background: #0f172a; border-right: 1px solid rgba(56,189,248,0.08); }
[data-testid="stSidebar"] label, [data-testid="stSidebar"] .stMarkdown { color: #94a3b8 !important; }
[data-testid="stSidebar"] h3 { color: #38bdf8 !important; font-size: 0.95rem; letter-spacing: 1px; }
[data-testid="stSidebar"] hr { border-color: rgba(56,189,248,0.08); }

/* 状态灯 */
.status-running {
    display: inline-block; width: 10px; height: 10px; border-radius: 50%;
    background: #10b981; box-shadow: 0 0 10px #10b981; margin-right: 8px;
    animation: pulse-green 2s infinite;
}
.status-stopped {
    display: inline-block; width: 10px; height: 10px; border-radius: 50%;
    background: #475569; margin-right: 8px;
}
@keyframes pulse-green {
    0%,100% { box-shadow: 0 0 6px #10b981; }
    50% { box-shadow: 0 0 16px #10b981, 0 0 24px rgba(16,185,129,0.4); }
}

/* 报警条 */
.alarm-bar {
    background: linear-gradient(90deg, rgba(239,68,68,0.2), rgba(239,68,68,0.05));
    border: 1px solid rgba(239,68,68,0.4); border-left: 4px solid #ef4444;
    border-radius: 8px; padding: 0.9rem 1.4rem; margin: 0.4rem 0;
    animation: alarm-pulse 1.5s infinite;
}
@keyframes alarm-pulse {
    0%,100% { border-color: rgba(239,68,68,0.4); }
    50% { border-color: rgba(239,68,68,0.8); box-shadow: 0 0 20px rgba(239,68,68,0.15); }
}

/* 指标卡 */
.metric-card {
    background: #131c33; border: 1px solid rgba(56,189,248,0.08);
    border-radius: 10px; padding: 0.9rem 1.2rem; text-align: center;
}
.metric-value {
    font-size: 1.8rem; font-weight: 800; color: #e2e8f0;
    font-family: 'Consolas','JetBrains Mono',monospace;
}
.metric-label { font-size: 0.72rem; color: #64748b; letter-spacing: 0.5px; }
.metric-accent { color: #38bdf8; }
.metric-green { color: #10b981; }
.metric-red { color: #ef4444; }

/* 按钮 */
.stButton > button {
    background: rgba(56,189,248,0.12); border: 1px solid rgba(56,189,248,0.2);
    border-radius: 8px; color: #38bdf8; font-weight: 600; transition: all 0.2s;
}
.stButton > button:hover:not(:disabled) {
    background: rgba(56,189,248,0.22); border-color: rgba(56,189,248,0.5); color: #e2e8f0;
}
.stButton > button:disabled { opacity: 0.4; }

/* 标签 */
.tag { display: inline-block; padding: 0.2rem 0.7rem; border-radius: 14px;
       font-size: 0.72rem; font-weight: 600; margin-right: 0.3rem; }
.tag-svm { background: rgba(56,189,248,0.12); color: #38bdf8; border: 1px solid rgba(56,189,248,0.2); }
.tag-cnn { background: rgba(129,140,248,0.12); color: #818cf8; border: 1px solid rgba(129,140,248,0.2); }

/* 数据表格 */
[data-testid="stDataFrame"] { background: #131c33 !important; border: 1px solid rgba(56,189,248,0.06); }
[data-testid="stDataFrame"] th { background: #1a2747 !important; color: #94a3b8 !important; }
[data-testid="stDataFrame"] td { color: #cbd5e1 !important; }

/* 页脚 */
.footer { text-align: center; padding: 0.3rem 0; font-size: 0.68rem; color: #475569; }
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
    "grid.alpha": 0.4,
})

LABEL_DICT = {v[0]: v[1] for v in LABEL_MAP.values()}
FAULT_LIST = FAULT_TYPES_CN  # ["正常","内圈故障","外圈故障","滚动体故障"]
COLORS = ["#10b981", "#ef4444", "#f59e0b", "#8b5cf6"]
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "saved")
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "diagnostics.db")
ALARM_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "alarms.db")

# ═══════════════════════════════════════════════
# session_state 初始化
# ═══════════════════════════════════════════════
# 系统固定常量
WINDOW_SIZE = 1024
CHUNK_SIZE = 512
BUFFER_CAPACITY = 4096
ALARM_CONSECUTIVE = 3
ALARM_CONFIDENCE_THRESHOLD = 0.95

D = {
    "running": False, "fault_type": "内圈故障",
    "interval": 0.3, "noise": 0.0, "model_type": "svm",
    "records": [], "latest_window": None, "latest_record": None,
    "total": 0, "alarm_on": False, "alarm_msg": "",
}
for k, v in D.items():
    if k not in st.session_state: st.session_state[k] = v

# ═══════════════════════════════════════════════
# 侧边栏
# ═══════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div style="text-align:center;padding:0.5rem 0 0.8rem 0;"><div style="font-size:2rem;">📡</div><div style="font-size:1rem;font-weight:700;color:#e2e8f0;letter-spacing:1px;">实时诊断控制台</div></div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 🔧 数据源")
    st.session_state.fault_type = st.selectbox("模拟故障类型", FAULT_LIST,
        index=FAULT_LIST.index(st.session_state.fault_type) if st.session_state.fault_type in FAULT_LIST else 0, label_visibility="collapsed")
    st.session_state.interval = st.slider("刷新间隔(秒)", 0.1, 2.0, st.session_state.interval, 0.1)
    st.session_state.noise = st.slider("噪声水平", 0.0, 0.1, st.session_state.noise, 0.001, format="%.3f")

    st.markdown("---")
    st.markdown("### 🤖 推理引擎")
    st.session_state.model_type = st.radio("模型", ["svm","cnn"], index=0 if st.session_state.model_type=="svm" else 1,
        format_func=str.upper, horizontal=True, label_visibility="collapsed")
    st.caption(f"窗口大小={WINDOW_SIZE} | 推送点数={CHUNK_SIZE} | 缓冲区={BUFFER_CAPACITY}")

    st.markdown("---")
    st.markdown("### 🚨 报警规则")
    st.caption(f"连续 {ALARM_CONSECUTIVE} 次同类故障 → WARNING")
    st.caption(f"连续 {ALARM_CONSECUTIVE+2} 次同类故障 → CRITICAL")
    st.caption(f"置信度 > {ALARM_CONFIDENCE_THRESHOLD:.0%} → 即时触发")

    st.markdown("---")
    cb1, cb2 = st.columns(2)
    if cb1.button("▶ 开始推流", use_container_width=True, disabled=st.session_state.running):
        st.session_state.running = True
        st.session_state.records = []; st.session_state.total = 0
        st.session_state.alarm_on = False; st.session_state.alarm_msg = ""
        # 清除旧对象，强制下一轮重建
        for _k in ("sim_obj", "buffer_obj", "engine_obj", "store_obj", "alarm_mgr", "_pf", "_pm", "_gen"):
            st.session_state.pop(_k, None)
        st.rerun()
    if cb2.button("⏹ 停止", use_container_width=True, disabled=not st.session_state.running):
        st.session_state.running = False; st.rerun()

    if st.session_state.running:
        st.markdown('<span class="status-running"></span> <span style="color:#10b981;font-weight:600;">推流运行中</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-stopped"></span> <span style="color:#64748b;">推流已停止</span>', unsafe_allow_html=True)
    st.caption(f"已处理: {st.session_state.total} 窗口")

# ═══════════════════════════════════════════════
# 主区域
# ═══════════════════════════════════════════════
st.markdown('<p style="font-size:1.3rem;font-weight:700;color:#e2e8f0;margin-bottom:0;">📡 实时故障诊断仪表盘</p>', unsafe_allow_html=True)
st.markdown(f'<p style="font-size:0.78rem;color:#64748b;margin-top:0;">模型: <span class="tag tag-{"cnn" if st.session_state.model_type=="cnn" else "svm"}">{st.session_state.model_type.upper()}</span> 故障类型: {st.session_state.fault_type}</p>', unsafe_allow_html=True)

if not st.session_state.running:
    st.markdown('<div style="background:rgba(56,189,248,0.06);border:1px solid rgba(56,189,248,0.12);border-radius:10px;padding:1.2rem 1.5rem;margin:0.8rem 0;"><b style="color:#38bdf8;">👈 操作指引</b>&nbsp;&nbsp;<span style="color:#94a3b8;">选择故障类型 → 选择模型 → 点击「开始推流」→ 实时仪表盘开始刷新</span></div>', unsafe_allow_html=True)
    from core.signal.demo_loader import load_dataset_stats
    _ds = load_dataset_stats()
    _demo_len = _ds.get("demo_length_per_class", 65536)
    ms = st.columns(4)
    ms[0].markdown(f'<div class="metric-card"><div class="metric-value metric-accent">{_demo_len:,}</div><div class="metric-label">信号长度 (点)</div></div>', unsafe_allow_html=True)
    ms[1].markdown(f'<div class="metric-card"><div class="metric-value">4</div><div class="metric-label">故障类别</div></div>', unsafe_allow_html=True)
    ms[2].markdown(f'<div class="metric-card"><div class="metric-value">{WINDOW_SIZE}</div><div class="metric-label">窗口大小</div></div>', unsafe_allow_html=True)
    ms[3].markdown(f'<div class="metric-card"><div class="metric-value metric-green">{ALARM_CONSECUTIVE}</div><div class="metric-label">报警阈值</div></div>', unsafe_allow_html=True)
    st.stop()

# ═══════════════════════════════════════════════
# 推流运行中
# ═══════════════════════════════════════════════

# 创建/恢复对象（从 demo_data/ 加载真实信号循环推流）
if "sim_obj" not in st.session_state or st.session_state.get("_pf") != st.session_state.fault_type:
    _demo_signal = load_demo_signal(st.session_state.fault_type)
    st.session_state.sim_obj = StreamSimulator.from_signal(
        _demo_signal, chunk_size=CHUNK_SIZE,
        interval=st.session_state.interval,
        noise_std=st.session_state.noise, loop=True)
    st.session_state._pf = st.session_state.fault_type
    st.session_state._gen = st.session_state.sim_obj.stream()

# 环形缓冲区（持久化在 session_state 中，跨 rerun 保持状态）
if "buffer_obj" not in st.session_state:
    st.session_state.buffer_obj = RingBuffer(capacity=BUFFER_CAPACITY)

# 推理引擎（模型切换时重建）
if "engine_obj" not in st.session_state or st.session_state.get("_pm") != st.session_state.model_type:
    engine = InferenceEngine()
    engine.load_model(st.session_state.model_type)
    st.session_state.engine_obj = engine
    st.session_state._pm = st.session_state.model_type
    # AlarmManager 内部维护状态，切换模型时自动重置

if "store_obj" not in st.session_state:
    st.session_state.store_obj = DiagnosticsStore(memory_size=5000, db_path=DB_PATH)

# AlarmManager — 持久化到 alarms.db，与报警历史/统计页面共享数据
if "alarm_mgr" not in st.session_state:
    st.session_state.alarm_mgr = AlarmManager(
        device_id="bearing_001",
        db_path=ALARM_DB_PATH,
        cooldown_seconds=0,  # 演示模式不禁用冷却
    )

buf = st.session_state.buffer_obj
engine = st.session_state.engine_obj
store = st.session_state.store_obj
alarm_mgr = st.session_state.alarm_mgr

# 消费一个 chunk
try:
    chunk = next(st.session_state._gen)
except StopIteration:
    st.warning("数据流结束。"); st.session_state.running = False; st.stop()

# 推入缓冲区 → 提取窗口 → 推理
buf.push(chunk)
window = buf.latest(WINDOW_SIZE)
if window is not None:
    result = engine.predict(window)
    st.session_state.total += 1
    st.session_state.latest_window = window

    # 提取特征用于记录
    feats, _ = extract_all_features(window.reshape(1, -1))
    td = feats[0]

    # 确定标签 ID
    pred_name = result["fault_type"]
    pred_label = next(k for k, v in LABEL_DICT.items() if v == pred_name)

    # 报警判定 — 通过 AlarmManager（自动持久化到 alarms.db）
    is_fault = pred_label != 0
    alarm_result = alarm_mgr.evaluate(
        fault_type=pred_name,
        confidence=result["confidence"],
        is_fault=is_fault,
    )
    alarm_triggered = alarm_result is not None

    record = DiagnosisRecord(
        timestamp=time.time(),
        predicted_label=pred_label,
        predicted_name=pred_name,
        confidence=result["confidence"],
        all_probs=result["probabilities"],
        mean_val=float(td[0]),
        rms=float(td[1]),
        kurtosis=float(td[2]),
        skewness=float(td[3]),
        dom_freq=float(td[4]),
        spectral_energy=float(td[5]),
        alarm=alarm_triggered,
    )
    st.session_state.latest_record = asdict(record)
    store.append(record)
    st.session_state.records.append(asdict(record))
    if len(st.session_state.records) > 200: st.session_state.records = st.session_state.records[-200:]
    if alarm_triggered:
        st.session_state.alarm_on = True
        st.session_state.alarm_msg = f"🚨 [{alarm_result.alarm_level.value}] {alarm_result.message}"
    elif pred_label == 0:
        st.session_state.alarm_on = False; st.session_state.alarm_msg = ""

# ═══════════════════════════════════════════════
# 渲染
# ═══════════════════════════════════════════════

# 报警条
if st.session_state.alarm_on:
    st.markdown(f'<div class="alarm-bar"><span style="font-size:1.1rem;font-weight:700;color:#fca5a5;">{st.session_state.alarm_msg}</span></div>', unsafe_allow_html=True)

# 指标卡
mc = st.columns(4)
mc[0].markdown(f'<div class="metric-card"><div class="metric-value metric-accent">{st.session_state.total}</div><div class="metric-label">已处理窗口</div></div>', unsafe_allow_html=True)
rec = st.session_state.latest_record
if rec:
    cname = "metric-green" if rec["predicted_label"] == 0 else "metric-red"
    mc[1].markdown(f'<div class="metric-card"><div class="metric-value {cname}">{rec["predicted_name"]}</div><div class="metric-label">最新诊断</div></div>', unsafe_allow_html=True)
    mc[2].markdown(f'<div class="metric-card"><div class="metric-value">{rec["confidence"]:.1%}</div><div class="metric-label">置信度</div></div>', unsafe_allow_html=True)
    mc[3].markdown(f'<div class="metric-card"><div class="metric-value">{rec["rms"]:.4f}</div><div class="metric-label">RMS 值</div></div>', unsafe_allow_html=True)

# 波形 + 频谱
col1, col2 = st.columns(2)
with col1:
    st.markdown('<p style="font-weight:700;color:#cbd5e1;margin-bottom:0.3rem;">📈 实时波形 (缓冲区最新数据)</p>', unsafe_allow_html=True)
    buf_data = buf.get_buffer()
    if len(buf_data) > 0:
        fig, ax = plt.subplots(figsize=(7,2.8), dpi=100)
        dn = min(2000, len(buf_data))
        ax.plot(np.arange(dn), buf_data[-dn:], color="#38bdf8", linewidth=0.7)
        ax.fill_between(np.arange(dn), buf_data[-dn:], 0, color="#38bdf8", alpha=0.06)
        ax.set_xlabel("采样点 (最近)", fontsize=7); ax.tick_params(labelsize=6)
        ax.set_title("实时振动波形", fontsize=9, fontweight=600, color="#cbd5e1")
        fig.tight_layout(pad=0.8); st.pyplot(fig); plt.close(fig)

with col2:
    st.markdown('<p style="font-weight:700;color:#cbd5e1;margin-bottom:0.3rem;">📊 实时频谱 (最新窗口 FFT)</p>', unsafe_allow_html=True)
    if st.session_state.latest_window is not None:
        freq, amp = compute_fft_spectrum(st.session_state.latest_window)
        fig, ax = plt.subplots(figsize=(7,2.8), dpi=100)
        ax.plot(freq, amp, color="#f59e0b", linewidth=0.7)
        ax.fill_between(freq, amp, 0, color="#f59e0b", alpha=0.04)
        ax.set_xlabel("频率 (Hz)", fontsize=7); ax.set_xlim(0,2000); ax.tick_params(labelsize=6)
        ax.set_title("FFT 实时频谱", fontsize=9, fontweight=600, color="#cbd5e1")
        fig.tight_layout(pad=0.8); st.pyplot(fig); plt.close(fig)

# 诊断结论 + 置信度趋势
col3, col4 = st.columns(2)
with col3:
    st.markdown('<p style="font-weight:700;color:#cbd5e1;margin-bottom:0.3rem;">🔍 最新诊断结论</p>', unsafe_allow_html=True)
    if rec:
        is_ok = rec["predicted_label"] == 0
        bg = "rgba(16,185,129,0.08)" if is_ok else "rgba(239,68,68,0.08)"
        bd = "#10b981" if is_ok else "#ef4444"
        icon = "✅" if is_ok else "⚠️"
        title = "设备正常运转" if is_ok else f"检测到故障: {rec['predicted_name']}"
        st.markdown(f'<div style="background:{bg};border-left:4px solid {bd};border-radius:8px;padding:1rem 1.2rem;margin:0.3rem 0;"><div style="font-size:1.1rem;font-weight:700;color:#e2e8f0;">{icon} {title}</div><div style="margin-top:0.4rem;font-size:0.82rem;color:#94a3b8;">置信度: <b style="color:#e2e8f0;">{rec["confidence"]:.2%}</b> &nbsp;|&nbsp; RMS: {rec["rms"]:.4f} &nbsp;|&nbsp; 峭度: {rec["kurtosis"]:.2f}</div></div>', unsafe_allow_html=True)

    st.markdown('<p style="font-weight:700;color:#cbd5e1;margin:1rem 0 0.3rem 0;">📋 最近诊断记录</p>', unsafe_allow_html=True)
    if st.session_state.records:
        df = pd.DataFrame(st.session_state.records[-8:])[["predicted_name","confidence","rms","kurtosis","alarm"]]
        df.columns = ["诊断结果","置信度","RMS","峭度","报警"]
        df["置信度"] = df["置信度"].apply(lambda x: f"{x:.1%}")
        df["RMS"] = df["RMS"].apply(lambda x: f"{x:.4f}")
        df["峭度"] = df["峭度"].apply(lambda x: f"{x:.2f}")
        df["报警"] = df["报警"].apply(lambda x: "🚨" if x else "")
        st.dataframe(df[::-1], use_container_width=True, height=240, hide_index=True)

with col4:
    st.markdown('<p style="font-weight:700;color:#cbd5e1;margin-bottom:0.3rem;">📉 置信度变化趋势</p>', unsafe_allow_html=True)
    if len(st.session_state.records) >= 2:
        rl = st.session_state.records[-120:]
        t0 = rl[0]["timestamp"]; rt = [r["timestamp"]-t0 for r in rl]
        fig, ax = plt.subplots(figsize=(7,3), dpi=100)
        for i, (label, c) in enumerate(LABEL_DICT.items()):
            ax.plot(rt, [r["all_probs"][i] for r in rl], color=COLORS[i], label=label, linewidth=1.5, alpha=0.9)
        ax.set_xlabel("运行时间 (秒)", fontsize=7); ax.set_ylabel("置信度", fontsize=7)
        ax.set_ylim(0,1.02); ax.legend(fontsize=6, loc="upper right", ncol=2); ax.tick_params(labelsize=6)
        ax.set_title("各类别置信度实时变化", fontsize=9, fontweight=600, color="#cbd5e1")
        fig.tight_layout(pad=1); st.pyplot(fig); plt.close(fig)

    st.markdown('<p style="font-weight:700;color:#cbd5e1;margin-top:0.8rem;margin-bottom:0.3rem;">📊 特征实时监测</p>', unsafe_allow_html=True)
    if len(st.session_state.records) >= 2:
        rl = st.session_state.records[-120:]
        rt = [r["timestamp"]-rl[0]["timestamp"] for r in rl]
        fig, (a1,a2) = plt.subplots(2,1,figsize=(7,3.2),dpi=100)
        a1.plot(rt,[r["rms"] for r in rl],color="#38bdf8",linewidth=1.2)
        a1.set_ylabel("RMS",fontsize=7); a1.set_title("RMS 趋势",fontsize=8,fontweight=600,color="#cbd5e1"); a1.tick_params(labelsize=6)
        a2.plot(rt,[r["kurtosis"] for r in rl],color="#8b5cf6",linewidth=1.2)
        a2.set_xlabel("运行时间 (秒)",fontsize=7); a2.set_ylabel("峭度",fontsize=7); a2.set_title("峭度趋势",fontsize=8,fontweight=600,color="#cbd5e1"); a2.tick_params(labelsize=6)
        fig.tight_layout(pad=1); st.pyplot(fig); plt.close(fig)

st.markdown('<p class="footer">数据源: {} (demo_data/) &nbsp;|&nbsp; 模型: {} &nbsp;|&nbsp; 已处理 {} 窗口</p>'.format(st.session_state.fault_type, st.session_state.model_type.upper(), st.session_state.total), unsafe_allow_html=True)
st.caption("💡 在线演示版 | 诊断记录可能在应用重启后重置")

time.sleep(st.session_state.interval)
st.rerun()
