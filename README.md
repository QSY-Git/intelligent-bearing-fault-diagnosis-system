# Industrial AI Predictive Maintenance Agent

工业轴承故障预测性维护智能体系统。

基于 CWRU 数据集，从**数据接入 → 实时推理 → 故障诊断 → 报警管理 → Web 仪表盘**，
形成完整的工业 AI 预测性维护闭环。架构从单文件演进到多 Agent 系统，约 5,000 行代码。

---

## 项目背景

对工业轴承振动信号进行分析，实现四种工作状态的自动分类与实时监测：

| 标签 | 故障类型 | CWRU 文件 |
|------|---------|-----------|
| 0 | 正常 | 97, 98 |
| 1 | 内圈故障 | 105, 106 |
| 2 | 外圈故障 | 130, 131 |
| 3 | 滚动体故障 | 118, 119 |

---

## 项目结构

```
Industrial-Predictive-Maintenance-Agent/
│
├── agents/                    # V6 Agent 架构
│   ├── base.py                #   Agent 抽象基类 (生命周期)
│   ├── diagnosis_agent.py     #   诊断 Agent (完整闭环)
│   ├── monitoring_agent.py    #   监控 Agent (报告+健康评分)
│   └── scheduler.py           #   多 Agent 调度器
│
├── alerts/                    # V5 报警系统
│   ├── models.py              #   AlarmRecord / AlarmLevel
│   ├── rules.py               #   规则引擎 (连续故障 + 高置信度)
│   └── manager.py             #   报警管理器 (SQLite 持久化)
│
├── core/                      # 核心引擎
│   ├── engine/
│   │   ├── ring_buffer.py     #   环形缓冲区
│   │   └── inference.py       #   InferenceEngine (SVM/CNN)
│   ├── signal/
│   │   ├── features.py        #   时域/频域特征 + PCA
│   │   ├── loader.py          #   MAT 文件加载 + 数据集构建
│   │   └── transform.py       #   滑动窗口变换
│   ├── pipeline/
│   │   ├── source.py          #   DataSource 抽象基类
│   │   └── sources/
│   │       └── mqtt_client.py #   V4 MQTT 数据源
│   └── persistence/
│       ├── models.py          #   DiagnosisRecord
│       └── store.py           #   deque + SQLite 存储
│
├── backend/                   # V3 FastAPI 服务
│   └── api.py                 #   POST /predict + /health + Swagger
│
├── models/                    # 模型定义与训练
│   ├── architectures/
│   │   └── cnn1d.py           #   1D-CNN 网络结构 (280K 参数)
│   ├── svm_model.py           #   SVM 训练入口
│   ├── cnn_model.py           #   CNN 训练入口
│   └── saved/                 #   已训练模型文件
│
├── pages/                     # Streamlit Dashboard (4页)
│   ├── 1_📈_实时诊断.py        #   实时推流仪表盘
│   ├── 2_📊_历史趋势.py        #   诊断历史 + PCA 轨迹
│   ├── 3_🚨_报警历史.py        #   报警记录查询
│   └── 4_📊_报警统计.py        #   报警统计可视化
│
├── tools/                     # 测试与工具
│   ├── mqtt_simulator.py      #   MQTT 设备模拟器
│   ├── integration_test.py    #   V4 数据管道测试
│   ├── test_v5_alarms.py      #   V5 报警测试 (40/40)
│   └── test_v6_agents.py      #   V6 Agent 测试 (19/19)
│
├── app.py                     # Dashboard 首页
├── data/                      # CWRU 数据集 (8 文件, ~146万采样点)
├── config/default.yaml        # MQTT / Signal / Alarm 配置
└── requirements.txt
```

---

## 技术栈

| 层级 | 技术选型 |
|------|---------|
| 数据接入 | MQTT (paho-mqtt), DataSource 抽象层, StreamSimulator |
| 信号处理 | NumPy, SciPy FFT, 滑动窗口 (Window=1024, Step=512) |
| 特征工程 | 时域 (均值/RMS/峭度/偏度), 频域 (主频/频谱能量), PCA |
| 传统 ML | PCA + SVM (RBF核, C=10, gamma=scale), scikit-learn |
| 深度学习 | 1D-CNN (Conv1D×2 + MaxPool, 280K参数), PyTorch |
| 环形缓冲 | 固定大小 NumPy 数组, O(1) push/latest, 环绕写入 |
| 报警系统 | 规则引擎 (连续故障 / 高置信度), WARNING/CRITICAL 分级, SQLite |
| API 服务 | FastAPI + Pydantic v2 + Swagger 自动文档 |
| Web 仪表盘 | Streamlit (4页), 工业 SCADA 暗色主题 |
| Agent 架构 | 多线程 Agent + Scheduler, 依赖注入, 独立生命周期 |
| 配置管理 | YAML (config/default.yaml) |
| 数据存储 | SQLite (diagnostics.db + alarms.db), deque 内存缓存 |
| 测试 | 集成测试 78 项 (V4+V5+V6) |

---

## 架构演进

```
V1: 静态文件诊断
    app.py 单文件，文件上传 → 滑动窗口 → SVM/CNN → 图表展示

V2: 实时流式诊断
    新增 StreamSimulator + StreamProcessor
    环形缓冲区 + 流式滑动窗口 + 实时仪表盘

V3: API 服务层
    新增 backend/api.py (FastAPI)
    POST /predict, GET /health, Swagger 文档

V4: 物联网接入
    新增 DataSource ABC + MQTTDataSource + MQTT 模拟器
    config/default.yaml 配置化

V5: 工业报警系统
    新增 alerts/ 模块 (规则引擎 + 分级报警 + SQLite)
    Dashboard 新增报警历史/统计页面

V6: Agent 架构
    新增 agents/ 模块 (BaseAgent + DiagnosisAgent + MonitoringAgent + Scheduler)
    依赖注入, 多线程, 完整闭环
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 训练模型

```bash
python models/svm_model.py      # SVM: 秒级训练, 100% 准确率
python models/cnn_model.py      # CNN: 分钟级训练, 100% 准确率
```

### 3. 启动 Web 仪表盘

```bash
streamlit run app.py
# → http://localhost:8501
```

### 4. 启动 API 服务

```bash
python backend/api.py
# → http://localhost:8000/docs (Swagger)
```

### 5. MQTT 端到端测试（需 Broker）

```bash
# 终端 1: 启动 MQTT Broker (mosquitto -v)
# 终端 2: 启动模拟设备
python tools/mqtt_simulator.py --fault 内圈故障
# 终端 3: 运行诊断客户端
python -c "
from core.pipeline.sources.mqtt_client import MQTTDataSource
from core.engine.inference import InferenceEngine
source = MQTTDataSource(); source.connect()
engine = InferenceEngine(); engine.load_model('svm')
while True:
    signal = source.read()
    result = engine.predict(signal)
    print(f'{result[\"fault_type\"]} {result[\"confidence\"]:.2%}')
"
```

### 6. 运行全部测试

```bash
python tools/test_v6_agents.py    # V6 Agent 集成测试
python tools/test_v5_alarms.py    # V5 报警系统测试
python tools/integration_test.py  # V4 数据管道测试
```

---

## API 文档

| 方法 | 端点 | 说明 |
|------|------|------|
| `GET` | `/health` | 健康检查, 返回模型状态与可用标签 |
| `POST` | `/predict?model=svm` | SVM 模型诊断 |
| `POST` | `/predict?model=cnn` | CNN 模型诊断 |
| `GET` | `/docs` | Swagger 交互式文档 |
| `GET` | `/redoc` | ReDoc 文档 |

请求示例:

```json
POST /predict?model=svm
{
  "signal": [0.012, -0.003, 0.018, ...]
}

→ {
  "fault_type": "内圈故障",
  "confidence": 0.9876,
  "probabilities": [0.001, 0.9876, 0.008, 0.0034],
  "model_used": "svm"
}
```

---

## 报警规则

| 规则 | 条件 | 等级 |
|------|------|------|
| 连续故障 ≥3 次 | consecutive_count ≥ 3 | WARNING |
| 连续故障 ≥5 次 | consecutive_count ≥ 5 | CRITICAL |
| 高置信度 | confidence > 0.95 | WARNING |
| 极高置信度 | confidence > 0.99 | CRITICAL |

---

## 命令行工具

```bash
# MQTT 模拟器
python tools/mqtt_simulator.py --fault 内圈故障 --interval 0.1

# 集成测试
python tools/test_v6_agents.py     # Agent 架构
python tools/test_v5_alarms.py     # 报警系统
python tools/integration_test.py   # 数据管道
```

---

## 依赖项

```
numpy, pandas, scipy         — 数据处理与科学计算
scikit-learn                 — 标准化, PCA, SVM
torch                        — 1D-CNN 深度学习
matplotlib, seaborn          — 数据可视化
streamlit                    — Web 仪表盘
fastapi, uvicorn, pydantic   — REST API 服务
paho-mqtt, pyyaml            — MQTT 物联网 + 配置
joblib                       — 模型序列化
