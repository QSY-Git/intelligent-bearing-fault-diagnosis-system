# ===================== 工业轴承故障诊断全流程代码 =====================
# 固定所有随机种子，保证结果可复现
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from scipy.io import loadmat
from scipy.fft import fft
from scipy.stats import kurtosis
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# -------------------------- 1. 全局配置 --------------------------
# 数据根路径
root_path = "D:/desktop/CWRU Bearing Dataset/原始数据/"

# 4个文件信息
file_config = [
    {"folder": "正常", "file": "97.mat.txt", "fault_name": "正常", "label": 0},
    {"folder": "内圈故障", "file": "105.mat.txt", "fault_name": "内圈故障", "label": 1},
    {"folder": "外圈故障", "file": "130.mat.txt", "fault_name": "外圈故障", "label": 2},
    {"folder": "滚动体故障", "file": "118.mat.txt", "fault_name": "滚动体故障", "label": 3},
]

# 切片参数
window_len = 1024  # 单样本长度
step_len = 512     # 50%重叠率
sample_rate = 12000  # 采样频率12kHz
random_seed = 42   # 固定随机种子，保证结果可复现

# 中文显示配置
plt.rcParams["font.family"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
np.random.seed(random_seed)

# -------------------------- 2. 特征提取函数--------------------------
def get_time_features(signal):
    """【面试必背】时域特征提取，工业故障诊断标准8个核心特征"""
    mean = np.mean(signal) # 均值：振动中心趋势
    std = np.std(signal) # 标准差：振动波动程度
    rms = np.sqrt(np.mean(signal ** 2)) # 均方根RMS：振动能量，设备磨损核心评估指标
    peak = np.max(np.abs(signal)) # 峰值：最大冲击幅值
    peak2peak = np.max(signal) - np.min(signal) # 峰峰值：振动全范围
    kurt = kurtosis(signal) # 峭度：早期故障冲击敏感核心指标
    skew = np.mean(((signal - mean)/std)**3) if std != 0 else 0 # 偏度：信号不对称性
    crest_factor = peak / rms if rms != 0 else 0 # 波形因子：无量纲，抗工况干扰
    return [mean, std, rms, peak, peak2peak, kurt, skew, crest_factor]

def get_freq_features(signal, fs=sample_rate):
    """【面试加分项】频域特征提取，FFT频谱分析，精准定位故障特征频率"""
    n = len(signal)
    yf = fft(signal) # 把时域信号转换为频域信号
    xf = np.linspace(0.0, fs/2.0, n//2) # 线性等分，生成和频域结果对应的频率轴
    yf_abs = 2.0 / n * np.abs(yf[:n//2])  # 单边频谱
    
    freq_peak = np.max(yf_abs) # 频谱峰值：故障特征频率幅值
    freq_mean = np.mean(yf_abs) # 频谱均值：频域整体能量
    freq_rms = np.sqrt(np.mean(yf_abs**2))  # 频谱均方根
    fc = np.sum(xf * yf_abs) / np.sum(yf_abs) if np.sum(yf_abs) != 0 else 0  # 重心频率
    fv = np.sum(((xf - fc)**2) * yf_abs) / np.sum(yf_abs) if np.sum(yf_abs) != 0 else 0  # 频率方差
    return [freq_peak, freq_mean, freq_rms, fc, fv]

# 特征名称
feature_names = [
    "均值", "标准差", "均方根RMS", "峰值", "峰峰值", "峭度", "偏度", "波形因子",
    "频谱峰值", "频谱均值", "频谱均方根", "重心频率", "频率方差"
]
target_names = [item["fault_name"] for item in file_config]

# -------------------------- 3. 批量读取数据+特征提取--------------------------
all_features = []
all_labels = []

print("="*60)
print("开始处理数据...")
print("="*60)

for config in file_config:
    file_path = os.path.join(root_path, config["folder"], config["file"])
    label = config["label"]
    fault_name = config["fault_name"]
    
    try:
        signal = None
        #用mat格式读取
        mat_data = loadmat(file_path)
        # 自动匹配DE_time振动信号，不用手动改key
        de_key = [k for k in mat_data.keys() if "DE_time" in k][0]
        signal = mat_data[de_key].reshape(-1) # 把振动数据从二维数组变成一维的行向量 
        signal = signal[~np.isnan(signal)]  # 过滤空值
        
        # 数据有效性校验
        if len(signal) < window_len:
            print(f"{fault_name} 数据长度不足，跳过")
            continue
        
        #滑窗切片，扩充样本量，避免过拟合，工业标准做法
        total_len = len(signal)
        sample_count = int((total_len - window_len) / step_len) + 1
        
        # 逐样本提取特征
        for i in range(sample_count):
            start = i * step_len
            end = start + window_len
            seg = signal[start:end]
            # 合并时域+频域特征
            time_feat = get_time_features(seg)
            freq_feat = get_freq_features(seg)
            all_features.append(time_feat + freq_feat)
            all_labels.append(label)
        
        print(f"{fault_name} 处理完成 | 生成样本数：{sample_count}")
    
    except Exception as e:
        print(f"{fault_name} 处理失败 | 路径：{file_path} | 错误：{e}")

# 安全校验，无样本直接退出
if not all_features:
    print("\n未生成任何样本，请检查根路径、文件夹名、文件名是否正确")
    exit()

# -------------------------- 4. 构建特征数据集 --------------------------
df_features = pd.DataFrame(all_features, columns=feature_names)
df_features["故障标签"] = all_labels

# 输出数据集信息
print("\n" + "="*60)
print("数据集信息")
print("="*60)
print(f"总样本数：{len(df_features)} | 特征维度：{len(feature_names)}")
print("\n样本类别分布：")
print(df_features["故障标签"].value_counts().sort_index().rename(dict(zip([0,1,2,3], target_names))))
print("\n特征表前5行：")
print(df_features.head())

# 自动保存特征表到本地
df_features.to_excel(os.path.join(root_path, "轴承故障特征表.xlsx"), index=False)
print(f"\n特征表已保存至：{os.path.join(root_path, '轴承故障特征表.xlsx')}")

# -------------------------- 5. 数据集划分+预处理--------------------------
# 特征与标签分离
X = df_features.drop("故障标签", axis=1)
y = df_features["故障标签"]

#分层抽样，保证训练/测试集标签分布一致，避免样本不平衡
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=random_seed, stratify=y
)

# 特征标准化，适配SVM模型
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n训练集样本数：{len(X_train)} | 测试集样本数：{len(X_test)}")

# -------------------------- 6. 多模型训练+对比 --------------------------
model_dict = {
    "随机森林": RandomForestClassifier(n_estimators=100, random_state=random_seed),
    "XGBoost": XGBClassifier(n_estimators=100, random_state=random_seed, use_label_encoder=False, eval_metric='mlogloss'),
    "SVM": SVC(kernel='rbf', random_state=random_seed)
}

# 模型训练与评估
model_result = []
best_model = None
best_acc = 0
best_model_name = ""

print("\n" + "="*60)
print("模型训练与对比结果")
print("="*60)

for name, model in model_dict.items():
    if name == "SVM":
        model.fit(X_train_scaled, y_train)
        train_acc = model.score(X_train_scaled, y_train)
        test_acc = model.score(X_test_scaled, y_test)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)
        y_pred = model.predict(X_test)
    
    # 计算核心评估指标
    report = classification_report(y_test, y_pred, output_dict=True)
    macro_recall = report["macro avg"]["recall"]
    macro_f1 = report["macro avg"]["f1-score"]
    
    # 保存结果
    model_result.append({
        "模型名称": name,
        "训练集准确率": round(train_acc, 4),
        "测试集准确率": round(test_acc, 4),
        "平均召回率": round(macro_recall, 4),
        "平均F1值": round(macro_f1, 4)
    })
    
    # 记录最优模型
    if test_acc > best_acc:
        best_acc = test_acc
        best_model = model
        best_model_name = name
        best_y_pred = y_pred

# 输出模型对比表
df_model_result = pd.DataFrame(model_result)
print(df_model_result)
print(f"\n最优模型：{best_model_name} | 测试集准确率：{best_acc:.4f}")

# 自动保存最优模型
joblib.dump(best_model, os.path.join(root_path, "最优故障诊断模型.pkl"))
print(f"最优模型已保存至：{os.path.join(root_path, '最优故障诊断模型.pkl')}")

# -------------------------- 7. 模型评估报告+可视化 --------------------------
print("\n" + "="*60)
print("最优模型分类详细报告")
print("="*60)
print(classification_report(y_test, best_y_pred, target_names=target_names))

# 1. 混淆矩阵热力图
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, best_y_pred)
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues", linewidths=0.5,
    xticklabels=target_names, yticklabels=target_names
)
plt.xlabel("预测故障类型", fontsize=12)
plt.ylabel("真实故障类型", fontsize=12)
plt.title("轴承故障诊断混淆矩阵", fontsize=14, fontweight="bold")
plt.tight_layout()
# 自动保存图片
plt.savefig(os.path.join(root_path, "混淆矩阵.png"), dpi=300, bbox_inches="tight")
plt.show()

# 2. 特征重要性排序
if best_model_name in ["随机森林", "XGBoost"]:
    feature_importance = pd.DataFrame({
        "特征名称": feature_names,
        "重要性得分": best_model.feature_importances_
    }).sort_values("重要性得分", ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x="重要性得分", y="特征名称", data=feature_importance, palette="viridis")
    plt.xlabel("重要性得分", fontsize=12)
    plt.ylabel("特征名称", fontsize=12)
    plt.title("故障诊断特征重要性排序", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(root_path, "特征重要性.png"), dpi=300, bbox_inches="tight")
    plt.show()
    
    print("\n" + "="*60)
    print("特征重要性排序")
    print("="*60)
    print(feature_importance)

print("\n" + "="*60)
print("全流程运行完成！所有结果已保存")
print("="*60)