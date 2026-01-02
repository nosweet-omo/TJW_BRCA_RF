# 癌旁 + LASSO

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel, VarianceThreshold
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

FILE_PATH = 'TCGA_RPKM_eRNA_300k_peaks_in_Super_enhancer_BRCA.txt' 

# 随机种子 (保证每次抽样结果一致)
SEED = 42

print(f">>> 正在读取文件: {FILE_PATH} ...")
try:
    # 假设文件是制表符分隔，行是eRNA，列是样本
    df = pd.read_csv(FILE_PATH, sep='\t', index_col=0)
    print(f"   原始数据形状: {df.shape}")
except Exception as e:
    print(f"❌ 读取失败: {e}")
    exit()

# ==========================================
# 2. 样本识别与提取 (只用 TCGA 内部)
# ==========================================
cols = df.columns.tolist()

# 尝试识别 01 (Tumor) 和 11 (Normal)
# 逻辑：如果列名符合 TCGA 标准 (第14-15位)，或者是 _tumor/_normal 后缀
tumor_cols = []
normal_cols = []

if any('_tumor' in c for c in cols):
    tumor_cols = [c for c in cols if '_tumor' in c]
    normal_cols = [c for c in cols if '_normal' in c]
else:
    # 假设是标准 Barcode，切片检查
    # 如果列名不够长，可能会报错，这里加个判断
    tumor_cols = [c for c in cols if len(c) > 15 and c[13:15] == '01']
    normal_cols = [c for c in cols if len(c) > 15 and c[13:15] == '11']

print(f"   识别到 Tumor (癌症): {len(tumor_cols)}")
print(f"   识别到 Normal (癌旁): {len(normal_cols)}")

if len(normal_cols) < 10:
    print("❌ 正常样本太少，无法进行实验！请检查文件列名格式。")
    exit()

# ==========================================
# 3. 下采样 (Downsampling) - 你的核心实验逻辑
# ==========================================
print(f"\n>>> 正在进行【等量抽取】实验...")
print(f"   目标：随机抽取 {len(normal_cols)} 个癌症样本，与癌旁 1:1 配对")

# 随机抽取与 normal 数量一致的 tumor
np.random.seed(SEED)
selected_tumor_cols = np.random.choice(tumor_cols, size=len(normal_cols), replace=False)

# 构建最终数据集
df_tumor = df[selected_tumor_cols].T
df_normal = df[normal_cols].T

df_tumor['Label'] = 1  # 癌症为 1
df_normal['Label'] = 0 # 癌旁为 0

full_data = pd.concat([df_tumor, df_normal])

# 检查 eRNA 特征是否需要 Log 转换 (如果最大值 > 100 就转)
X = full_data.drop(columns=['Label'])
y = full_data['Label']

if X.max().max() > 100:
    print("   数值较大，应用 Log2(x+1) 转换...")
    X = np.log2(X + 1)

# ==========================================
# 4. 划分数据集 (6:2:2)
# ==========================================
print("\n>>> 正在划分数据集 (60% 训练, 20% 验证, 20% 测试)...")

# 第一刀：切出 20% 测试集 (Test)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

# 第二刀：剩下的 80% 里再切出 25% 作为验证集 (0.8 * 0.25 = 0.2)
# 这样最终比例就是 6:2:2
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=SEED, stratify=y_temp
)

print(f"   训练集: {X_train.shape[0]} (Tumor: {sum(y_train==1)}, Normal: {sum(y_train==0)})")
print(f"   验证集: {X_val.shape[0]}")
print(f"   测试集: {X_test.shape[0]}")

print(f"\n{'='*40}")
print(f"🚀 开始特征筛选 (Feature Selection)")
print(f"{'='*40}")

# ==========================================
# 1. 预处理：标准化 (对 LASSO 至关重要)
# ==========================================
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

print(f"原始特征数: {X_train.shape[1]}")

# ==========================================
# 2. 初筛：方差过滤 (Variance Threshold)
# ==========================================
# 剔除那些在 99% 的样本中数值都一样的特征 (几乎没有区分度的)
# 阈值设为 0.01 (或者更严格)
selector_var = VarianceThreshold(threshold=0.01)
X_train_var = selector_var.fit_transform(X_train_scaled)
X_test_var = selector_var.transform(X_test_scaled)

# 获取剩余特征的列名
kept_indices = selector_var.get_support(indices=True)
kept_columns = X_train.columns[kept_indices]

# 更新 DataFrame
X_train_filtered = pd.DataFrame(X_train_var, columns=kept_columns, index=X_train.index)
X_test_filtered = pd.DataFrame(X_test_var, columns=kept_columns, index=X_test.index)

print(f"方差过滤后剩余特征数: {X_train_filtered.shape[1]}")

# ==========================================
# 3. 核心筛选：LASSO (Logistic Regression L1)
# ==========================================
print("\n>>> 正在运行 LASSO 进行稀疏特征选择...")
# C 值越小，正则化越强，选出来的特征越少；C 值越大，特征越多。
# 建议尝试 C=0.01, 0.05, 0.1, 0.5 来控制数量
lasso = LogisticRegression(penalty='l1', C=0.1, solver='liblinear', random_state=42)
lasso.fit(X_train_filtered, y_train)

# 获取系数不为 0 的特征
model_coef = lasso.coef_.flatten()
selected_mask = model_coef != 0
selected_features_lasso = X_train_filtered.columns[selected_mask].tolist()
selected_coefs = model_coef[selected_mask]

print(f"✅ LASSO 筛选出 {len(selected_features_lasso)} 个重要特征")

# 如果 LASSO 选太多，我们强制取绝对值系数最大的 Top 10
if len(selected_features_lasso) > 10:
    print("   (特征依然很多，强制选取系数绝对值最大的 Top 10)")
    # 创建 (特征名, 系数绝对值) 的列表并排序
    feature_importance = list(zip(selected_features_lasso, np.abs(selected_coefs)))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    final_features = [x[0] for x in feature_importance[:10]]
else:
    final_features = selected_features_lasso

print(f"\n💎 最终入选的 '黄金 eRNA' ({len(final_features)}个):")
for i, f in enumerate(final_features):
    print(f"   {i+1}. {f}")

# ==========================================
# 4. 终极验证：用这几个特征重跑模型
# ==========================================
print(f"\n>>> 正在使用 Top {len(final_features)} 特征重构随机森林...")

# 只取这些列
X_train_final = X_train_scaled[final_features]
X_test_final = X_test_scaled[final_features]

# 训练小模型
clf_mini = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf_mini.fit(X_train_final, y_train)

# 预测
y_pred_prob = clf_mini.predict_proba(X_test_final)[:, 1]
auc_mini = roc_auc_score(y_test, y_pred_prob)

print(f"\n{'-'*40}")
print(f"🏆 Top {len(final_features)} 特征模型测试集 AUC: {auc_mini:.4f}")
print(f"{'-'*40}")

# 混淆矩阵看一眼
y_pred = clf_mini.predict(X_test_final)
print("混淆矩阵:")
print(confusion_matrix(y_test, y_pred))

# 如果需要保存特征列表用于画图
# pd.Series(final_features).to_csv("Diagnostic_Signature_Genes.csv", index=False)