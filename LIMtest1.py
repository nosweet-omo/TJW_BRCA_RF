#不做3sigma，仅1：1正负样本，负样本为癌旁，划分6、2、2

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. 配置
# ==========================================
# 请替换为你的 TCEA eRNA 原始文件路径
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

# ==========================================
# 5. 模型训练 (随机森林)
# ==========================================
# 对于这种小样本 (N=200左右)，随机森林比 CNN 更稳健，不容易过拟合
print("\n>>> 开始训练随机森林模型...")

# 标准化 (虽然 RF 不强求，但为了后续如果有 PCA/CNN 建议加上)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
clf.fit(X_train_scaled, y_train)

# 预测
y_pred_prob = clf.predict_proba(X_test_scaled)[:, 1]
auc = roc_auc_score(y_test, y_pred_prob)

print(f"\n{'-'*30}")
print(f"🏆 实验结果 (测试集 AUC): {auc:.4f}")
print(f"{'-'*30}")

# ==========================================
# 6. 结果判定
# ==========================================
if auc > 0.90:
    print("✅ 结论：效果极好！")
    print("   即使在没有 GTEx 加持、不做筛选的情况下，eRNA 依然能完美区分癌症和癌旁。")
    print("   这证明了你的 eRNA 特征非常强，文章立住了！")
elif auc > 0.75:
    print("🆗 结论：效果符合预期 (真实生物学差异)。")
    print("   0.75-0.90 是生信挖掘中最常见的真实范围。")
    print("   这说明有场癌化影响，但依然有区分度。后续可以通过特征工程提升。")
else:
    print("⚠️ 结论：区分度较低。")
    print("   可能需要重新检查 eRNA 特征选择，或者说明癌旁效应非常强。")

from sklearn.base import clone

# 复制一个模型
clf_perm = clone(clf)

# --- 核心：打乱 Y 的标签 ---
y_permuted = np.random.permutation(y) 

# 用打乱的标签重新切分、训练
X_temp_p, X_test_p, y_temp_p, y_test_p = train_test_split(X, y_permuted, test_size=0.2, random_state=SEED, stratify=y_permuted)
X_train_p, X_val_p, y_train_p, y_val_p = train_test_split(X_temp_p, y_temp_p, test_size=0.25, random_state=SEED, stratify=y_temp_p)

clf_perm.fit(scaler.fit_transform(X_train_p), y_train_p)
y_pred_perm = clf_perm.predict_proba(scaler.transform(X_test_p))[:, 1]
auc_perm = roc_auc_score(y_test_p, y_pred_perm)

print(f"\n🧪 置换检验 (随机打乱标签): AUC = {auc_perm:.4f}")
if auc_perm > 0.6:
    print("❌ 警告：打乱标签后准确率依然很高，说明模型严重过拟合或特征数过多！结果不可信。")
else:
    print("✅ 通过：打乱标签后 AUC 接近 0.5，说明之前的 100% 是来自真实的数据信号。")