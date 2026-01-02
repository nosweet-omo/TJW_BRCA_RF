import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import ttest_ind
import random

# --- 0. 设置随机种子 (保证结果可复现) ---
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# --- 绘图风格设置 ---
font_title = {'size': 16, 'weight': 'bold'}
font_others = {'size': 14}
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# 检测设备 (优先使用 GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f">>> Using Device: {device}")

# ==========================================
# 1. 参数配置
# ==========================================
expression_file = 'data/TCGA_RPKM_eRNA_300k_peaks_in_Super_enhancer_BRCA.txt'
phenotype_file = 'data/TCGA.BRCA.sampleMap_BRCA_clinicalMatrix'
normal_samples_csv = 'data/Result.csv'
se_bed_file = 'data/final.SE.bed'

cancer_type = 'BRCA'
target_stage = 'Stage I'
EXPR_THRESHOLD = 0.1 

# CNN 超参数
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 50       # 训练轮数
TOP_N_FEATURES = 20 # CNN通常需要比RF稍微多一点的特征来提取模式，建议设为20或30

print(f">>> Starting CNN model training (6:2:2 Split) for '{cancer_type}' '{target_stage}'...")
print("-" * 70)

# ==========================================
# 2. 数据加载与样本筛选 (保持不变)
# ==========================================
print(">>> Step 1 & 2: Loading clinical, expression data and filtering samples...")
phenotype_df = pd.read_csv(phenotype_file, sep='\t', index_col=0)
expression_df = pd.read_csv(expression_file, sep='\t', index_col=0)
normal_like_info = pd.read_csv(normal_samples_csv)

stage_column = 'AJCC_Stage_nature2012'
stage_one_pheno = phenotype_df[phenotype_df[stage_column].str.startswith(target_stage, na=False)]
stage_one_patient_ids = set(stage_one_pheno.index.str[:12])

cancer_sample_ids = [sid for sid in expression_df.columns if
                     sid.endswith('_tumor') and sid[:12] in stage_one_patient_ids]

brca_normal_ids_from_csv = normal_like_info[normal_like_info['CancerType'] == cancer_type]['SampleID'].tolist()
good_normal_patient_ids = {sid[:12] for sid in brca_normal_ids_from_csv}

brca_normal_ids_in_expr = [sid for sid in expression_df.columns if
                            sid.endswith('_normal') and sid[:12] in good_normal_patient_ids]

cancer_df = expression_df[cancer_sample_ids]
normal_df = expression_df[brca_normal_ids_in_expr]

print(f"Data loaded. Stage I samples: {len(cancer_sample_ids)} | Normal samples: {len(brca_normal_ids_in_expr)}")

# ==========================================
# 3. 数据预处理 (保持不变)
# ==========================================
print(f">>> Step 3.1: Filtering inactive eRNAs (Mean > {EXPR_THRESHOLD})...")
mean_expr_in_cancer = cancer_df.mean(axis=1)
active_ernas = mean_expr_in_cancer[mean_expr_in_cancer > EXPR_THRESHOLD].index
expression_df = expression_df.loc[active_ernas]

print(">>> Step 3.2: Collapsing eRNAs based on Super-Enhancer regions...")
try:
    se_df = pd.read_csv(se_bed_file, sep='\t', header=None, usecols=[0, 1, 2], names=['chrom', 'start', 'end'])
except Exception as e:
    print(f"!!! Error reading SE file: {e}")
    se_df = pd.DataFrame(columns=['chrom', 'start', 'end'])

def parse_feature_coord(feat):
    try:
        chrom, pos = feat.split(':')
        if not chrom.startswith('chr'): chrom = 'chr' + chrom
        return chrom, int(pos)
    except:
        return None, None

coords_list = [parse_feature_coord(f) for f in expression_df.index]
valid_indices = [i for i, x in enumerate(coords_list) if x[0] is not None]

expression_df = expression_df.iloc[valid_indices]
erna_coords = pd.DataFrame([coords_list[i] for i in valid_indices], columns=['chrom', 'pos'])
erna_coords['feature_id'] = expression_df.index

keep_features = []
unique_chroms = erna_coords['chrom'].unique()

for chrom in unique_chroms:
    sub_se = se_df[se_df['chrom'] == chrom]
    sub_erna = erna_coords[erna_coords['chrom'] == chrom]
    if sub_se.empty or sub_erna.empty: continue
    for _, se_row in sub_se.iterrows():
        mask = (sub_erna['pos'] >= se_row['start']) & (sub_erna['pos'] <= se_row['end'])
        if mask.any():
            candidates = sub_erna.loc[mask, 'feature_id'].tolist()
            if len(candidates) == 1:
                keep_features.append(candidates[0])
            else:
                candidate_means = expression_df.loc[candidates].mean(axis=1)
                keep_features.append(candidate_means.idxmax())

keep_features = list(set(keep_features))
print(f"   Features retained (Representing SEs): {len(keep_features)}")

cancer_df = expression_df.loc[keep_features, cancer_sample_ids]
normal_df = expression_df.loc[keep_features, brca_normal_ids_in_expr]

X_cancer = cancer_df.T
X_normal = normal_df.T
y_cancer = pd.Series([1] * len(X_cancer), index=X_cancer.index)
y_normal = pd.Series([0] * len(X_normal), index=X_normal.index)

X_full = pd.concat([X_cancer, X_normal])
y_full = pd.concat([y_cancer, y_normal])
patient_groups = np.array([idx[:12] for idx in X_full.index])

print("-" * 70)

# ==========================================
# 4. 数据划分 (6:2:2) (保持不变)
# ==========================================
print(">>> Step 4: Splitting Data (Train 60% / Val 20% / Test 20%)...")
splitter_1 = GroupShuffleSplit(n_splits=1, train_size=0.6, random_state=42)
train_idx, temp_idx = next(splitter_1.split(X_full, y_full, groups=patient_groups))

X_train, y_train = X_full.iloc[train_idx], y_full.iloc[train_idx]
X_temp, y_temp = X_full.iloc[temp_idx], y_full.iloc[temp_idx]
groups_temp = patient_groups[temp_idx]

splitter_2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
val_idx, test_idx = next(splitter_2.split(X_temp, y_temp, groups=groups_temp))

X_val, y_val = X_temp.iloc[val_idx], y_temp.iloc[val_idx]
X_test, y_test = X_temp.iloc[test_idx], y_temp.iloc[test_idx]

print(f"   [Train]: {len(X_train)} | [Val]: {len(X_val)} | [Test]: {len(X_test)}")

# ==========================================
# 5. 特征选择 & 数据标准化 & 转换 (Modified for CNN)
# ==========================================
print("-" * 70)
print(">>> Step 5: Preparing Data for CNN...")

# --- 5.1 特征选择 ---
def select_top_features(X, y, top_n=10):
    cancer_data = X[y == 1]
    normal_data = X[y == 0]
    p_values = []
    features = X.columns
    for feat in features:
        c_vals = cancer_data[feat].values
        n_vals = normal_data[feat].values
        if np.var(c_vals) == 0 and np.var(n_vals) == 0:
            p_values.append(1.0)
        else:
            _, p = ttest_ind(c_vals, n_vals, equal_var=False, nan_policy='omit')
            p_values.append(p)
    p_values = np.nan_to_num(p_values, nan=1.0)
    top_indices = np.argsort(p_values)[:top_n]
    return features[top_indices].tolist()

selected_features = select_top_features(X_train, y_train, top_n=TOP_N_FEATURES)
print(f"   Top {TOP_N_FEATURES} features selected.")

X_train_sel = X_train[selected_features]
X_val_sel = X_val[selected_features]
X_test_sel = X_test[selected_features]

# --- 5.2 SMOTE (仅对训练集) ---
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_sel, y_train)

# --- 5.3 标准化 (StandardScaler) - CNN 必须步骤 ---
# 注意：必须在 Train 上 fit，在 Val/Test 上 transform
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_res)
X_val_scaled = scaler.transform(X_val_sel)
X_test_scaled = scaler.transform(X_test_sel)

# --- 5.4 转换为 PyTorch Tensor 并 reshape 为 (Batch, Channels, Length) ---
# 1D-CNN 的输入需要是 [Batch, 1, Features]
def to_loader(X, y, batch_size=32, shuffle=False):
    tensor_x = torch.Tensor(X).unsqueeze(1) # 增加一个维度变为 (N, 1, Features)
    tensor_y = torch.Tensor(y.values).float().unsqueeze(1)
    dataset = TensorDataset(tensor_x, tensor_y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

train_loader = to_loader(X_train_scaled, y_train_res, BATCH_SIZE, shuffle=True)
val_loader = to_loader(X_val_scaled, y_val, BATCH_SIZE, shuffle=False)
test_loader = to_loader(X_test_scaled, y_test, BATCH_SIZE, shuffle=False)

# ==========================================
# 6. 定义 CNN 模型与训练
# ==========================================
class GenomicCNN(nn.Module):
    def __init__(self, num_features):
        super(GenomicCNN, self).__init__()
        # 定义 1D 卷积层
        self.features = nn.Sequential(
            # Conv Layer 1
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            
            # Conv Layer 2
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        # 计算 Flatten 后的维度
        # 输入长度 num_features 经过两次 MaxPool(2) -> num_features // 4
        final_dim = num_features // 4
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * final_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.5), # 防止过拟合
            nn.Linear(32, 1),
            nn.Sigmoid() # 输出概率
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# 初始化模型
model = GenomicCNN(num_features=TOP_N_FEATURES).to(device)
criterion = nn.BCELoss() # 二分类交叉熵
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print(f"   Model Structure: \n{model}")
print(">>> Starting Training Loop...")

train_losses = []
val_scores = []

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    # 每个 Epoch 结束后验证一次
    model.eval()
    val_preds = []
    val_targets = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            val_preds.extend(outputs.cpu().numpy())
            val_targets.extend(labels.numpy())
    
    val_auc = roc_auc_score(val_targets, val_preds)
    train_losses.append(running_loss / len(train_loader))
    val_scores.append(val_auc)
    
    if (epoch + 1) % 10 == 0:
        print(f"   Epoch [{epoch+1}/{EPOCHS}] Train Loss: {running_loss:.4f} | Val AUC: {val_auc:.4f}")

print("   Training complete.")

# ==========================================
# 7. 模型评估 (Test Set)
# ==========================================
print("-" * 70)
print(">>> Step 7: Evaluation on TEST SET...")

model.eval()
y_test_pred_proba = []
y_test_true = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        y_test_pred_proba.extend(outputs.cpu().numpy().flatten())
        y_test_true.extend(labels.numpy().flatten())

y_test_pred_proba = np.array(y_test_pred_proba)
y_test_true = np.array(y_test_true)
y_test_pred = (y_test_pred_proba > 0.5).astype(int)

test_auc = roc_auc_score(y_test_true, y_test_pred_proba)
print(f"   [Test Set AUC]  : {test_auc:.4f} (FINAL RESULT)")

print("\n   Test Set Classification Report:")
print(classification_report(y_test_true, y_test_pred, target_names=['Normal', 'Cancer']))

# ==========================================
# 8. 绘图与结果保存
# ==========================================
print("-" * 70)
print(">>> Step 8: Generating Plots...")

# 8.1 混淆矩阵
cm = confusion_matrix(y_test_true, y_test_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Pred Normal', 'Pred Cancer'],
            yticklabels=['Actual Normal', 'Actual Cancer'])
plt.title(f'CNN Confusion Matrix (Test Set)', fontdict=font_title)
plt.savefig(f'{cancer_type}_CNN_Confusion_Matrix.png', dpi=300, bbox_inches='tight')

# 8.2 ROC 曲线
fpr, tpr, _ = roc_curve(y_test_true, y_test_pred_proba)
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color='purple', lw=2.5, label=f'CNN Test AUC = {test_auc:.4f}')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlabel('False Positive Rate', fontdict=font_others)
plt.ylabel('True Positive Rate', fontdict=font_others)
plt.title(f'CNN ROC Curve', fontdict=font_title)
plt.legend(loc="lower right")
plt.savefig(f'{cancer_type}_CNN_ROC_Curve.png', dpi=300, bbox_inches='tight')

# 8.3 训练过程曲线 (Loss vs AUC)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss', color='blue')
plt.title('Training Loss')
plt.xlabel('Epoch')

plt.subplot(1, 2, 2)
plt.plot(val_scores, label='Val AUC', color='green')
plt.title('Validation AUC')
plt.xlabel('Epoch')
plt.savefig(f'{cancer_type}_CNN_Training_History.png', dpi=300, bbox_inches='tight')

# 8.4 保存特征表
final_stats = []
# 恢复到原始数据计算 p-value (Test Set)
for i, feat in enumerate(selected_features):
    c_vals = X_test_sel[y_test == 1][feat].values
    n_vals = X_test_sel[y_test == 0][feat].values
    
    if len(c_vals) > 0 and len(n_vals) > 0:
        _, p_val = ttest_ind(c_vals, n_vals, equal_var=False)
        lfc = np.log2((np.mean(c_vals) + 1e-6) / (np.mean(n_vals) + 1e-6))
    else:
        p_val, lfc = np.nan, np.nan
        
    final_stats.append({
        'Feature': feat,
        'P_Value_Test': p_val,
        'Log2FC_Test': lfc
    })

res_df = pd.DataFrame(final_stats)
res_df.to_csv(f'{cancer_type}_CNN_Selected_Features.csv', index=False)

print(">>> CNN Model Analysis Completed Successfully.")