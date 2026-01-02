import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

# --- 绘图风格设置 ---
font_title = {'size': 16, 'weight': 'bold'}
font_others = {'size': 14}
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# --- 1. 文件路径和参数设置 ---
expression_file = 'data/TCGA_RPKM_eRNA_300k_peaks_in_Super_enhancer_BRCA.txt'
phenotype_file = 'data/TCGA.BRCA.sampleMap_BRCA_clinicalMatrix'
normal_samples_csv = 'data/Result.csv'
se_bed_file = 'data/final.SE.bed'

cancer_type = 'BRCA'
target_stage = 'Stage I'
EXPR_THRESHOLD = 0.1 

print(f">>> Starting model training and analysis for '{cancer_type}' '{target_stage}'...")
print("-" * 70)

# --- 2. 加载并准备数据 ---
print(">>> Step 1 & 2: Loading clinical, expression data and filtering samples...")
phenotype_df = pd.read_csv(phenotype_file, sep='\t', index_col=0)
expression_df = pd.read_csv(expression_file, sep='\t', index_col=0)
normal_like_info = pd.read_csv(normal_samples_csv)

# 样本ID匹配逻辑
all_brca_patients = {idx[:12] for idx in phenotype_df.index}
stage_column = 'AJCC_Stage_nature2012'
stage_one_pheno = phenotype_df[phenotype_df[stage_column].str.startswith(target_stage, na=False)]
brca_normal_ids_from_csv = normal_like_info[normal_like_info['CancerType'] == cancer_type]['SampleID'].tolist()

stage_one_patient_ids = set(stage_one_pheno.index.str[:12])
cancer_sample_ids = [sid for sid in expression_df.columns if
                     sid.endswith('_tumor') and sid[:12] in stage_one_patient_ids]
cancer_df = expression_df[cancer_sample_ids]

good_normal_patient_ids = {sid[:12] for sid in brca_normal_ids_from_csv}
brca_normal_ids_in_expr = [sid for sid in expression_df.columns if
                           sid.endswith('_normal') and sid[:12] in good_normal_patient_ids]
normal_df = expression_df[brca_normal_ids_in_expr]

print(f"Data ready. Stage I samples: {len(cancer_sample_ids)} | Normal samples: {len(brca_normal_ids_in_expr)}")
print("-" * 70)

# --- 2.4. (预处理) 基于表达活跃度过滤 eRNA ---
print(f">>> Step 2.4: Filtering inactive eRNAs (Mean Expression < {EXPR_THRESHOLD})...")
mean_expr_in_cancer = cancer_df.mean(axis=1)
active_ernas = mean_expr_in_cancer[mean_expr_in_cancer > EXPR_THRESHOLD].index
expression_df = expression_df.loc[active_ernas]
print(f"   Retained {len(expression_df)} potentially active eRNAs.")

# --- 2.5. (预处理) 基于合并后的 Super-Enhancer 注释 collapse eRNAs ---
print(">>> Step 2.5: Collapsing eRNAs based on merged Super-Enhancer regions...")
try:
    se_df = pd.read_csv(se_bed_file, sep='\t', header=None, usecols=[0, 1, 2], names=['chrom', 'start', 'end'])
except Exception as e:
    raise FileNotFoundError(f"!!! Error reading '{se_bed_file}': {e}")

# 解析坐标 (增加错误处理)
def parse_feature_coord(feat):
    try:
        chrom, pos = feat.split(':')
        if not chrom.startswith('chr'): chrom = 'chr' + chrom
        return chrom, int(pos)
    except:
        return None, None # 处理格式异常

coords_list = [parse_feature_coord(f) for f in expression_df.index]
valid_indices = [i for i, x in enumerate(coords_list) if x[0] is not None]
expression_df = expression_df.iloc[valid_indices]
erna_coords = pd.DataFrame([coords_list[i] for i in valid_indices], columns=['chrom', 'pos'])
erna_coords['feature_id'] = expression_df.index

# --- 初始化变量 ---
keep_features = []
unique_chroms = erna_coords['chrom'].unique()

print(f"   Mapping {len(expression_df)} eRNAs to SE regions...")

# 循环映射 SE
for chrom in unique_chroms:
    sub_se = se_df[se_df['chrom'] == chrom]
    sub_erna = erna_coords[erna_coords['chrom'] == chrom]
    if sub_se.empty or sub_erna.empty: continue
    
    for _, se_row in sub_se.iterrows():
        mask = (sub_erna['pos'] >= se_row['start']) & (sub_erna['pos'] <= se_row['end'])
        if mask.any():
            candidates = sub_erna.loc[mask, 'feature_id'].tolist()
            # 选取该SE区域内表达量最高的代表
            if len(candidates) == 1:
                keep_features.append(candidates[0])
            else:
                candidate_means = expression_df.loc[candidates].mean(axis=1)
                keep_features.append(candidate_means.idxmax())

keep_features = list(set(keep_features))
print(f"   Features retained (Representing SEs): {len(keep_features)}")

# 更新矩阵，准备进入交叉验证
cancer_df = expression_df.loc[keep_features, cancer_sample_ids]
normal_df = expression_df.loc[keep_features, brca_normal_ids_in_expr]

X_cancer = cancer_df.T
X_normal = normal_df.T

# --- 关键修正：显式指定 index，确保 y 和 X 的索引完全一致 ---
y_cancer = pd.Series([1] * len(X_cancer), index=X_cancer.index)
y_normal = pd.Series([0] * len(X_normal), index=X_normal.index)

X_full = pd.concat([X_cancer, X_normal])
y_full = pd.concat([y_cancer, y_normal])
patient_groups = X_full.index.str[:12] # 用于分组交叉验证

print("-" * 70)

# --- 辅助函数：特征选择 ---
def perform_feature_selection(X_tr, y_tr, top_n=10):
    """
    仅在训练集上计算差异表达，防止泄露。
    """
    # 这里的 y_tr == 1 现在能正确对齐 X_tr 的索引了
    cancer_tr = X_tr[y_tr == 1]
    normal_tr = X_tr[y_tr == 0]
    
    p_values = []
    features = X_tr.columns
    
    for feature in features:
        c_vals = cancer_tr[feature].values
        n_vals = normal_tr[feature].values
        # 处理方差为0的情况
        if np.var(c_vals) == 0 and np.var(n_vals) == 0:
            p_values.append(1.0)
        else:
            _, p_val = ttest_ind(c_vals, n_vals, equal_var=False, nan_policy='omit')
            p_values.append(p_val)
            
    p_values = np.array([p if not np.isnan(p) else 1.0 for p in p_values])
    # 排序取 Top N (最小 P 值)
    top_indices = np.argsort(p_values)[:top_n]
    return features[top_indices].tolist()

# --- 4. 修正后的交叉验证 (Strict CV) ---
print(">>> Step 4: Performing 5-fold stratified group CV with INTERNAL feature selection...")
print("    (Feature selection is now done inside each fold to prevent leakage)")

n_splits = 5
sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)

fold_auc_scores = []
all_y_test = []
all_y_pred_proba = []
all_y_pred = []
# 用于记录每一折选到了哪些特征
features_selected_per_fold = []

rf_threads = 4 

for fold, (train_idx, test_idx) in enumerate(sgkf.split(X_full, y_full, groups=patient_groups)):
    # 1. 数据分割
    X_train, y_train = X_full.iloc[train_idx], y_full.iloc[train_idx]
    X_test, y_test = X_full.iloc[test_idx], y_full.iloc[test_idx]
    
    # 2. 内部特征选择 (Key Fix!)
    # 这里的 selected_feats 完全由 X_train 决定
    selected_feats = perform_feature_selection(X_train, y_train, top_n=10)
    features_selected_per_fold.append(selected_feats)
    
    X_train_sel = X_train[selected_feats]
    X_test_sel = X_test[selected_feats]
    
    # 3. SMOTE (仅对训练集)
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_sel, y_train)
    
    # 4. 模型训练
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=rf_threads)
    rf.fit(X_train_res, y_train_res)
    
    # 5. 预测
    y_pred_proba = rf.predict_proba(X_test_sel)[:, 1]
    y_pred = rf.predict(X_test_sel)
    
    auc = roc_auc_score(y_test, y_pred_proba)
    fold_auc_scores.append(auc)
    
    all_y_test.extend(y_test)
    all_y_pred_proba.extend(y_pred_proba)
    all_y_pred.extend(y_pred)
    
    print(f"   Fold {fold+1} AUC: {auc:.4f} | Selected features overlap with previous: {len(set(selected_feats))}")

print("-" * 70)

# --- 5. 汇总评估结果 ---
print(">>> Step 5: Aggregating model results...")
mean_auc = np.mean(fold_auc_scores)
std_auc = np.std(fold_auc_scores)
print(f"   Random Forest Mean AUC: {mean_auc:.4f} ± {std_auc:.4f}")

print("Overall Classification Report:")
print(classification_report(all_y_test, all_y_pred, target_names=['Normal', 'Cancer']))

# 混淆矩阵
cm = confusion_matrix(all_y_test, all_y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted Normal', 'Predicted Cancer'],
            yticklabels=['Actual Normal', 'Actual Cancer'])
plt.ylabel('Actual Class', fontdict=font_others)
plt.xlabel('Predicted Class', fontdict=font_others)
plt.title(f'Confusion Matrix (Strict CV)', fontdict=font_title)
plt.savefig(f'{cancer_type}_{target_stage.replace(" ", "")}_Confusion_Matrix.png', dpi=300, bbox_inches='tight')

# ROC 曲线
fpr, tpr, _ = roc_curve(all_y_test, all_y_pred_proba)
plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='red', lw=2.5, label=f'Random Forest (Mean AUC = {mean_auc:.4f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random Chance')
plt.xlabel('False Positive Rate', fontdict=font_others)
plt.ylabel('True Positive Rate', fontdict=font_others)
plt.title(f'ROC Curve for {cancer_type} ({target_stage})', fontdict=font_title)
plt.legend(loc="lower right")
plt.grid(alpha=0.4)
plt.savefig(f'{cancer_type}_{target_stage.replace(" ", "")}_ROC_Curve.png', dpi=300, bbox_inches='tight')
print("-" * 70)

# --- 6. 生成最终的 Top 10 生物标志物 (用于论文表格和 PCA) ---
print(">>> Step 6 & 7: Identifying Final Biomarkers using FULL dataset...")
print("    (Note: This is for final reporting/visualization, distinct from CV accuracy)")

# 在全量数据上做一次特征选择，确定最终的 Top 10
final_top_features = perform_feature_selection(X_full, y_full, top_n=10)

# 输出表格
final_feature_stats = []
pseudo_count = 1e-6
for feat in final_top_features:
    c_vals = X_cancer[feat].values
    n_vals = X_normal[feat].values
    _, p_val = ttest_ind(c_vals, n_vals, equal_var=False)
    lfc = np.log2((np.mean(c_vals) + pseudo_count) / (np.mean(n_vals) + pseudo_count))
    final_feature_stats.append({
        'Feature': feat,
        'p_value': p_val,
        'log2FC': lfc
    })

importance_df = pd.DataFrame(final_feature_stats)
output_csv = f'{cancer_type}_{target_stage.replace(" ", "")}_Top10_eRNA_Table.csv'
importance_df.to_csv(output_csv, index=False)
print(f"   Final Top 10 eRNA table saved to: {output_csv}")
print("   Features:", final_top_features)

# --- PCA 绘图 (使用最终选出的特征) ---
pca_scaler = StandardScaler()
X_final_selected = X_full[final_top_features]
X_scaled_for_pca = pca_scaler.fit_transform(X_final_selected)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled_for_pca)
pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
pca_df['label'] = y_full.values
pca_df['label'] = pca_df['label'].map({1: f'{cancer_type} ({target_stage})', 0: 'Normal'})

plt.figure(figsize=(10, 8))
sns.scatterplot(x='PC1', y='PC2', hue='label', data=pca_df, alpha=0.8, s=60, style='label')
plt.title(f'PCA of Final Top 10 eRNAs', fontdict=font_title)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontdict=font_others)
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontdict=font_others)
plt.legend(title='Sample Type')
plt.savefig(f'{cancer_type}_{target_stage.replace(" ", "")}_PCA_Plot.png', dpi=300, bbox_inches='tight')

print("-" * 70)
print("Analysis complete.")


