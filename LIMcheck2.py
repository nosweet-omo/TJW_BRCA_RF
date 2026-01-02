import pandas as pd
import numpy as np
import os
import glob
import subprocess
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix

# ==========================================
# 1. 你的 Top 10 黄金特征 (必须严格一致)
# ==========================================
MY_TOP_10 = [
    "chr7:27184167", "chr8:22859168", "chr2:216585197", 
    "chr2:216585387", "chr2:216584917", "chr3:138069944", 
    "chr2:216585497", "chr7:27183697", "chr1:56957682", 
    "chr2:216584427"
]

# ==========================================
# 2. 路径配置 (请再次确认)
# ==========================================
# 存放 BAM 文件的文件夹
BAM_DIR = "External_Validation/results" 
# 存放元数据的 CSV
META_FILE = "External_Validation/SraRunTable.csv" 
# 你的 TCGA 训练数据
TRAIN_FILE = "TCGA_RPKM_eRNA_300k_peaks_in_Super_enhancer_BRCA.txt"

print(f"{'='*50}")
print("🚀 正在启动：核弹级外部验证 (Re-Quantification Pipeline)")
print(f"{'='*50}")

# ==========================================
# Step 1: 制作 SAF 刻度尺
# ==========================================
print("\n>>> [1/5] 生成 SAF 注释文件...")
saf_rows = []
for feat in MY_TOP_10:
    # 拆分坐标
    chrom, pos = feat.split(':')
    # 设定宽度：队友虽然过滤了，但我们重新定量要给够宽度
    # 假设 peak 宽度为 500bp (+/- 250)
    start = int(pos) - 250
    end = int(pos) + 250
    saf_rows.append([feat, chrom, start, end, '+'])

df_saf = pd.DataFrame(saf_rows, columns=['GeneID', 'Chr', 'Start', 'End', 'Strand'])
df_saf.to_csv("my_top10.saf", sep='\t', index=False, header=False)
print(" ✅ my_top10.saf 已生成")

# ==========================================
# Step 2: 运行 featureCounts (数数)
# ==========================================
print("\n>>> [2/5] 正在运行 featureCounts (直接读取 BAM)...")
bam_files = glob.glob(os.path.join(BAM_DIR, "*.bam"))

if not bam_files:
    print(f"❌ 致命错误: 在 {BAM_DIR} 下没找到任何 .bam 文件！无法验证。")
    exit()

print(f"   发现 {len(bam_files)} 个 BAM 文件，开始定量...")

# 检查是否已存在结果，避免重复跑 (featureCounts 挺快的，但能省则省)
output_counts = "my_direct_counts.txt"
if os.path.exists(output_counts) and os.path.getsize(output_counts) > 100:
    print("   ⚠️ 检测到已存在计数文件，跳过定量步骤，直接使用现成文件。")
    print("   (如果想重新跑，请手动删除 my_direct_counts.txt)")
else:
    cmd = [
        "featureCounts",
        "-T", "8",                  # 线程
        "-p",                       # 双端测序
        "-F", "SAF",                # 格式
        "-a", "my_top10.saf",       # 注释
        "-o", output_counts,        # 输出
        "-M",                       # 【关键】统计 Multi-mapping reads (救回那几千万条数据)
        "-O",                       # Allow Multi-overlap
        "--fraction",               # 如果一条read比对到3个地方，每个地方算1/3 (更科学)
        "-s", "0"                   # 【关键】忽略链的方向 (Unstranded)，防止正负链搞反
    ] + bam_files
    
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(" ✅ 定量完成！")
    except Exception as e:
        print(f"❌ featureCounts 运行失败: {e}")
        print("   请确认服务器安装了 subread (conda install subread)")
        exit()

# ==========================================
# Step 3: 数据清洗与标准化
# ==========================================
print("\n>>> [3/5] 处理验证集数据...")
val_df = pd.read_csv(output_counts, sep='\t', comment='#', index_col=0)
# 提取数据列 (第6列以后)
val_X = val_df.iloc[:, 5:].T

# 清洗样本名: '.../SRR12345.bam' -> 'SRR12345'
val_X.index = [os.path.basename(x).split('.')[0] for x in val_X.index]
print(f"   提取到 {val_X.shape[0]} 个外部样本。")

# --- 简单的 CPM 归一化 ---
# 因为 TCGA 是 RPKM，这里用 CPM (Counts Per Million) 近似替代
# log2(CPM + 1)
lib_sizes = val_X.sum(axis=1)
val_X_cpm = val_X.div(lib_sizes, axis=0) * 1e6
val_X_final = np.log2(val_X_cpm + 1)

# ==========================================
# Step 4: 重新训练 TCGA 模型
# ==========================================
print("\n>>> [4/5] 用 Top 10 特征重训 TCGA 模型...")
try:
    tcga = pd.read_csv(TRAIN_FILE, sep='\t', index_col=0)
    # 只取这10个
    X_train = tcga.loc[MY_TOP_10].T
except KeyError as e:
    print(f"❌ 训练失败: TCGA 原始数据里找不到这些特征: {e}")
    exit()

# 制作标签
y_train = []
for idx in X_train.index:
    if '01' in idx[13:15] or '_tumor' in idx:
        y_train.append(1)
    elif '11' in idx[13:15] or '_normal' in idx:
        y_train.append(0)
    else:
        y_train.append(-1) # 丢弃

y_train = np.array(y_train)
mask = y_train != -1
X_train = X_train[mask]
y_train = y_train[mask]

# Log转换 (TCGA数据如果本身没log，这里要log；如果你的文件已经是RPKM，通常需要log)
# 假设你的文件是 RPKM/FPKM raw value
if X_train.max().max() > 100:
    X_train = np.log2(X_train + 1)

# 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_scaled, y_train)
print(" ✅ 模型训练完毕。")

# ==========================================
# Step 5: 预测与结果匹配
# ==========================================
print("\n>>> [5/5] 最终预测与验证...")

# 确保列顺序一致
val_X_final = val_X_final[MY_TOP_10]
val_X_scaled = scaler.transform(val_X_final) # 用训练集的scaler

# 预测概率
probs = clf.predict_proba(val_X_scaled)[:, 1]
val_X_final['Pred_Prob'] = probs

# 尝试读取元数据计算 AUC
try:
    meta = pd.read_csv(META_FILE)
    # 自动寻找 ID 列 (SRR...)
    id_col = None
    for col in meta.columns:
        if meta[col].astype(str).str.contains('SRR').any():
            id_col = col
            break
    
    # 自动寻找 Label 列
    label_col = None
    for col in meta.columns:
        if meta[col].astype(str).str.lower().isin(['tumor', 'cancer', 'normal', 'tissue']).any():
            label_col = col
            # 优先找 explicit 的
            if 'source_name' in col or 'Group' in col:
                break
    
    if id_col and label_col:
        print(f"   自动识别元数据: ID列=[{id_col}], 分组列=[{label_col}]")
        
        # 映射字典
        meta_dict = dict(zip(meta[id_col], meta[label_col]))
        
        y_true = []
        y_scores = []
        
        print("\n   --- 详细预测结果 ---")
        print(f"   {'SampleID':<15} {'True_Label':<20} {'Pred_Prob (Cancer)':<10}")
        print("-" * 50)
        
        for sid in val_X_final.index:
            if sid in meta_dict:
                true_label_str = str(meta_dict[sid])
                prob = val_X_final.loc[sid, 'Pred_Prob']
                
                # 简单的关键词判断
                is_cancer = 1 if ('tumor' in true_label_str.lower() or 'cancer' in true_label_str.lower()) else 0
                if 'normal' in true_label_str.lower(): is_cancer = 0
                
                y_true.append(is_cancer)
                y_scores.append(prob)
                
                print(f"   {sid:<15} {true_label_str[:20]:<20} {prob:.4f}")
        
        if len(y_true) > 0:
            auc = roc_auc_score(y_true, y_scores)
            print(f"\n{'='*30}")
            print(f"🏆 外部验证 AUC: {auc:.4f}")
            print(f"{'='*30}")
            if auc > 0.8:
                print("🎉 恭喜！结果非常棒！文章稳了！")
            elif auc > 0.6:
                print("🆗 结果还可以，有预测潜力。")
            else:
                print("⚠️ 结果一般，可能存在批次效应，或Top10在外部数据不表达。")
    else:
        print("⚠️ 无法自动解析元数据列名，请手动查看 my_direct_counts.txt 和预测概率。")

except Exception as e:
    print(f"⚠️ 元数据处理出错: {e}")
    print("   预测概率已保存在 val_X_final DataFrame 中。")