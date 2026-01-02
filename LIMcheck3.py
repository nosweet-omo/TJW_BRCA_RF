import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

print(f"{'='*40}")
print("🚀 最终验证：特征聚合与 AUC 计算")
print(f"{'='*40}")

# 1. 读取刚才生成的计数文件
val_df = pd.read_csv("my_direct_counts.txt", sep='\t', comment='#', index_col=0)
val_X = val_df.iloc[:, 5:].T # 转置，行是样本
# 清洗样本名
val_X.index = [x.split('/')[-1].split('.')[0] for x in val_X.index]

print("\n📊 原始计数矩阵 (部分):")
print(val_X.iloc[:, :5].head())

# 2. 简单的 CPM 归一化 (关键步骤，消除测序深度差异)
lib_sizes = val_X.sum(axis=1)
val_X_cpm = val_X.div(lib_sizes, axis=0) * 1e6

# 3. 【核心策略】聚合 Chr2 Cluster 的信号
# 找出所有 chr2 的特征
chr2_cols = [c for c in val_X_cpm.columns if c.startswith('chr2')]
print(f"\n🧩 发现 {len(chr2_cols)} 个 Chr2 簇特征，正在聚合...")

# 创建一个新特征：Chr2_Cluster_Sum
val_X_cpm['Chr2_Cluster_Sum'] = val_X_cpm[chr2_cols].sum(axis=1)

# 看看聚合后的数值是不是好看了？
print("\n🔥 聚合后的 Chr2 信号强度:")
print(val_X_cpm['Chr2_Cluster_Sum'])

# 4. 获取真实标签 (你需要手动确认一下)
# 这里我们尝试自动读取，读不到就请你手动填
try:
    meta = pd.read_csv("External_Validation/SraRunTable.csv")
    # 假设 'Run' 是 ID， 'source_name' 是类型
    # 请根据实际情况调整列名！
    meta_dict = dict(zip(meta['Run'], meta['source_name'])) 
    
    y_true = []
    for sid in val_X.index:
        label = meta_dict.get(sid, 'Unknown')
        print(f"   样本 {sid} -> {label}")
        # 简单判断: 包含 Tumor/Cancer 设为 1，否则 0
        if 'Tumor' in label or 'Cancer' in label:
            y_true.append(1)
        else:
            y_true.append(0)
            
    print(f"\n🏷️ 提取到的标签: {y_true}")

    # 5. 用聚合特征算 AUC
    # 我们只用这个聚合特征，或者 Top 10 所有特征
    if len(set(y_true)) > 1: # 必须要有正负两类
        # 方法 A: 单独看 Chr2 Sum 的区分度
        auc_cluster = roc_auc_score(y_true, val_X_cpm['Chr2_Cluster_Sum'])
        print(f"\n🏆 基于 Chr2 聚合信号的 AUC: {auc_cluster:.4f}")
        
        if auc_cluster > 0.8:
            print("🎉 成功了！聚合后的信号足够强！")
        else:
            print("⚠️ 信号还是太弱。")
            
    else:
        print("❌ 无法计算 AUC：样本标签只有一种，或者没匹配上元数据。")

except Exception as e:
    print(f"❌ 元数据匹配失败: {e}")
    print("👉 请手动核对上面的 'Chr2_Cluster_Sum' 数值：")
    print("   如果是 Tumor 样本数值高，Normal 样本数值低，那就直接赢了！")