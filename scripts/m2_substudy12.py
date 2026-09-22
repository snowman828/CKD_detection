"""M2 子研究①② —— 足迹探针（SHAP×CKM 通路）+ NADKD 分层检测
遵循 MACKI-Dry v3 管线（modeling.py）：同特征/同参数/同划分，保证模型一致。
预注册判定阈值（06 文档 v1.1 §7.1-7.2）：
  子研究①：ARI ≥ 0.30 且 p<0.001(vs 置换) → 足迹假说；ARI≤0.10 或单特征>60% SHAP → 共现假说
  子研究②：NADKD 层 AUC ≥ 0.75 且 ΔAUC(全体−层)<0.10 → 独立；AUC≤0.65 或 ΔAUC≥0.15 → 依赖
输出：results/m2/substudy12_results.json + 控制台报告
"""
import pandas as pd, numpy as np, json, os, sys
from xgboost import XGBClassifier, DMatrix
from sklearn.metrics import roc_auc_score
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score

OUT = r"G:/项目文件/CKD多模态AI早诊_本项目专属归档/09_无湿实验执行/results"
M2OUT = os.path.join(OUT, "m2"); os.makedirs(M2OUT, exist_ok=True)
cohort = pd.read_parquet(os.path.join(OUT, "cohort.parquet"))

FEATURES = ["age","sex","race","poverty_ratio","education","bmi","sbp","dbp",
            "hba1c","diabetes","total_cholesterol","hdl"]
TARGET = "ckd"

def prep_fit(X):
    med = X.median()
    def transform(X_):
        X_ = X_.copy(); miss = X_.isna()
        return pd.concat([X_.fillna(med), miss.add_prefix("miss_")], axis=1)
    return transform

def add_dummies(X):
    X = X.copy()
    X["race_black"] = (X["race"] == 4).astype(float)
    X["race_hisp"]  = (X["race"].isin([1, 2])).astype(float)
    X["race_other"] = (~X["race"].isin([3, 4, 1, 2])).astype(float)
    X["female"] = (X["sex"] == 2).astype(float)
    return X.drop(columns=["race", "sex"])

train = cohort[cohort["year"].isin([2011, 2013, 2015])]
test  = cohort[cohort["year"] == 2017]
Xtr_raw, Xte_raw = add_dummies(train[FEATURES]), add_dummies(test[FEATURES])
fit = prep_fit(Xtr_raw)
Xtr, Xte = fit(Xtr_raw), fit(Xte_raw)
ytr, yte = train[TARGET].values, test[TARGET].values
features_model = list(Xtr.columns)

model = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                      n_jobs=4, random_state=42)
model.fit(Xtr, ytr)
p_te = model.predict_proba(Xte)[:, 1]
print(f"[基线] test AUC = {roc_auc_score(yte, p_te):.4f} (论文 0.810) | n_test={len(yte)}")

results = {"baseline_auc": round(float(roc_auc_score(yte, p_te)), 4), "n_test": int(len(yte))}

# ============ 子研究① 足迹探针 ============
print("\n=== 子研究① 足迹探针（SHAP × CKM 通路） ===")
shap = model.get_booster().predict(DMatrix(Xte), pred_contribs=True)
shap = np.asarray(shap)[:, :-1]  # 去掉 bias 列

# 预注册映射表（对模型输入特征）
PATHWAYS = {
    "hemodynamic": ["sbp", "dbp"],
    "glycotoxic":  ["hba1c", "diabetes"],
    "metabolic":   ["total_cholesterol", "hdl"],
    "demographic": ["age", "female", "race_black", "race_hisp", "race_other",
                    "poverty_ratio", "education", "bmi"],
}
# miss_* 列单独一层（数据质量层）
miss_feats = [f for f in features_model if f.startswith("miss_")]
PATHWAYS["missing_indicator"] = miss_feats
feat_to_path = {f: p for p, fs in PATHWAYS.items() for f in fs}
assert set(feat_to_path) == set(features_model), f"映射表与模型特征不一致: {set(feat_to_path) ^ set(features_model)}"

mean_abs_shap = np.abs(shap).mean(axis=0)
order = np.argsort(-mean_abs_shap)
print("SHAP 重要性排序:")
for i in order:
    print(f"  {features_model[i]:<22} {mean_abs_shap[i]:.4f}")

# 特征间距离：1 - |SHAP 相关|（特征作用模式相似度）
corr = np.corrcoef(shap.T)
corr = np.nan_to_num(corr, nan=0.0)          # 零方差特征（如全 0 的 miss 列）→ 相关设 0
dist = 1 - np.abs(corr)
np.fill_diagonal(dist, 0)
dist = (dist + dist.T) / 2                     # 强制对称
dist = np.clip(dist, 0, None)
link = linkage(squareform(dist), method="ward")
n_clusters = len(PATHWAYS)
labels = fcluster(link, n_clusters, criterion="maxclust")
true = np.array([list(PATHWAYS).index(feat_to_path[f]) for f in features_model])
ari = adjusted_rand_score(true, labels)

# 置换零分布：打乱映射标签 500 次
rng = np.random.default_rng(42)
null_aris = []
for _ in range(500):
    perm = rng.permutation(true)
    null_aris.append(adjusted_rand_score(perm, labels))
null_aris = np.array(null_aris)
p_val = float((null_aris >= ari).mean())

# 通路口袋命中率：每个通路的特征是否聚类到一起（同通路特征对在同一簇的比例）
pairs_same = pairs_total = 0
for i in range(len(features_model)):
    for j in range(i+1, len(features_model)):
        if feat_to_path[features_model[i]] == feat_to_path[features_model[j]]:
            pairs_total += 1
            if labels[i] == labels[j]:
                pairs_same += 1
pocket_hit = pairs_same / pairs_total if pairs_total else np.nan

# 单特征占比（年龄 >60% → 共现假说信号）
age_share = mean_abs_shap[features_model.index("age")] / mean_abs_shap.sum()
top1_share = mean_abs_shap.max() / mean_abs_shap.sum()

s1 = {
    "ari": round(float(ari), 4),
    "ari_permutation_p": round(p_val, 4),
    "null_ari_mean": round(float(null_aris.mean()), 4),
    "null_ari_95": [round(float(np.percentile(null_aris, 2.5)), 4),
                    round(float(np.percentile(null_aris, 97.5)), 4)],
    "pocket_hit_rate": round(float(pocket_hit), 4),
    "age_share_of_shap": round(float(age_share), 4),
    "top1_feature_share": round(float(top1_share), 4),
    "top_feature": features_model[order[0]],
    "cluster_labels": {features_model[i]: int(labels[i]) for i in range(len(features_model))},
    "decision": ("H1a 足迹假说支持" if ari >= 0.30 and p_val < 0.001
                 else "H1b 共现假说支持" if (ari <= 0.10 or top1_share > 0.60)
                 else "中间状态（需进一步判别）"),
}
results["substudy1_footprint"] = s1
print(f"\n[子研究①] ARI = {ari:.3f} (置换 p={p_val:.4f}, null 均值={null_aris.mean():.3f})")
print(f"  通路口袋命中率 = {pocket_hit:.3f} | 年龄 SHAP 占比 = {age_share:.3f} | 判决: {s1['decision']}")

# ============ 子研究② NADKD 分层检测 ============
print("\n=== 子研究② uACR 盲区检测独立性 ===")
te = test.reset_index(drop=True)

def boot_auc(y, p, n=2000, seed=7):
    rng = np.random.default_rng(seed)
    aucs = []
    idx = np.arange(len(y))
    for _ in range(n):
        i = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[i])) < 2: continue
        aucs.append(roc_auc_score(y[i], p[i]))
    return [round(float(np.percentile(aucs, 2.5)), 4), round(float(np.percentile(aucs, 97.5)), 4)]

strata = {
    "all": te["ckd"].notna(),
    "uACR<30 全部成人（盲区人群）": te["uacr"] < 30,
    "uACR<30 且糖尿病（NADKD 风险人群）": (te["uacr"] < 30) & (te["diabetes"] == 1),
    "uACR<30 且非糖尿病": (te["uacr"] < 30) & (te["diabetes"] == 0),
    "uACR<30 且年龄 18-44（衔接子研究③）": (te["uacr"] < 30) & (te["age"] < 45),
    "uACR<30 且 eGFR>=60（无肾功能下降的盲区对照）": (te["uacr"] < 30) & (te["egfr"] >= 60),
}
s2 = {"strata": {}}
for name, mask in strata.items():
    m = mask.values if hasattr(mask, "values") else mask
    n = int(m.sum())
    if n < 20 or len(np.unique(yte[m])) < 2:
        s2["strata"][name] = {"n": n, "auc": None, "note": "样本不足或标签单一"}
        print(f"  [{name}] n={n} — 样本不足或标签单一")
        continue
    auc = roc_auc_score(yte[m], p_te[m])
    ci = boot_auc(yte[m], p_te[m])
    dauc = roc_auc_score(yte, p_te) - auc
    s2["strata"][name] = {"n": n, "auc": round(float(auc), 4), "auc_ci95": ci,
                          "delta_auc_vs_all": round(float(dauc), 4)}
    print(f"  [{name}] n={n} | AUC={auc:.3f} {ci} | ΔAUC vs 全体={dauc:+.3f}")

nad = s2["strata"].get("uACR<30 全部成人（盲区人群）", {})
if nad.get("auc") is not None:
    nad["decision"] = ("H2a 独立足迹支持" if nad["auc"] >= 0.75 and nad["delta_auc_vs_all"] < 0.10
                       else "H2b 依赖白蛋白尿支持" if (nad["auc"] <= 0.65 or nad["delta_auc_vs_all"] >= 0.15)
                       else "中间状态")
    print(f"\n[子研究② 判决] {nad['decision']}")
results["substudy2_nadkd"] = s2

with open(os.path.join(M2OUT, "substudy12_results.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"\n已保存: {M2OUT}/substudy12_results.json")
