"""M2 sub-studies 1-2: footprint probe (SHAP x CKM pathway) and NADKD-stratified detection.
Follows the CKD-detection v3 pipeline (modeling.py): same features, hyperparameters and split.
Pre-specified decision thresholds:
  Sub-study 1: ARI >= 0.30 with permutation p < 0.001 supports the footprint hypothesis; ARI <= 0.10 or a single feature carrying > 60% of SHAP supports co-occurrence.
  Sub-study 2: NADKD stratum AUC >= 0.75 with delta AUC (all minus stratum) < 0.10 indicates independent detection; AUC <= 0.65 or delta >= 0.15 indicates dependence.
Output: results/m2/substudy12_results.json and a console report.
"""
import pandas as pd, numpy as np, json, os, sys
from xgboost import XGBClassifier, DMatrix
from sklearn.metrics import roc_auc_score
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score

OUT = r"results"
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
print(f"[baseline] test AUC = {roc_auc_score(yte, p_te):.4f} (manuscript 0.810) | n_test={len(yte)}")

results = {"baseline_auc": round(float(roc_auc_score(yte, p_te)), 4), "n_test": int(len(yte))}

# ============ sub-study 1: footprint probe ============
print("\n=== sub-study 1: footprint probe (SHAP x CKM pathway) ===")
shap = model.get_booster().predict(DMatrix(Xte), pred_contribs=True)
shap = np.asarray(shap)[:, :-1]  # drop the bias column

# pre-specified feature-to-pathway map
PATHWAYS = {
    "hemodynamic": ["sbp", "dbp"],
    "glycotoxic":  ["hba1c", "diabetes"],
    "metabolic":   ["total_cholesterol", "hdl"],
    "demographic": ["age", "female", "race_black", "race_hisp", "race_other",
                    "poverty_ratio", "education", "bmi"],
}
# miss_* indicators form a separate data-quality layer
miss_feats = [f for f in features_model if f.startswith("miss_")]
PATHWAYS["missing_indicator"] = miss_feats
feat_to_path = {f: p for p, fs in PATHWAYS.items() for f in fs}
assert set(feat_to_path) == set(features_model), f"feature-to-pathway map does not match model features: {set(feat_to_path) ^ set(features_model)}"

mean_abs_shap = np.abs(shap).mean(axis=0)
order = np.argsort(-mean_abs_shap)
print("SHAP importance ranking:")
for i in order:
    print(f"  {features_model[i]:<22} {mean_abs_shap[i]:.4f}")

# between-feature distance: 1 - |SHAP correlation| (similarity of effect patterns)
corr = np.corrcoef(shap.T)
corr = np.nan_to_num(corr, nan=0.0)          # zero-variance features (e.g. all-zero miss_* columns) get zero correlation
dist = 1 - np.abs(corr)
np.fill_diagonal(dist, 0)
dist = (dist + dist.T) / 2                     # enforce symmetry
dist = np.clip(dist, 0, None)
link = linkage(squareform(dist), method="ward")
n_clusters = len(PATHWAYS)
labels = fcluster(link, n_clusters, criterion="maxclust")
true = np.array([list(PATHWAYS).index(feat_to_path[f]) for f in features_model])
ari = adjusted_rand_score(true, labels)

# permutation null: shuffle pathway labels 500 times
rng = np.random.default_rng(42)
null_aris = []
for _ in range(500):
    perm = rng.permutation(true)
    null_aris.append(adjusted_rand_score(perm, labels))
null_aris = np.array(null_aris)
p_val = float((null_aris >= ari).mean())

# pocket hit rate: whether features of the same pathway cluster together
pairs_same = pairs_total = 0
for i in range(len(features_model)):
    for j in range(i+1, len(features_model)):
        if feat_to_path[features_model[i]] == feat_to_path[features_model[j]]:
            pairs_total += 1
            if labels[i] == labels[j]:
                pairs_same += 1
pocket_hit = pairs_same / pairs_total if pairs_total else np.nan

# single-feature share (age > 60% would signal co-occurrence)
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
    "decision": ("H1a footprint hypothesis supported" if ari >= 0.30 and p_val < 0.001
                 else "H1b co-occurrence hypothesis supported" if (ari <= 0.10 or top1_share > 0.60)
                 else "intermediate (requires further discrimination)"),
}
results["substudy1_footprint"] = s1
print(f"\n[sub-study 1] ARI = {ari:.3f} (permutation p={p_val:.4f}, null mean={null_aris.mean():.3f})")
print(f"  pocket hit rate = {pocket_hit:.3f} | age SHAP share = {age_share:.3f} | decision: {s1['decision']}")

# ============ sub-study 2: NADKD-stratified detection ============
print("\n=== sub-study 2: detection independence in the uACR blind spot ===")
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
    "uACR < 30, all adults (blind-spot population)": te["uacr"] < 30,
    "uACR < 30 with diabetes (NADKD risk group)": (te["uacr"] < 30) & (te["diabetes"] == 1),
    "uACR < 30 without diabetes": (te["uacr"] < 30) & (te["diabetes"] == 0),
    "uACR < 30, age 18-44 (links to sub-study 3)": (te["uacr"] < 30) & (te["age"] < 45),
    "uACR < 30 with eGFR >= 60 (blind-spot control without eGFR loss)": (te["uacr"] < 30) & (te["egfr"] >= 60),
}
s2 = {"strata": {}}
for name, mask in strata.items():
    m = mask.values if hasattr(mask, "values") else mask
    n = int(m.sum())
    if n < 20 or len(np.unique(yte[m])) < 2:
        s2["strata"][name] = {"n": n, "auc": None, "note": "insufficient sample size or single-class labels"}
        print(f"  [{name}] n={n} - insufficient sample or single-class labels")
        continue
    auc = roc_auc_score(yte[m], p_te[m])
    ci = boot_auc(yte[m], p_te[m])
    dauc = roc_auc_score(yte, p_te) - auc
    s2["strata"][name] = {"n": n, "auc": round(float(auc), 4), "auc_ci95": ci,
                          "delta_auc_vs_all": round(float(dauc), 4)}
    print(f"  [{name}] n={n} | AUC={auc:.3f} {ci} | delta AUC vs all = {dauc:+.3f}")

nad = s2["strata"].get("uACR < 30, all adults (blind-spot population)", {})
if nad.get("auc") is not None:
    nad["decision"] = ("H2a independent footprint supported" if nad["auc"] >= 0.75 and nad["delta_auc_vs_all"] < 0.10
                       else "H2b dependence on albuminuria supported" if (nad["auc"] <= 0.65 or nad["delta_auc_vs_all"] >= 0.15)
                       else "intermediate")
    print(f"\n[sub-study 2 decision] {nad['decision']}")
results["substudy2_nadkd"] = s2

with open(os.path.join(M2OUT, "substudy12_results.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"\nsaved: {M2OUT}/substudy12_results.json")
