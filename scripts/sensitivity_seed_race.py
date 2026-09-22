# -*- coding: utf-8 -*-
"""N4+N5 敏感性分析：
N4 随机种子敏感性：3 个 seed 重跑 XGB，报测试 AUC 稳定性
N5 race-blind 敏感性：移除 race 特征重跑 XGB，量化 race 特征贡献
复用 modeling.py 的特征工程与 XGB 超参。
"""
import pandas as pd, numpy as np, json, os
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

OUT = r"G:/项目文件/CKD多模态AI早诊_本项目专属归档/09_无湿实验执行/results"
cohort = pd.read_parquet(os.path.join(OUT, "cohort.parquet"))

FEATURES = ["age","sex","race","poverty_ratio","education","bmi","sbp","dbp",
            "hba1c","diabetes","total_cholesterol","hdl"]
FEATURES_NO_RACE = [f for f in FEATURES if f != "race"]

def add_dummies(X):
    X = X.copy()
    if "race" in X.columns:
        X["race_black"] = (X["race"] == 4).astype(float)
        X["race_hisp"]  = (X["race"].isin([1, 2])).astype(float)
        X["race_other"] = (~X["race"].isin([3, 4, 1, 2])).astype(float)
    X["female"] = (X["sex"] == 2).astype(float)
    return X.drop(columns=[c for c in ["race", "sex"] if c in X.columns])

def prep_fit(X):
    med = X.median()
    def transform(X_):
        X_ = X_.copy(); miss = X_.isna()
        return pd.concat([X_.fillna(med), miss.add_prefix("miss_")], axis=1)
    return transform

train = cohort[cohort["year"].isin([2011, 2013, 2015])]
test  = cohort[cohort["year"] == 2017]
ytr, yte = train["ckd"].values, test["ckd"].values
print(f"train n={len(train)} (ckd={ytr.sum():.0f}) | test n={len(test)} (ckd={yte.sum():.0f})", flush=True)

def run_xgb(Xtr, Xte, seed):
    m = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                      n_jobs=4, random_state=seed)
    m.fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1]

# ---- N4: seed 敏感性（全 12 特征）----
Xtr_raw, Xte_raw = add_dummies(train[FEATURES]), add_dummies(test[FEATURES])
fit = prep_fit(Xtr_raw)
Xtr, Xte = fit(Xtr_raw), fit(Xte_raw)
print("\n=== N4 seed 敏感性 ===", flush=True)
seeds = [7, 42, 2026]
seed_aucs = {}
for s in seeds:
    p = run_xgb(Xtr, Xte, s)
    a = roc_auc_score(yte, p)
    seed_aucs[str(s)] = round(float(a), 4)
    print(f"  seed {s}: AUC={a:.4f}", flush=True)
mean_a, sd_a = float(np.mean(list(seed_aucs.values()))), float(np.std(list(seed_aucs.values())))
print(f"  mean={mean_a:.4f} ± SD={sd_a:.4f}", flush=True)

# ---- N5: race-blind（移除 race 特征）----
Xtr_nr, Xte_nr = add_dummies(train[FEATURES_NO_RACE]), add_dummies(test[FEATURES_NO_RACE])
fit_nr = prep_fit(Xtr_nr)
Xtr_nr2, Xte_nr2 = fit_nr(Xtr_nr), fit_nr(Xte_nr)
print(f"\n=== N5 race-blind（11 特征，seed 42）===", flush=True)
p_nr = run_xgb(Xtr_nr2, Xte_nr2, 42)
auc_raceblind = float(roc_auc_score(yte, p_nr))
print(f"  race-blind AUC={auc_raceblind:.4f}", flush=True)
print(f"  race 特征增量 = {seed_aucs['42'] - auc_raceblind:+.4f} (全特征 seed42 {seed_aucs['42']} - race-blind {auc_raceblind})", flush=True)

out = {
    "seed_sensitivity": seed_aucs,
    "seed_mean": round(mean_a, 4),
    "seed_sd": round(sd_a, 4),
    "seed_range": [round(min(seed_aucs.values()), 4), round(max(seed_aucs.values()), 4)],
    "race_blind_auc": round(auc_raceblind, 4),
    "race_feature_delta": round(float(seed_aucs["42"]) - auc_raceblind, 4),
}
os.makedirs(os.path.join(OUT, "m2"), exist_ok=True)
with open(os.path.join(OUT, "m2/sensitivity_seed_race.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print("\nsaved: results/m2/sensitivity_seed_race.json", flush=True)
