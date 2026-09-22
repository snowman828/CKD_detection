# -*- coding: utf-8 -*-
"""S4 熵平衡 bootstrap 补算 —— n_boot 50 -> 1000（红队 P1-8）
复用 m5_substudy4.py 的熵平衡逻辑，只重算 ΔAUC bootstrap CI 并更新 robustness_results.json 的 s4_entropy。
方向与 m11_robustness.py 一致：low-PIR deficit = auc_low - auc_high（负值表示低 PIR 组更差）。
"""
import pandas as pd, numpy as np, json, os
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from scipy.optimize import minimize

BASE = r"G:/项目文件/CKD多模态AI早诊_本项目专属归档/09_无湿实验执行"
cohort = pd.read_parquet(os.path.join(BASE, "results/cohort.parquet"))

FEATURES = ["age","sex","race","poverty_ratio","education","bmi","sbp","dbp",
            "hba1c","diabetes","total_cholesterol","hdl"]
BALANCE_VARS = ["age","bmi","sbp","dbp","hba1c","diabetes","total_cholesterol","hdl",
                "education","race"]

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

def entropy_balance_weights(X_t, X_c):
    mu_c = X_c.mean(0)
    def mom(lmb):
        w = np.exp(X_t @ lmb)
        return (w[:, None] * X_t).sum(0) / w.sum() - mu_c
    def loss(lmb):
        m = mom(lmb)
        return 0.5 * m @ m + 1e-6 * lmb @ lmb
    res = minimize(loss, np.zeros(X_t.shape[1]), method="BFGS",
                   options={"maxiter": 500, "gtol": 1e-8})
    w = np.exp(X_t @ res.x)
    return w / w.mean()

train = cohort[cohort["year"].isin([2011, 2013, 2015])]
test  = cohort[cohort["year"] == 2017]
Xtr_raw, Xte_raw = add_dummies(train[FEATURES]), add_dummies(test[FEATURES])
fit = prep_fit(Xtr_raw)
Xtr, Xte = fit(Xtr_raw), fit(Xte_raw)
ytr, yte = train["ckd"].values, test["ckd"].values

model = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                      n_jobs=4, random_state=42)
model.fit(Xtr, ytr)
p_te = model.predict_proba(Xte)[:, 1]

te = test.reset_index(drop=True)
lo, hi = te["poverty_ratio"] < 1.3, te["poverty_ratio"] >= 1.3

cov_lo = te.loc[lo, BALANCE_VARS].fillna(te.loc[lo, BALANCE_VARS].median())
cov_hi = te.loc[hi, BALANCE_VARS].fillna(te.loc[hi, BALANCE_VARS].median())
mu, sd = cov_hi.mean(0), cov_hi.std(0).replace(0, 1)
cov_lo_s = (cov_lo - mu) / sd
cov_hi_s = (cov_hi - mu) / sd
w_lo = entropy_balance_weights(cov_lo_s.values, cov_hi_s.values)

def auc_w(y, p, w=None):
    if w is None: w = np.ones(len(y))
    return roc_auc_score(y, p, sample_weight=w)

auc_lo_b = auc_w(yte[lo], p_te[lo], w_lo)
auc_hi_b = auc_w(yte[hi], p_te[hi])
d_point = auc_lo_b - auc_hi_b  # low - high（与 m11 方向一致）
print(f"[点估计] 平衡后 low AUC={auc_lo_b:.4f} high AUC={auc_hi_b:.4f} | low−high Δ={d_point:+.4f}")

# bootstrap n=1000（红队 P1-8）——对观测 bootstrap，lo/hi 组内局部索引
rng = np.random.default_rng(2026)
y_lo, p_lo, w_lo_arr = yte[lo], p_te[lo], w_lo
y_hi, p_hi = yte[hi], p_te[hi]
idx_lo = np.arange(len(y_lo)); idx_hi = np.arange(len(y_hi))
diffs = []
for _ in range(1000):
    i = rng.choice(idx_lo, len(idx_lo), replace=True)
    j = rng.choice(idx_hi, len(idx_hi), replace=True)
    if len(np.unique(y_lo[i])) < 2 or len(np.unique(y_hi[j])) < 2:
        continue
    d = roc_auc_score(y_lo[i], p_lo[i], sample_weight=w_lo_arr[i]) - roc_auc_score(y_hi[j], p_hi[j])
    diffs.append(d)
ci = [round(float(np.percentile(diffs, 2.5)), 4), round(float(np.percentile(diffs, 97.5)), 4)]
print(f"[bootstrap n=1000] low−high deficit 95% CI: {ci}")

# 更新 robustness_results.json
rp = os.path.join(BASE, "results/m2/robustness_results.json")
r = json.load(open(rp, encoding="utf-8"))
r["s4_entropy"] = {
    "delta_auc_point": round(float(d_point), 4),
    "bootstrap_ci95": ci,
    "n_boot": 1000,
}
with open(rp, "w", encoding="utf-8") as f:
    json.dump(r, f, ensure_ascii=False, indent=2)
print("已更新 robustness_results.json -> s4_entropy.n_boot=1000")
