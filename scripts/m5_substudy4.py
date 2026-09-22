"""M5 子研究④ —— PIR 梯度归因（机制均等 vs 统计均等，熵平衡）
预注册判定（06 v1.1 §7.4）：
  熵平衡特征分布后 ΔAUC(PIR<1.3 vs ≥1.3) < 0.02 → H4a 统计均等（可校准解决）
  残留 ≥ 0.03 → H4b 机制均等（需群体特异模型/干预）
熵平衡：PIR<1.3 组权重 w=exp(X·λ)，λ 使协变量矩与 ≥1.3 组匹配（Hainmueller 2012）
"""
import pandas as pd, numpy as np, json, os
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from scipy.optimize import minimize

OUT = r"G:/项目文件/CKD多模态AI早诊_本项目专属归档/09_无湿实验执行/results"
cohort = pd.read_parquet(os.path.join(OUT, "cohort.parquet"))

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
def auc_w(y, p, w=None):
    if w is None: w = np.ones(len(y))
    return roc_auc_score(y, p, sample_weight=w)

def boot_auc(y, p, w=None, n=2000, seed=11):
    rng = np.random.default_rng(seed)
    if w is None: w = np.ones(len(y))
    idx = np.arange(len(y)); aucs = []
    for _ in range(n):
        i = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[i])) < 2: continue
        aucs.append(roc_auc_score(y[i], p[i], sample_weight=w[i]))
    return [round(float(np.percentile(aucs, 2.5)), 4), round(float(np.percentile(aucs, 97.5)), 4)]

# ---- 基线 ----
auc_lo, auc_hi = auc_w(yte[lo], p_te[lo]), auc_w(yte[hi], p_te[hi])
d_auc_base = auc_hi - auc_lo
print(f"[基线] PIR<1.3: n={lo.sum()} AUC={auc_lo:.3f} | PIR≥1.3: n={hi.sum()} AUC={auc_hi:.3f} | ΔAUC={d_auc_base:+.3f} (论文 0.040)")

# ---- 熵平衡：PIR<1.3 组协变量矩匹配到 ≥1.3 组 ----
def entropy_balance_weights(X_t, X_c):
    """X_t: 处理组协变量（已标准化），X_c: 对照组。返回权重（均值=1）。"""
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

cov_lo = te.loc[lo, BALANCE_VARS].fillna(te.loc[lo, BALANCE_VARS].median())
cov_hi = te.loc[hi, BALANCE_VARS].fillna(te.loc[hi, BALANCE_VARS].median())
mu, sd = cov_hi.mean(0), cov_hi.std(0).replace(0, 1)
cov_lo_s = (cov_lo - mu) / sd
cov_hi_s = (cov_hi - mu) / sd
w_lo = entropy_balance_weights(cov_lo_s.values, cov_hi_s.values)

# 平衡后矩检验
bal_mom = (w_lo[:, None] * cov_lo_s.values).mean(0)
print("\n[熵平衡] 最大协变量矩偏差（标准化尺度）:")
print(f"  平衡前: {np.abs(cov_lo_s.mean(0)).max():.3f} | 平衡后: {np.abs(bal_mom).max():.4f}")

# ---- 平衡后 AUC ----
w_all = np.ones(len(te))
w_all[lo] = w_lo
auc_lo_b = auc_w(yte[lo], p_te[lo], w_lo)
auc_hi_b = auc_w(yte[hi], p_te[hi])
d_auc_bal = auc_hi_b - auc_lo_b
ci_lo = boot_auc(yte[lo], p_te[lo], w_lo)
print(f"\n[平衡后] PIR<1.3 AUC={auc_lo_b:.3f} {ci_lo} | PIR≥1.3 AUC={auc_hi_b:.3f} | ΔAUC={d_auc_bal:+.3f}")
print(f"  ΔAUC 变化: {d_auc_base:+.3f} → {d_auc_bal:+.3f} (变化 {d_auc_bal - d_auc_base:+.3f})")

if abs(d_auc_bal) < 0.02:
    decision = "H4a 统计均等（梯度来自特征分布差异，重校准可解决）"
elif abs(d_auc_bal) >= 0.03:
    decision = "H4b 机制均等（梯度残留 = 通路强度差异，需群体特异模型）"
else:
    decision = f"中间状态（ΔAUC={d_auc_bal:+.3f}）"
print(f"\n[子研究④ 判决] {decision}")

results = {
    "baseline": {"auc_low_pir": round(float(auc_lo), 4), "auc_high_pir": round(float(auc_hi), 4),
                 "delta_auc": round(float(d_auc_base), 4)},
    "after_balancing": {"auc_low_pir": round(float(auc_lo_b), 4), "auc_high_pir": round(float(auc_hi_b), 4),
                        "delta_auc": round(float(d_auc_bal), 4),
                        "delta_change": round(float(d_auc_bal - d_auc_base), 4),
                        "max_covariate_moment_deviation": round(float(np.abs(bal_mom).max()), 5)},
    "decision": decision,
}
with open(os.path.join(OUT, "m2", "substudy4_results.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"\n已保存: substudy4_results.json")
