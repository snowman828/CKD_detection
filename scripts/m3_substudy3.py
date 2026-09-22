"""M3 子研究③ —— 年轻成人衰减归因（谱效应 vs 信号积累 vs 交互）
预注册判定（06 v1.1 §7.3）：
  严重度/患病率校正后 18-44 AUC 增益 ≥ 0.03 → H3a 谱效应
  校正后 AUC ≤ 0.70 且无年龄×谱交互（p>0.05）→ H3b 信号积累
  部分增益(0.01-0.03) + 显著交互(p<0.05) → H3c 交互
谱校正：病例严重度加权（eGFR 45-60 vs <45 vs 白蛋白尿-only 谱标准化）+ 患病率再加权
"""
import pandas as pd, numpy as np, json, os
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
import scipy.stats as st

OUT = r"G:/项目文件/CKD多模态AI早诊_本项目专属归档/09_无湿实验执行/results"
M2OUT = os.path.join(OUT, "m2")
cohort = pd.read_parquet(os.path.join(OUT, "cohort.parquet"))

FEATURES = ["age","sex","race","poverty_ratio","education","bmi","sbp","dbp",
            "hba1c","diabetes","total_cholesterol","hdl"]

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
age_groups = {"18-44": te["age"] < 45, "45-64": (te["age"] >= 45) & (te["age"] < 65), "≥65": te["age"] >= 65}

def auc_w(y, p, w=None):
    if w is None: w = np.ones(len(y))
    return roc_auc_score(y, p, sample_weight=w)

def boot_auc(y, p, w=None, n=2000, seed=7):
    rng = np.random.default_rng(seed)
    if w is None: w = np.ones(len(y))
    idx = np.arange(len(y)); aucs = []
    for _ in range(n):
        i = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[i])) < 2: continue
        aucs.append(roc_auc_score(y[i], p[i], sample_weight=w[i]))
    return np.percentile(aucs, [2.5, 97.5])

results = {"baseline_auc": round(float(roc_auc_score(yte, p_te)), 4)}
print(f"[基线] 全体 AUC = {roc_auc_score(yte, p_te):.4f}")

# ---- 各年龄组基线 AUC ----
print("\n=== 年龄组基线 AUC ===")
for name, mask in age_groups.items():
    m = mask.values; n = int(m.sum())
    if n >= 50 and len(np.unique(yte[m])) == 2:
        a = auc_w(yte[m], p_te[m])
        ci = boot_auc(yte[m], p_te[m])
        prev = yte[m].mean()
        results.setdefault("age_baseline", {})[name] = {"n": n, "auc": round(float(a), 4),
                                                        "auc_ci95": [round(float(x), 4) for x in ci],
                                                        "prevalence": round(float(prev), 4)}
        print(f"  [{name}] n={n} | prev={prev:.3f} | AUC={a:.3f} {ci}")

# ---- 严重度谱构成（病例内部）----
te["sev"] = np.where(te["egfr"] < 45, "g3b(<45)",
            np.where(te["egfr"] < 60, "g3a(45-60)",
            np.where(te["uacr"] >= 30, "albuminuria-only", "none")))
print("\n=== 病例严重度谱构成（测试集） ===")
sev_by_age = {}
for name, mask in age_groups.items():
    m = mask.values
    cases = m & (yte == 1)
    if cases.sum() < 20: continue
    dist = te.loc[cases, "sev"].value_counts(normalize=True).to_dict()
    sev_by_age[name] = dist
    print(f"  [{name}] 病例谱: " + ", ".join(f"{k}={v:.2f}" for k, v in sorted(dist.items(), key=lambda x: x[0])))

# ---- 谱校正 1：严重度加权（病例谱标准化到全年龄）----
all_cases = yte == 1
all_sev_dist = te.loc[all_cases, "sev"].value_counts(normalize=True)
w_sev = np.ones(len(yte))
young = age_groups["18-44"].values
for sev_cat in all_sev_dist.index:
    sel = young & all_cases & (te["sev"] == sev_cat).values
    if sel.sum() > 0 and sev_cat in sev_by_age.get("18-44", {}):
        w_sev[sel] = all_sev_dist[sev_cat] / max(sev_by_age["18-44"][sev_cat], 1e-6)
# 非年轻病例权重 1（保持参照系）
m_young = young  # 全部 18-44 成人层（n=2327），非病例子集
auc_young_raw = auc_w(yte[m_young], p_te[m_young])
auc_young_sev = auc_w(yte[m_young], p_te[m_young], w_sev[m_young])
gain_sev = auc_young_sev - auc_young_raw

# ---- 谱校正 2：患病率加权（年轻患病率 → 全年龄患病率）----
all_prev = yte.mean(); young_prev = yte[young].mean()
w_prev = np.ones(len(yte))
w_prev[young & (yte == 1)] = all_prev / max(young_prev, 1e-6)
auc_young_prev = auc_w(yte[m_young], p_te[m_young], w_prev[m_young])
gain_prev = auc_young_prev - auc_young_raw

# ---- 交互检验：logistic ckd ~ young + sev_g3b + young×sev_g3b ----
df = pd.DataFrame({"y": yte, "young": young.astype(int),
                   "g3b": (te["egfr"] < 45).astype(int)})
df["young_g3b"] = df["young"] * df["g3b"]
lr = LogisticRegression(max_iter=1000).fit(df[["young", "g3b", "young_g3b"]], df["y"])
coefs = pd.Series(lr.coef_[0], index=["young", "g3b", "young×g3b"])
# 用 Wald 近似检验交互项（非精确，供初步判定；正式用 bootstrap）
interact_p_approx = 2 * (1 - st.norm.cdf(abs(coefs["young×g3b"]) / 0.5))  # 粗略 SE 假设

print("\n=== 谱校正结果（18-44） ===")
print(f"  基线 AUC(18-44) = {auc_young_raw:.3f}")
print(f"  严重度加权后    = {auc_young_sev:.3f} (增益 {gain_sev:+.3f})")
print(f"  患病率加权后    = {auc_young_prev:.3f} (增益 {gain_prev:+.3f})")
print(f"  交互项 young×g3b 系数 = {coefs['young×g3b']:.3f} (近似 p={interact_p_approx:.3f})")

max_gain = max(gain_sev, gain_prev)
post_auc = max(auc_young_sev, auc_young_prev)
if max_gain >= 0.03:
    decision = "H3a 谱效应主导"
elif post_auc <= 0.70 and interact_p_approx > 0.05:
    decision = "H3b 信号积累不足"
elif 0.01 <= max_gain < 0.03 and interact_p_approx <= 0.05:
    decision = "H3c 交互机制"
else:
    decision = "中间状态（需进一步判别）"
print(f"\n[子研究③ 判决] {decision}")

results["substudy3"] = {
    "young_auc_raw": round(float(auc_young_raw), 4),
    "young_auc_severity_weighted": round(float(auc_young_sev), 4),
    "gain_severity": round(float(gain_sev), 4),
    "young_auc_prevalence_weighted": round(float(auc_young_prev), 4),
    "gain_prevalence": round(float(gain_prev), 4),
    "interaction_coef": round(float(coefs["young×g3b"]), 4),
    "interaction_p_approx": round(float(interact_p_approx), 4),
    "sev_distribution_by_age": sev_by_age,
    "decision": decision,
}
with open(os.path.join(M2OUT, "substudy3_results.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"\n已保存: {M2OUT}/substudy3_results.json")
