"""MACKI-Dry 补强分析 v2（审稿修复版）
新增（回应红队 P0-1/P0-3）：
  S1 年龄-only 基线：仅 age 特征的 XGB → 量化"总体 AUC 有多少来自年龄组间分层"
  S2 临床规则基线：LR 仅用 age+diabetes+sbp → 与"复杂 ML"公平对比
  S3 年龄组内性能：18-44/45-64/65+ 内部分层（年龄主导质疑的直接证据）
保留：NADKD、周期患病率、图（ROC/校准/DCA/亚组）
"""
import pandas as pd, numpy as np, json, os
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = r"G:/项目文件/CKD多模态AI早诊_本项目专属归档/09_无湿实验执行/results"
test = pd.read_parquet(os.path.join(OUT, "test_set.parquet"))
y = np.load(os.path.join(OUT, "test_y.npy"))
p_lr = np.load(os.path.join(OUT, "test_proba_lr.npy"))
p_xgb = np.load(os.path.join(OUT, "test_proba_xgb.npy"))
p_mlp = np.load(os.path.join(OUT, "test_proba_mlp.npy"))
cohort = pd.read_parquet(os.path.join(OUT, "cohort.parquet"))

# --- S1: 年龄-only 基线（XGB，仅 age）---
train = cohort[cohort["year"].isin([2011, 2013, 2015])]
Xtr_age = train[["age"]].copy()
Xte_age = test[["age"]].copy()
age_only = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                         n_jobs=4, random_state=42)
age_only.fit(Xtr_age, train["ckd"].values)
p_age = age_only.predict_proba(Xte_age)[:, 1]
auc_age = roc_auc_score(y, p_age)
print(f"[S1] Age-only XGB AUC = {auc_age:.4f} (vs full XGB {np.load(OUT+'/../results/test_proba_xgb.npy') is not None})")

# --- S2: 临床规则基线（LR：age + diabetes + sbp）---
FEAT_RULE = ["age", "diabetes", "sbp"]
tr = train[FEAT_RULE].fillna(train[FEAT_RULE].median())
te = test[FEAT_RULE].fillna(train[FEAT_RULE].median())
rule = LogisticRegression(max_iter=2000, C=0.1)
rule.fit(tr, train["ckd"].values)
p_rule = rule.predict_proba(te)[:, 1]
auc_rule = roc_auc_score(y, p_rule)
print(f"[S2] Clinical-rule LR (age+DM+SBP) AUC = {auc_rule:.4f}")

# --- S3: 年龄组内性能（全特征 XGB 在年龄组内的 AUC —— 回应"年龄主导"质疑）---
full_xgb = None
rng = np.random.default_rng(2026)
te_df = test.reset_index(drop=True)
def within_age_auc(p, y, age):
    out = {}
    for label, mask in [("18_44", age < 45), ("45_64", (age >= 45) & (age < 65)), ("65plus", age >= 65)]:
        m = mask.values if hasattr(mask, "values") else mask
        if m.sum() >= 50 and len(np.unique(y[m])) == 2:
            out[label] = {"n": int(m.sum()), "auc": round(float(roc_auc_score(y[m], p[m])), 4)}
    return out
age_in = within_age_auc(p_xgb, y, te_df["age"])
age_in_rule = within_age_auc(p_rule, y, te_df["age"])
print(f"[S3] Within-age-group AUC (full XGB): {age_in}")
print(f"[S3] Within-age-group AUC (rule LR): {age_in_rule}")

# --- NADKD（测试集）---
nadkd = (te_df["diabetes"] == 1) & (te_df["egfr"] < 60) & (te_df["uacr"] < 30)
n_diab = int((te_df["diabetes"] == 1).sum())
print(f"[NADKD] n={int(nadkd.sum())} | 占糖尿病者 {nadkd.sum()/max(n_diab,1):.1%} (n_diab={n_diab}) | 占 CKD {nadkd.sum()/max(int(y.sum()),1):.1%}")

# --- 周期加权患病率 ---
prev_by_cycle = {}
for c in ["G", "H", "I", "J"]:
    sub_c = cohort[cohort["cycle"] == c]
    w = sub_c["wt"].fillna(0)
    prev_by_cycle[c] = round(float((sub_c.loc[sub_c["ckd"] == 1, "wt"].fillna(0).sum() / w.sum()) * 100), 2)
print(f"[PREV] cycle-weighted CKD: {prev_by_cycle}")

# --- 图 1: ROC（三模型 + 年龄only + 规则基线）---
fig, ax = plt.subplots(1, 1, figsize=(5.5, 5))
for name, p, c in [("XGBoost (full)", p_xgb, "#d62728"), ("Logistic regression", p_lr, "#1f77b4"),
                   ("MLP", p_mlp, "#2ca02c"), ("Clinical rule (age+DM+SBP)", p_rule, "#ff7f0e"),
                   ("Age-only", p_age, "#9467bd")]:
    fpr, tpr, _ = roc_curve(y, p)
    ax.plot(fpr, tpr, c=c, lw=2, label=f"{name} (AUC={roc_auc_score(y, p):.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
ax.set(xlabel="1 − Specificity", ylabel="Sensitivity", title="CKD detection — external temporal validation (NHANES 2017–2018)")
ax.legend(loc="lower right", fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_roc.png"), dpi=300); plt.close()

# --- 图 2: 校准（XGB/LR/规则）---
fig, ax = plt.subplots(1, 1, figsize=(5.5, 5))
for name, p, c in [("XGBoost", p_xgb, "#d62728"), ("Logistic regression", p_lr, "#1f77b4"), ("Clinical rule", p_rule, "#ff7f0e")]:
    df = pd.DataFrame({"y": y, "p": p})
    df["bin"] = pd.qcut(df["p"], 10, duplicates="drop")
    g = df.groupby("bin", observed=True).agg(mp=("p", "mean"), my=("y", "mean"))
    ax.plot(g["mp"], g["my"], "o-", c=c, lw=2, label=name)
ax.plot([0, 0.8], [0, 0.8], "k--", lw=1)
ax.set(xlabel="Predicted probability", ylabel="Observed proportion", title="Calibration (external validation)")
ax.legend(fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_calibration.png"), dpi=300); plt.close()

# --- 图 3: DCA ---
def nb(y, p, pts):
    out = []
    for pt in pts:
        tp = ((p >= pt) & (y == 1)).sum() / len(y)
        fp = ((p >= pt) & (y == 0)).sum() / len(y)
        out.append(tp - fp * pt / (1 - pt))
    return np.array(out)
pts = np.linspace(0.02, 0.30, 30)
prev = y.mean()
fig, ax = plt.subplots(1, 1, figsize=(5.5, 5))
ax.plot(pts, nb(y, p_xgb, pts), "-", c="#d62728", lw=2, label="XGBoost (full)")
ax.plot(pts, nb(y, p_lr, pts), "--", c="#1f77b4", lw=2, label="Logistic regression")
ax.plot(pts, nb(y, p_rule, pts), ":", c="#ff7f0e", lw=2, label="Clinical rule")
ax.plot(pts, np.maximum(0, prev - pts / (1 - pts) * (1 - prev)), "-.", c="gray", lw=1.5, label="Screen all")
ax.plot(pts, np.zeros_like(pts), "-", c="black", lw=1, label="Screen none")
ax.set(xlabel="Threshold probability", ylabel="Net benefit", title="Decision curve analysis (external validation)")
ax.legend(fontsize=8); ax.set_xlim(0, 0.3)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_dca.png"), dpi=300); plt.close()

# --- 图 4: 亚组森林图（全特征 XGB）---
mr = json.load(open(os.path.join(OUT, "model_results.json"), encoding="utf-8"))
sub = mr["XGB"]["subgroups"]
labels = {"age_18_44": "Age 18–44", "age_45_64": "Age 45–64", "age_65plus": "Age 65+",
          "sex_M": "Male", "sex_F": "Female", "race_white": "White", "race_black": "Black",
          "race_hisp": "Hispanic", "pov_lt1.3": "PIR <1.3", "pov_ge1.3": "PIR ≥1.3"}
order = ["age_18_44", "age_45_64", "age_65plus", "sex_M", "sex_F", "race_white", "race_black", "race_hisp", "pov_lt1.3", "pov_ge1.3"]
fig, ax = plt.subplots(figsize=(6, 5))
aucs = [sub[k]["auc"] for k in order]
ax.barh(range(len(order)), aucs, color="#1f77b4", alpha=0.8)
ax.set_yticks(range(len(order))); ax.set_yticklabels([labels[k] for k in order], fontsize=10)
ax.axvline(mr["XGB"]["auc_test"], color="red", ls="--", lw=1, label=f"Overall AUC {mr['XGB']['auc_test']}")
ax.set_xlabel("AUC (external validation)"); ax.set_xlim(0.5, 0.9)
ax.legend(fontsize=9); ax.invert_yaxis()
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_subgroups.png"), dpi=300); plt.close()

# --- S4: 亚组 bootstrap CI（全特征 XGB）---
rng_ci = np.random.default_rng(42)
def subgroup_auc_ci(p, y, masks):
    out = {}
    for k, m in masks.items():
        m = m.values if hasattr(m, "values") else m
        if m.sum() < 50 or len(np.unique(y[m])) < 2: continue
        aucs = []
        idx = np.arange(len(y))
        for _ in range(1000):
            i = rng_ci.choice(idx[m], len(idx[m]), replace=True)
            if len(np.unique(y[i])) < 2: continue
            aucs.append(roc_auc_score(y[i], p[i]))
        out[k] = {"auc": round(float(roc_auc_score(y[m], p[m])), 4),
                  "ci95": [round(float(np.percentile(aucs, 2.5)), 4), round(float(np.percentile(aucs, 97.5)), 4)]}
    return out
masks_ci = {"age_18_44": te_df["age"] < 45, "age_45_64": (te_df["age"] >= 45) & (te_df["age"] < 65),
            "age_65plus": te_df["age"] >= 65, "sex_M": te_df["sex"] == 1, "sex_F": te_df["sex"] == 2,
            "race_white": te_df["race"] == 3, "race_black": te_df["race"] == 4,
            "race_hisp": te_df["race"].isin([1, 2]), "pov_lt1.3": te_df["poverty_ratio"] < 1.3,
            "pov_ge1.3": te_df["poverty_ratio"] >= 1.3}
subgroup_ci = subgroup_auc_ci(p_xgb, y, masks_ci)

# --- S5: 4 周期总体加权患病率 ---
w_all = cohort["wt"].fillna(0)
prev_overall = round(float((cohort.loc[cohort["ckd"] == 1, "wt"].fillna(0).sum() / w_all.sum()) * 100), 2)

summary = {
    "age_only_xgb_auc": round(float(auc_age), 4),
    "clinical_rule_lr_auc": round(float(auc_rule), 4),
    "within_age_group_full_xgb": age_in,
    "within_age_group_rule_lr": age_in_rule,
    "nadkd_test": {"n": int(nadkd.sum()), "n_diabetes": n_diab,
                   "pct_of_diabetes": round(float(nadkd.sum() / max(n_diab, 1) * 100), 1),
                   "pct_of_ckd": round(float(nadkd.sum() / max(int(y.sum()), 1) * 100), 1)},
    "cycle_weighted_ckd_prevalence": prev_by_cycle,
    "overall_weighted_ckd_prevalence_4cycle": prev_overall,
    "subgroup_auc_ci_xgb": subgroup_ci,
}
with open(os.path.join(OUT, "supplementary_results.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print("\nsaved: supplementary_results.json (v2) + 4 figures")
