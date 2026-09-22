"""M7 Figure 2 —— 分层 AUC 森林图（子研究②③④ 整合证据视图）
行：全体（参考）/ uACR 盲区层 / 糖尿病盲区层 / 非糖尿病盲区层 / 年轻 18-44 / 中年 45-64 / 老年 ≥65 / PIR<1.3 / PIR≥1.3
参考线：0.75（临床可用阈值）+ 全体 AUC 0.810（虚线）
"""
import pandas as pd, numpy as np, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

OUT = r"G:/项目文件/CKD多模态AI早诊_本项目专属归档/09_无湿实验执行/results/m2"
cohort = pd.read_parquet(r"G:/项目文件/CKD多模态AI早诊_本项目专属归档/09_无湿实验执行/results/cohort.parquet")
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

def boot_auc(y, p, n=2000, seed=7):
    rng = np.random.default_rng(seed); aucs = []
    idx = np.arange(len(y))
    for _ in range(n):
        i = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[i])) < 2: continue
        aucs.append(roc_auc_score(y[i], p[i]))
    return np.percentile(aucs, [2.5, 97.5])

layers = [
    ("全体人群（参考）", te["ckd"].notna(), "#333333"),
    ("uACR<30 盲区人群", te["uacr"] < 30, "#D62728"),
    ("  盲区 + 糖尿病（NADKD 风险）", (te["uacr"] < 30) & (te["diabetes"] == 1), "#E8890C"),
    ("  盲区 + 非糖尿病", (te["uacr"] < 30) & (te["diabetes"] == 0), "#F5B942"),
    ("年龄 18–44", te["age"] < 45, "#1F77B4"),
    ("年龄 45–64", (te["age"] >= 45) & (te["age"] < 65), "#4C9BD9"),
    ("年龄 ≥65", te["age"] >= 65, "#A6CBE3"),
    ("PIR < 1.3（低社会经济）", te["poverty_ratio"] < 1.3, "#2CA6A4"),
    ("PIR ≥ 1.3", te["poverty_ratio"] >= 1.3, "#7FD1CF"),
]
rows = []
for name, mask, c in layers:
    m = mask.values
    if m.sum() < 50 or len(np.unique(yte[m])) < 2:
        continue
    a = roc_auc_score(yte[m], p_te[m])
    lo, hi = boot_auc(yte[m], p_te[m])
    rows.append({"name": name, "auc": a, "ci_lo": lo, "ci_hi": hi,
                 "n": int(m.sum()), "prev": float(yte[m].mean()), "color": c})
    print(f"{name:<28} n={m.sum():>5} prev={yte[m].mean():.3f} AUC={a:.3f} [{lo:.3f},{hi:.3f}]")

# ---- 森林图 ----
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.7})
fig, ax = plt.subplots(figsize=(6.6, 4.2))
ypos = np.arange(len(rows))[::-1]
for yi, r in zip(ypos, rows):
    ax.plot([r["ci_lo"], r["ci_hi"]], [yi, yi], color=r["color"], lw=1.6, zorder=2)
    ax.plot(r["auc"], yi, "o", color=r["color"], ms=7, mec="white", mew=0.8, zorder=3)
ax.axvline(0.75, color="#888888", ls="--", lw=0.9, zorder=1)
ax.axvline(rows[0]["auc"], color="#BBBBBB", ls=":", lw=0.9, zorder=1)
ax.text(0.75, len(rows) - 0.25, "clinical threshold 0.75", fontsize=6.5, color="#666666", ha="center")
ax.text(rows[0]["auc"], len(rows) - 0.25, f"overall {rows[0]['auc']:.3f}", fontsize=6.5, color="#999999", ha="center")
ax.set_yticks(ypos); ax.set_yticklabels([r["name"] for r in rows], fontsize=8)
ax.set_xlabel("AUC (95% CI, 2000× bootstrap)", fontsize=8.5)
ax.set_xlim(0.55, 0.95)
ax.set_title("Stratified CKD detection performance — NHANES 2017–18 external validation", loc="left", fontsize=9, fontweight="bold")
for yi, r in zip(ypos, rows):
    ax.annotate(f"n={r['n']:,}", xy=(0.555, yi), fontsize=6.5, color="#555555", va="center")
ax.grid(axis="x", color="#EEEEEE", lw=0.5)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "figure2_forest.png"), dpi=300, bbox_inches="tight")
fig.savefig(os.path.join(OUT, "figure2_forest.pdf"), bbox_inches="tight")
print("已保存: figure2_forest.png / .pdf")
