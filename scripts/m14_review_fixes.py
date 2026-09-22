# -*- coding: utf-8 -*-
"""M14 审稿修复补算：
1) age-only / age+sex / 简单临床规则 检测 AUC（对照 0.810）
2) age-only / age+sex 死亡 Harrell C（对照 0.848/0.879）
3) S1 置换 5000 次（p 值精度修正）
4) S4 符号统一（PIR>=1.3 − PIR<1.3）
"""
import pandas as pd, numpy as np, os, json
from xgboost import XGBClassifier, DMatrix
from sklearn.metrics import roc_auc_score
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score

BASE = r"G:/项目文件/CKD多模态AI早诊_本项目专属归档/09_无湿实验执行"
OUT = os.path.join(BASE, "results/m2")
cohort = pd.read_parquet(os.path.join(BASE, "results/cohort.parquet"))
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
    if "race" in X.columns:
        X["race_black"] = (X["race"] == 4).astype(float)
        X["race_hisp"]  = (X["race"].isin([1, 2])).astype(float)
        X["race_other"] = (~X["race"].isin([3, 4, 1, 2])).astype(float)
        X = X.drop(columns=["race"])
    if "sex" in X.columns:
        X["female"] = (X["sex"] == 2).astype(float)
        X = X.drop(columns=["sex"])
    return X

train = cohort[cohort["year"].isin([2011, 2013, 2015])]
test  = cohort[cohort["year"] == 2017]
ytr, yte = train["ckd"].values, test["ckd"].values
rng = np.random.default_rng(2026)
res = {}

def xgb_fit_predict(Xtr_raw, Xte_raw, y):
    fit = prep_fit(Xtr_raw)
    Xtr, Xte = fit(Xtr_raw), fit(Xte_raw)
    m = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                      n_jobs=4, random_state=42)
    m.fit(Xtr, y)
    return m.predict_proba(Xte)[:, 1]

def boot_auc(y, p, n=2000, seed=9):
    r = np.random.default_rng(seed); a = []
    idx = np.arange(len(y))
    for _ in range(n):
        i = r.choice(idx, len(idx), replace=True)
        if len(np.unique(y[i])) < 2: continue
        a.append(roc_auc_score(y[i], p[i]))
    return [round(float(np.percentile(a, 2.5)), 4), round(float(np.percentile(a, 97.5)), 4)]

# ============ 1) 检测对照基线 ============
print("=== 检测对照基线 ===", flush=True)
sets = {
    "age-only": ["age"],
    "age+sex": ["age", "sex"],
    "age+sex+race": ["age", "sex", "race"],
    "clinical-rule (age+diabetes+SBP)": ["age", "diabetes", "sbp"],
}
for name, feats in sets.items():
    Xtr_raw, Xte_raw = add_dummies(train[feats]), add_dummies(test[feats])
    p = xgb_fit_predict(Xtr_raw, Xte_raw, ytr)
    a = roc_auc_score(yte, p)
    ci = boot_auc(yte, p)
    res.setdefault("detection_baselines", {})[name] = {"auc": round(float(a), 4), "ci95": ci}
    print(f"  {name:<32} AUC={a:.3f} {ci}", flush=True)

# ============ 2) 死亡对照基线 ============
print("=== 死亡对照基线 ===", flush=True)
lmf_dir = os.path.join(BASE, "data/lmf")
def read_lmf(path):
    widths = [(1,6),(15,15),(16,16),(17,19),(20,20),(21,21),(22,22),(23,26),(43,45),(46,48)]
    cols = ["seqn","eligstat","mortstat","ucod_leading","diabetes","hyperten","dodqtr","dodyear","permth_int","permth_exm"]
    recs = []
    for line in open(path):
        if len(line) < 48: continue
        vals = {}
        for (a,b), c in zip(widths, cols):
            s = line[a-1:b].strip()
            vals[c] = float(s) if s not in ("", ".") else np.nan
        recs.append(vals)
    return pd.DataFrame(recs)
parts = []
for cyc, fn in [("2011_2012","NHANES_2011_2012_MORT_2019_PUBLIC.dat"),
                ("2013_2014","NHANES_2013_2014_MORT_2019_PUBLIC.dat"),
                ("2015_2016","NHANES_2015_2016_MORT_2019_PUBLIC.dat"),
                ("2017_2018","NHANES_2017_2018_MORT_2019_PUBLIC.dat")]:
    p = read_lmf(os.path.join(lmf_dir, fn)); p["cycle"] = cyc; parts.append(p)
lmf = pd.concat(parts, ignore_index=True)
df = cohort.merge(lmf[["seqn","mortstat","ucod_leading","permth_int"]], left_on="SEQN", right_on="seqn", how="left")
df["fu"] = df["permth_int"] / 12.0
df["mort"] = df["mortstat"] == 1
df["cvd"] = df["mort"] & (df["ucod_leading"] == 1)
tr = df[df["year"].isin([2011,2013,2015])]; te5 = df[df["year"] == 2017]

def harrell_c(time, event, risk):
    time, event, risk = np.asarray(time,float), np.asarray(event,bool), np.asarray(risk,float)
    comp=conc=ties=0
    for i in np.where(event)[0]:
        later = time > time[i]
        if later.sum()==0: continue
        comp += later.sum(); conc += (risk[i] > risk[later]).sum(); ties += (risk[i] == risk[later]).sum()
    return (conc+0.5*ties)/comp if comp else np.nan

for oname, ev in [("all_cause","mort"), ("cvd","cvd")]:
    yt5 = tr[ev].astype(int).values
    if yt5.sum() < 50: continue
    print(f"  [{oname}] 训练事件={yt5.sum()}", flush=True)
    for name, feats in {"age-only": ["age"], "age+sex": ["age","sex"]}.items():
        fit5 = prep_fit(add_dummies(tr[feats]))
        Xtr5, Xte5 = fit5(add_dummies(tr[feats])), fit5(add_dummies(te5[feats]))
        m5 = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                           subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", n_jobs=4, random_state=42)
        m5.fit(Xtr5, yt5)
        risk = m5.predict_proba(Xte5)[:, 1]
        c = harrell_c(te5["fu"].values, te5[ev].values, risk)
        res.setdefault("mortality_baselines", {}).setdefault(oname, {})[name] = round(float(c), 4)
        print(f"    {name:<12} C={c:.3f}", flush=True)

# ============ 3) 置换 5000 次 ============
print("=== S1 置换 5000 次 ===", flush=True)
fit_full = prep_fit(add_dummies(train[FEATURES]))
Xtr_f, Xte_f = fit_full(add_dummies(train[FEATURES])), fit_full(add_dummies(test[FEATURES]))
model = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                      n_jobs=4, random_state=42)
model.fit(Xtr_f, ytr)
shap = np.asarray(model.get_booster().predict(DMatrix(Xte_f), pred_contribs=True))[:, :-1]
feat_names = list(Xte_f.columns)
PATHWAY = {"hemodynamic": ["sbp","dbp"], "glycotoxic": ["hba1c","diabetes"],
           "metabolic": ["total_cholesterol","hdl"],
           "demographic": ["age","female","race_black","race_hisp","race_other",
                           "poverty_ratio","education","bmi"]}
miss = [f for f in feat_names if f.startswith("miss_")]
PATHWAY["missing_indicator"] = miss
ftop = {f: p for p, fs in PATHWAY.items() for f in fs}
ref = np.array([list(PATHWAY).index(ftop[f]) for f in feat_names])
corr = np.corrcoef(shap.T); corr = np.nan_to_num(corr, nan=0.0)
d = 1 - np.abs(corr); np.fill_diagonal(d, 0); d = (d + d.T) / 2; d = np.clip(d, 0, None)
Z = linkage(squareform(d), method="ward")
lab = fcluster(Z, len(PATHWAY), criterion="maxclust")
ari = adjusted_rand_score(ref, lab)
nulls = []
for _ in range(5000):
    nulls.append(adjusted_rand_score(rng.permutation(ref), lab))
nulls = np.array(nulls)
p5000 = (nulls >= ari).mean()
res["s1"] = {"ari": round(float(ari), 4), "permutation_p_5000": round(float(p5000), 4),
             "p_report": f"p<{1/5000:.4f}" if p5000 == 0 else f"p={p5000:.4f}"}
print(f"  ARI={ari:.3f} | 5000 次置换 p={p5000:.4f}（最小可达 0.0002）", flush=True)

# ============ 4) S4 符号统一 ============
lo, hi = (test["poverty_ratio"] < 1.3).values, (test["poverty_ratio"] >= 1.3).values
p_te = model.predict_proba(Xte_f)[:, 1]
d4 = roc_auc_score(yte[hi], p_te[hi]) - roc_auc_score(yte[lo], p_te[lo])
res["s4_unified_sign"] = {"delta_auc_high_minus_low": round(float(d4), 4)}
print(f"  S4 统一符号（PIR≥1.3 − PIR<1.3）: ΔAUC={d4:+.3f}", flush=True)

with open(os.path.join(OUT, "review_fixes.json"), "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=2)
print(f"\n已保存: review_fixes.json")
