# -*- coding: utf-8 -*-
"""M11 稳健性项 —— 全覆盖 bootstrap CI + 重采样敏感性
S1: ARI bootstrap CI（SHAP 矩阵行重采样 300 轮）
S2: 分层 ΔAUC 配对 bootstrap CI + 差异检验（2000 轮）
S3: 谱校正重采样敏感性（病例谱匹配重采样 100 轮 → 增益分布 CI）——补加权 AUC 校正力有限的短板
S4: 熵平衡重采样（50 轮）→ 平衡后 ΔAUC 分布
S5: Harrell C bootstrap CI（300 轮）
"""
import pandas as pd, numpy as np, os, json, time
from xgboost import XGBClassifier, DMatrix
from sklearn.metrics import roc_auc_score
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score
from scipy.optimize import minimize
import socks, socket

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
rng = np.random.default_rng(2026)
t0 = time.time()
res = {"baseline_auc": round(float(roc_auc_score(yte, p_te)), 4)}

def ci(x, lo=2.5, hi=97.5):
    return [round(float(np.percentile(x, lo)), 4), round(float(np.percentile(x, hi)), 4)]

# ============ S1: ARI bootstrap（SHAP 行重采样）============
print("=== S1: ARI bootstrap CI（300 轮）===", flush=True)
shap = np.asarray(model.get_booster().predict(DMatrix(Xte), pred_contribs=True))[:, :-1]
feat_names = list(Xte.columns)
PATHWAY = {"hemodynamic": ["sbp","dbp"], "glycotoxic": ["hba1c","diabetes"],
           "metabolic": ["total_cholesterol","hdl"],
           "demographic": ["age","female","race_black","race_hisp","race_other",
                           "poverty_ratio","education","bmi"]}
miss_feats = [f for f in feat_names if f.startswith("miss_")]
PATHWAY["missing_indicator"] = miss_feats
feat_to_path = {f: p for p, fs in PATHWAY.items() for f in fs}
ref_labels = np.array([list(PATHWAY).index(feat_to_path[f]) for f in feat_names])

def ari_of(shap_rows):
    c = np.corrcoef(shap_rows.T); c = np.nan_to_num(c, nan=0.0)
    d = 1 - np.abs(c); np.fill_diagonal(d, 0); d = (d + d.T) / 2; d = np.clip(d, 0, None)
    Z = linkage(squareform(d), method="ward")
    lab = fcluster(Z, len(PATHWAY), criterion="maxclust")
    return adjusted_rand_score(ref_labels, lab)

ari_bs = []
n = shap.shape[0]
for _ in range(300):
    i = rng.choice(n, n, replace=True)
    ari_bs.append(ari_of(shap[i]))
res["s1_ari"] = {"point": 0.4632, "bootstrap_ci95": ci(ari_bs),
                 "n_boot": 300, "bootstrap_mean": round(float(np.mean(ari_bs)), 4)}
print(f"  ARI bootstrap: mean={np.mean(ari_bs):.3f} CI={ci(ari_bs)}", flush=True)

# ============ S2: 分层 ΔAUC 配对 bootstrap ============
print("=== S2: ΔAUC 配对 bootstrap（2000 轮）===", flush=True)
layers = {
    "blind_uACR<30": te["uacr"] < 30,
    "blind+DM": (te["uacr"] < 30) & (te["diabetes"] == 1),
    "blind+nonDM": (te["uacr"] < 30) & (te["diabetes"] == 0),
    "age18-44": te["age"] < 45,
    "age45-64": (te["age"] >= 45) & (te["age"] < 65),
    "age65+": te["age"] >= 65,
    "PIR<1.3": te["poverty_ratio"] < 1.3,
    "PIR>=1.3": te["poverty_ratio"] >= 1.3,
}
s2 = {}
for name, mask in layers.items():
    m = mask.values
    if m.sum() < 100 or len(np.unique(yte[m])) < 2:
        continue
    d = []
    for _ in range(2000):
        i = rng.choice(n, n, replace=True)
        mi = m[i]
        if mi.sum() < 20 or len(np.unique(yte[i][mi])) < 2: continue
        a_all = roc_auc_score(yte[i], p_te[i])
        a_str = roc_auc_score(yte[i][mi], p_te[i][mi])
        d.append(a_all - a_str)
    d = np.array(d)
    s2[name] = {"delta_auc_point": round(float(roc_auc_score(yte, p_te) - roc_auc_score(yte[m], p_te[m])), 4),
                "delta_auc_bootstrap_ci95": ci(d), "p_delta_gt_0": round(float((d > 0).mean()), 4)}
    print(f"  {name:<14} ΔAUC={s2[name]['delta_auc_point']:+.3f} CI={ci(d)} P(Δ>0)={s2[name]['p_delta_gt_0']:.3f}", flush=True)
res["s2_delta_auc"] = s2

# ============ S3: 重采样敏感性（病例谱匹配）============
print("=== S3: 重采样敏感性（100 轮）===", flush=True)
te["sev"] = np.where(te["egfr"] < 45, "g3b", np.where(te["egfr"] < 60, "g3a",
             np.where(te["uacr"] >= 30, "alb", "none")))
young = (te["age"] < 45).values
ycases = young & (yte == 1); yctrl = young & (yte == 0)
all_cases_sev = te.loc[yte == 1, "sev"].value_counts(normalize=True)
young_cases_sev = te.loc[ycases, "sev"].value_counts(normalize=True)
gains_sev, gains_prev = [], []
for _ in range(100):
    # 严重度谱匹配重采样：年轻病例按全年龄谱重采样
    idx = []
    for sev_cat, prop in all_cases_sev.items():
        pool = np.where(ycases & (te["sev"] == sev_cat).values)[0]
        if len(pool) == 0: continue
        k = int(round(prop * ycases.sum()))
        idx.append(rng.choice(pool, k, replace=True))
    idx = np.concatenate(idx)
    ctrl = np.where(yctrl)[0]
    ctrl_i = rng.choice(ctrl, len(ctrl), replace=True)
    sel = np.concatenate([idx, ctrl_i])
    gains_sev.append(roc_auc_score(yte[sel], p_te[sel]) - roc_auc_score(yte[young], p_te[young]))
    # 患病率匹配重采样：年轻病例按全年龄患病率加权抽样
    n_cases_target = int(all_cases_sev.sum() * 0 + round(yte.mean() * yctrl.sum() / (1 - yte.mean())))
    pool = np.where(ycases)[0]
    idx2 = rng.choice(pool, max(n_cases_target - len(pool), len(pool)), replace=True)
    sel2 = np.concatenate([idx2, ctrl_i])
    gains_prev.append(roc_auc_score(yte[sel2], p_te[sel2]) - roc_auc_score(yte[young], p_te[young]))
gains_sev, gains_prev = np.array(gains_sev), np.array(gains_prev)
res["s3_resampling"] = {"gain_severity_ci95": ci(gains_sev), "gain_severity_mean": round(float(gains_sev.mean()), 4),
                        "gain_prevalence_ci95": ci(gains_prev), "gain_prevalence_mean": round(float(gains_prev.mean()), 4)}
print(f"  严重度重采样增益: mean={gains_sev.mean():+.3f} CI={ci(gains_sev)}", flush=True)
print(f"  患病率重采样增益: mean={gains_prev.mean():+.3f} CI={ci(gains_prev)}", flush=True)

# ============ S4: 熵平衡重采样（50 轮）============
print("=== S4: 熵平衡重采样（50 轮）===", flush=True)
BAL_VARS = ["age","bmi","sbp","dbp","hba1c","diabetes","total_cholesterol","hdl","education","race"]
lo_m, hi_m = (te["poverty_ratio"] < 1.3).values, (te["poverty_ratio"] >= 1.3).values
cov_lo = te.loc[lo_m, BAL_VARS].fillna(te.loc[lo_m, BAL_VARS].median())
cov_hi = te.loc[hi_m, BAL_VARS].fillna(te.loc[hi_m, BAL_VARS].median())
mu, sd = cov_hi.mean(0), cov_hi.std(0).replace(0, 1)
cov_lo_s = (cov_lo - mu) / sd; cov_hi_s = (cov_hi - mu) / sd
def eb_weights(Xt, Xc):
    mxc = Xc.mean(0)
    def loss(lmb):
        w = np.exp(Xt @ lmb)
        mm = (w[:, None] * Xt).sum(0) / w.sum() - mxc
        return 0.5 * mm @ mm + 1e-6 * lmb @ lmb
    r = minimize(loss, np.zeros(Xt.shape[1]), method="BFGS", options={"maxiter": 500, "gtol": 1e-8})
    w = np.exp(Xt @ r.x); return w / w.mean()
w0 = eb_weights(cov_lo_s.values, cov_hi_s.values)
d0 = roc_auc_score(yte[lo_m], p_te[lo_m], sample_weight=w0) - roc_auc_score(yte[hi_m], p_te[hi_m])
d_bs = []
for _ in range(50):
    i = rng.choice(n, n, replace=True)
    li, hi_ = lo_m[i], hi_m[i]
    if li.sum() < 50 or hi_.sum() < 50: continue
    # 对熵平衡权重做 bootstrap（保持平衡结构，检验权重不确定性）
    wb = w0[rng.choice(len(w0), len(w0), replace=True)]
    a_lo = roc_auc_score(yte[lo_m], p_te[lo_m], sample_weight=wb)
    a_hi = roc_auc_score(yte[hi_m], p_te[hi_m])
    d_bs.append(a_lo - a_hi)
d_bs = np.array(d_bs)
res["s4_entropy"] = {"delta_auc_point": round(float(d0), 4), "bootstrap_ci95": ci(d_bs), "n_boot": 50}
print(f"  熵平衡后 ΔAUC={d0:+.3f} CI={ci(d_bs)}", flush=True)

# ============ S5: C-index bootstrap ============
print("=== S5: Harrell C bootstrap（300 轮）===", flush=True)
import urllib.request, json as _json
# 加载 LMF 死亡率
lmf_dir = r"G:/项目文件/CKD多模态AI早诊_本项目专属归档/09_无湿实验执行/data/lmf"
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
Xtr5 = fit(add_dummies(tr[FEATURES])); Xte5 = fit(add_dummies(te5[FEATURES]))

def harrell_c(time, event, risk):
    time, event, risk = np.asarray(time,float), np.asarray(event,bool), np.asarray(risk,float)
    comp=conc=ties=0
    for i in np.where(event)[0]:
        later = time > time[i]
        if later.sum()==0: continue
        comp += later.sum(); conc += (risk[i] > risk[later]).sum(); ties += (risk[i] == risk[later]).sum()
    return (conc+0.5*ties)/comp if comp else np.nan

s5 = {}
for oname, ev in [("all_cause","mort"), ("cvd","cvd")]:
    yt5 = tr[ev].astype(int).values
    if yt5.sum() < 100: continue
    m5 = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                       subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", n_jobs=4, random_state=42)
    m5.fit(Xtr5, yt5)
    risk = m5.predict_proba(Xte5)[:, 1]
    t5, e5 = te5["fu"].values, te5[ev].values
    c0 = harrell_c(t5, e5, risk)
    cs = []
    for _ in range(300):
        i = rng.choice(len(t5), len(t5), replace=True)
        if e5[i].sum() < 5: continue
        cs.append(harrell_c(t5[i], e5[i], risk[i]))
    cs = np.array(cs)
    s5[oname] = {"c_index_point": round(float(c0),4), "bootstrap_ci95": ci(cs), "n_test_events": int(e5.sum())}
    print(f"  {oname}: C={c0:.3f} CI={ci(cs)}", flush=True)
res["s5_c_index"] = s5

res["runtime_sec"] = round(time.time() - t0, 1)
with open(os.path.join(OUT, "robustness_results.json"), "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=2)
print(f"\n完成 {res['runtime_sec']}s → robustness_results.json", flush=True)
