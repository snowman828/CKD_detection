# -*- coding: utf-8 -*-
"""阶段 A · Wave 2 · 统计批：P1-01 校准CI / P1-02 NRI-IDI / P1-03 临床阈值 / P1-04 加权+设计型CI /
P1-05 重校准实证 / P1-06 FDR / P1-09 1233画像 / P1-10 加权校准与DCA / P1-11 亚组PPV / P1-12 分组阈值 / P1-17 交叉性
自检：XGB 0.8099 / LR 0.7943 / age-only 0.7393 / 临床规则 0.7708（±1e-4）否则中止
输出：results/p1_statistics.json, results/p1_clinical_equity.json
"""
import json, os, sys, time
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

t0 = time.time()
RES = r"G:/项目文件/CKD多模态AI早诊_本项目专属归档/09_无湿实验执行/results"
DATA = r"G:/项目文件/CKD多模态AI早诊_本项目专属归档/09_无湿实验执行/data"
FEATURES = ["age", "sex", "race", "poverty_ratio", "education", "bmi", "sbp", "dbp", "hba1c", "diabetes", "total_cholesterol", "hdl"]
RNG = 2026
NBOOT = 5000

cohort = pd.read_parquet(os.path.join(RES, "cohort.parquet"))
train = cohort[cohort["year"].isin([2011, 2013, 2015])].copy()
test = cohort[cohort["year"] == 2017].copy().reset_index(drop=True)

def enc(X):
    X = X.copy()
    X["race_black"] = (X["race"] == 4).astype(float); X["race_hisp"] = (X["race"].isin([1, 2])).astype(float)
    X["race_other"] = (~X["race"].isin([3, 4, 1, 2])).astype(float); X["female"] = (X["sex"] == 2).astype(float)
    return X.drop(columns=["race", "sex"])

med = enc(train[FEATURES]).median()
f = lambda D: pd.concat([D.fillna(med), D.isna().add_prefix("miss_")], axis=1)
Xtr, Xte = f(enc(train[FEATURES])), f(enc(test[FEATURES]))
ytr, yte = train["ckd"].values, test["ckd"].values
XGBP = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", n_jobs=4, random_state=42)
xgb = XGBClassifier(**XGBP).fit(Xtr, ytr); lr = LogisticRegression(max_iter=2000, C=0.1).fit(Xtr, ytr)
age_only = XGBClassifier(**XGBP).fit(enc(train[FEATURES])[["age"]], ytr)
rc = ["age", "diabetes", "sbp"]
rule = LogisticRegression(max_iter=2000, C=0.1).fit(Xtr[rc], ytr)
p_x, p_l = xgb.predict_proba(Xte)[:, 1], lr.predict_proba(Xte)[:, 1]
p_a = age_only.predict_proba(enc(test[FEATURES])[["age"]])[:, 1]
p_r = rule.predict_proba(Xte[rc])[:, 1]
sup = json.load(open(os.path.join(RES, "supplementary_results.json"), encoding="utf-8"))
chk = {"XGB": (roc_auc_score(yte, p_x), 0.8099), "LR": (roc_auc_score(yte, p_l), 0.7943),
       "age_only": (roc_auc_score(yte, p_a), sup["age_only_xgb_auc"]), "rule": (roc_auc_score(yte, p_r), sup["clinical_rule_lr_auc"])}
print("自检:", {k: f"{v[0]:.4f} vs {v[1]}" for k, v in chk.items()}, flush=True)
if any(abs(a - b) > 1e-4 for a, b in chk.values()):
    sys.exit("❌ 自检未通过")
YTHR = 0.1602  # SI S2 开发集 Youden 阈值

# ---------- 抽样设计变量（DEMO 原始文件） ----------
design = {}
for fn, yr in (("DEMO_G.XPT", 2011), ("DEMO_H.XPT", 2013), ("DEMO_I.XPT", 2015), ("DEMO_J.XPT", 2017)):
    p = os.path.join(DATA, fn)
    if not os.path.exists(p):
        print("  ⚠️ 缺", fn); continue
    try:
        d = pd.read_sas(p, format="xport")
    except Exception as e:
        print("  ⚠️ 读取失败", fn, e); continue
    cols = [c for c in ("SEQN", "SDMVPSU", "SDMVSTRA", "WTMEC2YR", "WTINT2YR") if c in d.columns]
    d = d[cols].copy(); d["year"] = yr
    design[yr] = d
if design:
    D = pd.concat(design.values(), ignore_index=True)
    test = test.merge(D, on=["SEQN", "year"], how="left", suffixes=("", "_demo"))
    HAVE_DESIGN = "SDMVPSU" in test.columns and test["SDMVPSU"].notna().any()
else:
    HAVE_DESIGN = False
print("  抽样设计变量可用:", HAVE_DESIGN, flush=True)

def w_auc(y, p, w):
    o = np.argsort(p)
    y, w = y[o], w[o]
    tw = np.cumsum(w * y); fw = np.cumsum(w * (1 - y))
    tp = np.concatenate([[0], tw / max(tw[-1], 1e-12)])
    fp = np.concatenate([[0], fw / max(fw[-1], 1e-12)])
    # 升序排序得到的是 ROC 的镜像 -> 取 1-面积（等价于降序阈值扫描）
    return float(1.0 - np.trapezoid(tp, fp))

def dump1():
    json.dump({"note": "P1 statistical addendum (expert-panel items A-phase)", "results": out1},
              open(os.path.join(RES, "p1_statistics.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)

def calib(y, p, w=None, nd=10):
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    q = np.quantile(p, np.linspace(0, 1, nd + 1)); q[0], q[-1] = -np.inf, np.inf
    mp, ob, ww = [], [], []
    for i in range(nd):
        m = (p > q[i]) & (p <= q[i + 1])
        if m.sum() == 0: continue
        mp.append(np.average(p[m], weights=w[m])); ob.append(np.average(y[m], weights=w[m])); ww.append(w[m].sum())
    mp, ob, ww = np.array(mp), np.array(ob), np.array(ww)
    A = np.vstack([mp, np.ones_like(mp)]).T
    W = np.diag(ww)
    coef = np.linalg.lstsq(A.T @ W @ A, A.T @ W @ ob, rcond=None)[0]
    ece = float(np.sum(ww * np.abs(mp - ob)) / np.sum(ww))
    return float(coef[0]), float(coef[1]), ece

def boot_ci(fn, n=NBOOT, seed=RNG):
    rng = np.random.default_rng(seed); idx = np.arange(len(yte)); out = []
    for _ in range(n):
        i = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(yte[i])) < 2: continue
        try: out.append(fn(i))
        except Exception: continue
    a = np.array(out)
    return [round(float(np.percentile(a, 2.5)), 4), round(float(np.percentile(a, 97.5)), 4)]

out1, out2 = {}, {}
print("P1-01 校准 CI ...", flush=True)
sl, ic, ec = calib(yte, p_x)
out1["calibration_xgb"] = {"slope": round(sl, 3), "intercept": round(ic, 4), "ece": round(ec, 4),
    "slope_ci": boot_ci(lambda i: calib(yte[i], p_x[i])[0]), "intercept_ci": boot_ci(lambda i: calib(yte[i], p_x[i])[1]),
    "ece_ci": boot_ci(lambda i: calib(yte[i], p_x[i])[2])}
sl2, ic2, ec2 = calib(yte, p_l)
out1["calibration_lr"] = {"slope": round(sl2, 3), "intercept": round(ic2, 4), "ece": round(ec2, 4),
    "slope_ci": boot_ci(lambda i: calib(yte[i], p_l[i])[0]), "ece_ci": boot_ci(lambda i: calib(yte[i], p_l[i])[2])}
print("   ", out1["calibration_xgb"], flush=True)
dump1()

print("P1-02 NRI/IDI ...", flush=True)
def nri_idi(y, pn, po):
    up = (pn - po) > 0; dn = (pn - po) < 0
    cont = (up[y == 1].mean() - dn[y == 1].mean()) + (dn[y == 0].mean() - up[y == 0].mean())
    is_n = pn[y == 1].mean() - pn[y == 0].mean(); is_o = po[y == 1].mean() - po[y == 0].mean()
    return float(cont), float(is_n - is_o)
ctx, idix = nri_idi(yte.astype(bool), p_x, p_r)
cl, idil = nri_idi(yte.astype(bool), p_x, p_l)
def nri_cat(y, pn, po, thr):
    yn, yo = pn >= thr, po >= thr
    up, dn = (yn & ~yo), (~yn & yo); yb = y == 1
    return float((up[yb].mean() - dn[yb].mean()) + (dn[~yb].mean() - up[~yb].mean()))
out1["nri_idi"] = {"vs_clinical_rule": {"nri_continuous": round(ctx, 4), "idi": round(idix, 4)},
                   "vs_logistic": {"nri_continuous": round(cl, 4), "idi": round(idil, 4)},
                   "nri_categorical_vs_rule_thr_0.16": round(nri_cat(yte.astype(bool), p_x, p_r, YTHR), 4),
                   "nri_continuous_ci_vs_rule": boot_ci(lambda i: nri_idi(yte[i].astype(bool), p_x[i], p_r[i])[0]),
                   "idi_ci_vs_rule": boot_ci(lambda i: nri_idi(yte[i].astype(bool), p_x[i], p_r[i])[1])}
print("   ", out1["nri_idi"], flush=True)
dump1()

print("P1-03 临床阈值操作特性 ...", flush=True)
def ops(y, p, thr):
    pred = p >= thr; n = len(y); tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
    fn_ = int((~pred & (y == 1)).sum()); tn = int((~pred & (y == 0)).sum())
    return {"threshold": thr, "sensitivity": round(tp / (tp + fn_), 4), "specificity": round(tn / (tn + fp), 4),
            "ppv": round(tp / max(tp + fp, 1), 4), "npv": round(tn / max(tn + fn_, 1), 4),
            "flagged_per_1000": round(1000 * (tp + fp) / n, 1), "tp_per_1000": round(1000 * tp / n, 1),
            "fp_per_1000": round(1000 * fp / n, 1)}
out1["ops_at_thresholds"] = {"xgb": {str(t): ops(yte, p_x, t) for t in (0.05, 0.10)},
                             "rule": {str(t): ops(yte, p_r, t) for t in (0.05, 0.10)}}
print("   XGB@10%:", out1["ops_at_thresholds"]["xgb"]["0.1"], flush=True)
dump1()

print("P1-04 加权 AUC + 设计型 CI ...", flush=True)
w = test["wt"].values.astype(float) if "wt" in test.columns else np.ones(len(test))
out1["weighted"] = {"auc_xgb": round(w_auc(yte, p_x, w), 4), "auc_lr": round(w_auc(yte, p_l, w), 4),
                    "auc_rule": round(w_auc(yte, p_r, w), 4)}
# 个体级加权 bootstrap（权固定）
def wb(p):
    rng = np.random.default_rng(RNG); idx = np.arange(len(yte)); o = []
    for _ in range(2000):
        i = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(yte[i])) < 2: continue
        o.append(w_auc(yte[i], p[i], w[i]))
    a = np.array(o); return [round(float(np.percentile(a, 2.5)), 4), round(float(np.percentile(a, 97.5)), 4)]
out1["weighted"]["auc_xgb_ci_individual"] = wb(p_x); out1["weighted"]["auc_rule_ci_individual"] = wb(p_r)
if HAVE_DESIGN:
    psu = test["SDMVPSU"].values; stra = test["SDMVSTRA"].values
    keys = pd.DataFrame({"s": stra, "p": psu}).drop_duplicates().values
    rng = np.random.default_rng(RNG); o = []
    for _ in range(2000):
        pick = np.concatenate([rng.choice(keys[keys[:, 0] == s][:, 1], (keys[:, 0] == s).sum(), replace=True) for s in np.unique(stra)])
        # 用 PSU 抽样重排（重复 PSU 加倍权重）
        cnt = pd.Series(pick).value_counts()
        mult = pd.Series(psu).map(cnt).fillna(0).values
        m = mult > 0
        if len(np.unique(yte[m])) < 2: continue
        o.append(w_auc(yte[m], p_x[m], w[m] * mult[m]))
    a = np.array(o)
    out1["weighted"]["auc_xgb_ci_design"] = [round(float(np.percentile(a, 2.5)), 4), round(float(np.percentile(a, 97.5)), 4)]
    out1["weighted"]["design_cols"] = {"PSU": int(len(np.unique(psu))), "strata": int(len(np.unique(stra)))}
print("   ", out1["weighted"], flush=True)
json.dump({"note": "P1 statistical addendum (expert-panel items A-phase)", "results": out1},
          open(os.path.join(RES, "p1_statistics.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("  ✅ p1_statistics.json 已写（用时 %.0fs）" % (time.time() - t0), flush=True)
json.dump({"placeholder": True}, open(os.path.join(RES, "_p1_stage1_done.json"), "w", encoding="utf-8"))
print("STAGE1_DONE", flush=True)
