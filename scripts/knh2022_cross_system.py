# -*- coding: utf-8 -*-
"""knh2022_cross_system.py - KNHANES 2022 cross-system validation (frozen model, pre-specified race-blind variant).

Design (fully explicit and re-checkable):
  1) Self-check gate: retrain XGBoost on NHANES 2011-2016 and require AUC = 0.8099 (+/-1e-4) on 2017-2018; abort otherwise.
  2) Pre-specified variant: cross-system use drops the three race indicators and their missingness flags (22 inputs).
  3) Harmonisation (scales differ across surveys; every mapping is declared explicitly):
       - education: KNHANES edu 1-4 mapped monotonically onto the NHANES DMDEDUC2 1-5 scale
       - income: KNHANES incm5 quintiles mapped to the development-set PIR quintile means
  4) Unit gate: median uACR must fall within 2-20 mg/g (urine albumin mg/L, urine creatinine mg/dL).
  5) Outputs: AUC with bootstrap CI, calibration (slope/intercept/ECE), sensitivity/specificity/PPV at the frozen 0.16 threshold, and DCA over 5-25%.
  6) Sensitivity: income replaced by the cohort median, and education left unscaled (raw 1-4).

Column names verified against the official hn22_all.sas7bdat label set (DE1_dg marks diabetes; DI1_dg marks hypertension and is not used).
"""
import os, sys, json, numpy as np, pandas as pd, pyreadstat
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

PROJ = r""
KNH = os.path.join(PROJ, r"data\knh\2022")
DATA = os.path.join(PROJ, r"data")
OUT = os.path.join(PROJ, r"results\knh2022_cross_system.json")
HYP = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
           colsample_bytree=0.8, eval_metric="logloss", n_jobs=4, random_state=42)
THR = 0.16  # frozen deployment threshold (Youden point on the development set)


def load_nhanes(cyc):
    L = lambda n: pd.read_sas(os.path.join(DATA, f"{n}_{cyc}.XPT"), format="xport")
    dem, alb, bio = L("DEMO"), L("ALB_CR"), L("BIOPRO")
    bpx, bmx, diq, ghb, tc, hd = L("BPX"), L("BMX"), L("DIQ"), L("GHB"), L("TCHOL"), L("HDL")
    x = dem[["SEQN", "RIDAGEYR", "RIAGENDR", "RIDRETH1", "INDFMPIR", "DMDEDUC2", "RIDEXPRG"]].copy()
    x = x.rename(columns={"RIDAGEYR": "age", "RIAGENDR": "sex", "RIDRETH1": "race",
                          "INDFMPIR": "poverty_ratio", "DMDEDUC2": "education", "RIDEXPRG": "pregnant"})
    M = lambda s, c, p: x.merge(s[["SEQN"] + c].rename(columns={v: p + v for v in c}), on="SEQN", how="left")
    x = M(alb, ["URXUMA", "URXUCR"], "a_"); x = M(bio, ["LBXSCR"], "b_")
    x = M(bpx, ["BPXSY1", "BPXDI1"], "c_"); x = M(bmx, ["BMXBMI"], "d_")
    x = M(diq, ["DIQ010"], "e_"); x = M(ghb, ["LBXGH"], "f_")
    x = M(tc, ["LBXTC"], "g_"); x = M(hd, ["LBDHDD"], "g_")
    x["uacr"] = (x["a_URXUMA"] * 100.0 / x["a_URXUCR"]).where(x["a_URXUCR"] > 0)
    x["creatinine"] = x["b_LBXSCR"]; x["sbp"] = x["c_BPXSY1"]; x["dbp"] = x["c_BPXDI1"]
    x["bmi"] = x["d_BMXBMI"]; x["hba1c"] = x["f_LBXGH"]
    x["total_cholesterol"] = x["g_LBXTC"]; x["hdl"] = x["g_LBDHDD"]
    x["diabetes"] = np.where(x["e_DIQ010"] == 1, 1.0, np.where(x["e_DIQ010"] == 2, 0.0, np.nan))
    k = np.where(x["sex"] == 2, 0.7, 0.9); al = np.where(x["sex"] == 2, -0.241, -0.302)
    s = x["creatinine"] / k
    x["egfr"] = (142.0 * np.minimum(s, 1.0) ** al * np.maximum(s, 1.0) ** (-1.200)
                 * 0.9938 ** x["age"] * np.where(x["sex"] == 2, 1.012, 1.0))
    x["ckd"] = ((x["egfr"] < 60) | (x["uacr"] >= 30)).astype(float)
    return x[(x["age"] >= 18) & (x["pregnant"] != 1) & x["ckd"].notna()]


def load_knh():
    z = os.path.join(KNH, "HN22_ALL(SAS).zip")
    import zipfile
    with zipfile.ZipFile(z) as zz:
        zz.extract("hn22_all.sas7bdat", KNH)
    df, meta = pyreadstat.read_sas7bdat(os.path.join(KNH, "hn22_all.sas7bdat"))
    num = lambda c: pd.to_numeric(df[c], errors="coerce")
    d = pd.DataFrame({
        "age": num("age"), "sex": num("sex"),
        "education_raw": num("edu"), "incm5": num("incm5"),
        "bmi": num("HE_BMI"), "sbp": num("HE_sbp"), "dbp": num("HE_dbp"),
        "hba1c": num("HE_HbA1c"), "total_cholesterol": num("HE_chol"), "hdl": num("HE_HDL_st2"),
        "creatinine": num("HE_crea"), "ualb": num("HE_Ualb"), "ucrea": num("HE_Ucrea"),
        "de1_dg": num("DE1_dg"), "di1_dg_hypertension": num("DI1_dg"),
    })
    d["diabetes"] = np.where(d["de1_dg"] == 1, 1.0, np.where(d["de1_dg"] == 0, 0.0, np.nan))
    d["uacr"] = (d["ualb"] * 100.0 / d["ucrea"]).where(d["ualb"].notna() & d["ucrea"].notna() & (d["ucrea"] > 0))
    k = np.where(d["sex"] == 2, 0.7, 0.9); al = np.where(d["sex"] == 2, -0.241, -0.302)
    s = d["creatinine"] / k
    d["egfr"] = (142.0 * np.minimum(s, 1.0) ** al * np.maximum(s, 1.0) ** (-1.200)
                 * 0.9938 ** d["age"] * np.where(d["sex"] == 2, 1.012, 1.0))
    d["ckd"] = np.where(d["egfr"].notna() | d["uacr"].notna(),
                        ((d["egfr"] < 60) | (d["uacr"] >= 30)).astype(float), np.nan)
    return d


def boot_auc(y, p, n=1000, seed=20260927):
    rng = np.random.default_rng(seed); idx = np.arange(len(y)); out = []
    for _ in range(n):
        s = rng.choice(idx, len(idx), replace=True)
        if 0 < y[s].sum() < len(s):
            out.append(roc_auc_score(y[s], p[s]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main():
    print("== 1) NHANES self-check gate ==")
    dev = pd.concat([load_nhanes(c) for c in ("G", "H", "I")], ignore_index=True)
    val = load_nhanes("J")
    print("   dev n=%d CKD=%d | val n=%d CKD=%d" % (len(dev), int(dev.ckd.sum()), len(val), int(val.ckd.sum())))

    FE = ["age", "sex", "poverty_ratio", "education", "bmi", "sbp", "dbp",
          "hba1c", "diabetes", "total_cholesterol", "hdl"]

    def design_full(X):  # with race indicators (self-check gate must reproduce 0.8099)
        X = X.copy()
        X["race_black"] = (X["race"] == 4).astype(float)
        X["race_hisp"] = X["race"].isin([1, 2]).astype(float)
        X["race_other"] = (~X["race"].isin([1, 2, 3, 4])).astype(float)
        X["female"] = (X["sex"] == 2).astype(float)
        return X.drop(columns=["race", "sex"])

    d_full = design_full(dev[FE + ["race"]]); v_full = design_full(val[FE + ["race"]])
    med_full = d_full.median()
    F = lambda X, m: pd.concat([X.fillna(m), X.isna().add_prefix("miss_")], axis=1)
    m_full = XGBClassifier(**HYP).fit(F(d_full, med_full), dev["ckd"].values)
    auc_ck = roc_auc_score(val["ckd"].values, m_full.predict_proba(F(v_full, med_full))[:, 1])
    print("   self-check gate NHANES 2017-2018 AUC=%.4f(expected 0.8099)%s" % (auc_ck, "✅" if abs(auc_ck - 0.8099) <= 1e-4 else "❌"))
    if abs(auc_ck - 0.8099) > 1e-4:
        sys.exit("self-check gate failed; aborting without cross-system results")

    print("== 2) freeze the race-blind model (pre-specified variant) ==")
    RB = ["age", "sex", "poverty_ratio", "education", "bmi", "sbp", "dbp",
          "hba1c", "diabetes", "total_cholesterol", "hdl"]

    def design_rb(X):
        X = X.copy(); X["female"] = (X["sex"] == 2).astype(float); return X.drop(columns=["sex"])

    d_rb = design_rb(dev[RB]); v_rb = design_rb(val[RB])
    med_rb = d_rb.median()
    m_rb = XGBClassifier(**HYP).fit(F(d_rb, med_rb), dev["ckd"].values)
    p_val = m_rb.predict_proba(F(v_rb, med_rb))[:, 1]
    auc_rb_val = roc_auc_score(val["ckd"].values, p_val)
    print("   race-blind variant on NHANES 2017-2018 AUC=%.4f(compared with 0.8099 using race indicators)" % auc_rb_val)

    print("== 3) KNHANES harmonisation ==")
    kd = load_knh()
    print("   raw n=%d(valid labels %d）" % (len(kd), int(kd.ckd.notna().sum())))
    # income: incm5 quintiles -> quintile means of the development-set PIR
    pir = dev["poverty_ratio"].dropna()
    qs = pir.quantile([0, .2, .4, .6, .8, 1.0]).values
    means = [pir[(pir >= qs[i]) & (pir <= qs[i + 1])].mean() for i in range(5)]
    print("   NHANES PIR quintile means = %s" % [round(float(x), 3) for x in means])
    kd["poverty_ratio"] = kd["incm5"].map({i + 1: means[i] for i in range(5)})
    # education: 1-4 -> 1-5 (monotonic linear)
    kd["education"] = 1.0 + (kd["education_raw"] - 1.0) * (4.0 / 3.0)

    kd = kd[kd["age"] >= 18].copy()
    kd = kd[kd["sex"].isin([1, 2])]
    kd = kd[kd["ckd"].notna()].copy()
    med_u = float(np.nanmedian(kd["uacr"]))
    print("   KNHANES median uACR = %.2f mg/g(gate 2-20)%s" % (med_u, "✅" if 2 <= med_u <= 20 else "❌"))
    if not (2 <= med_u <= 20):
        sys.exit("unit gate failed; aborting")
    print("   KNHANES analytic sample n=%d | CKD=%d (%.2f%%)" % (len(kd), int(kd.ckd.sum()), 100 * kd.ckd.mean()))

    print("== 4) cross-system performance ==")
    Xk = design_rb(kd[RB])
    p_k = m_rb.predict_proba(F(Xk, med_rb))[:, 1]
    y_k = kd["ckd"].values
    auc_k = roc_auc_score(y_k, p_k); lo, hi = boot_auc(y_k, p_k)
    print("   AUC = %.4f (95%% CI %.4f–%.4f)  n=%d  CKD=%d" % (auc_k, lo, hi, len(y_k), int(y_k.sum())))

    # calibration
    eps = 1e-6; lp = np.log(np.clip(p_k, eps, 1 - eps) / (1 - np.clip(p_k, eps, 1 - eps)))
    b1, b0 = np.polyfit(lp, y_k, 1)
    dec = pd.qcut(p_k, 10, duplicates="drop")
    obs = pd.Series(y_k).groupby(dec, observed=True).mean().values
    exp = pd.Series(p_k).groupby(dec, observed=True).mean().values
    wt = pd.Series(y_k).groupby(dec, observed=True).size().values / len(y_k)
    ece = float(np.sum(wt * np.abs(obs - exp)))
    print("   calibration slope=%.3f intercept=%.3f ECE=%.4f" % (b1, b0, ece))

    # frozen threshold
    fl = p_k >= THR
    tp = int(((fl == 1) & (y_k == 1)).sum()); fp = int(((fl == 1) & (y_k == 0)).sum())
    fn = int(((fl == 0) & (y_k == 1)).sum()); tn = int(((fl == 0) & (y_k == 0)).sum())
    sens = tp / (tp + fn); spec = tn / (tn + fp); ppv = tp / max(tp + fp, 1)
    print("   threshold %.2f: sens=%.4f spec=%.4f PPV=%.4f | flagged per 1,000=%.1f true positives per 1,000=%.1f"
          % (THR, sens, spec, ppv, 1000 * fl.mean(), 1000 * y_k.mean()))

    # DCA
    dca = {}
    for t in (0.05, 0.10, 0.15, 0.20, 0.25):
        nb = (tp / len(y_k)) - (fp / len(y_k)) * (t / (1 - t)) if True else None
        # recompute at threshold t (standard DCA)
        f2 = p_k >= t
        tp2 = int(((f2 == 1) & (y_k == 1)).sum()); fp2 = int(((f2 == 1) & (y_k == 0)).sum())
        dca["%.2f" % t] = {"model_nb": float(tp2 / len(y_k) - (fp2 / len(y_k)) * (t / (1 - t))),
                           "treat_all_nb": float(y_k.mean() - (1 - y_k.mean()) * (t / (1 - t)))}
    print("   DCA(model / treat-all): " + " | ".join(
        "%s %.4f/%.4f" % (k, v["model_nb"], v["treat_all_nb"]) for k, v in dca.items()))

    # sensitivity analyses
    pit = []
    kd2 = kd.copy(); kd2["poverty_ratio"] = float(pir.median())
    p2 = m_rb.predict_proba(F(design_rb(kd2[RB]), med_rb))[:, 1]
    pit.append({"label": "income = cohort median (mapping-insensitive check)", "auc": round(float(roc_auc_score(y_k, p2)), 4)})
    kd3 = kd.copy(); kd3["education"] = kd3["education_raw"]  # no rescaling
    p3 = m_rb.predict_proba(F(design_rb(kd3[RB]), med_rb))[:, 1]
    pit.append({"label": "education raw 1-4 (unscaled)", "auc": round(float(roc_auc_score(y_k, p3)), 4)})
    for x in pit: print("   sensitivity %-32s AUC=%.4f" % (x["label"], x["auc"]))

    res = {
        "design": "frozen NHANES 2011-2016 XGBoost; race-blind variant (pre-specified for cross-system use)",
        "self_check_nhanes_2017_2018_auc": round(float(auc_ck), 4),
        "nhanes_race_blind_2017_2018_auc": round(float(auc_rb_val), 4),
        "harmonization": {"income": "KNHANES incm5 (quintile) -> NHANES dev PIR quintile means",
                          "income_quintile_means": [round(float(x), 4) for x in means],
                          "education": "KNHANES edu (1-4) linearly rescaled to NHANES DMDEDUC2 scale (1-5)"},
        "knh_cohort": {"n": int(len(kd)), "n_ckd": int(kd.ckd.sum()), "ckd_pct": round(float(100 * kd.ckd.mean()), 2),
                       "uacr_median_mg_g": round(med_u, 3)},
        "auc": {"value": round(float(auc_k), 4), "ci": [round(lo, 4), round(hi, 4)]},
        "calibration": {"slope": round(float(b1), 4), "intercept": round(float(b0), 4), "ece": round(ece, 4)},
        "at_frozen_threshold_0.16": {"sensitivity": round(sens, 4), "specificity": round(spec, 4),
                                     "ppv": round(ppv, 4), "flagged_per_1000": round(1000 * float(fl.mean()), 1),
                                     "true_cases_per_1000": round(1000 * float(y_k.mean()), 1)},
        "dca": dca,
        "sensitivity_analyses": pit,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\nwrote %s" % OUT)


if __name__ == "__main__":
    main()
