# -*- coding: utf-8 -*-
"""KNHANES cross-system validation (runs once the source files are in place).

Design
- Development set, hyperparameters, imputation, features and threshold (0.1602) are frozen and identical to the main analysis.
- Self-check gate: the same frozen model must reproduce AUC 0.8099 on NHANES 2017-2018; otherwise abort.
- Variables are matched by regular expression against the official KNHANES codebook; unmatched items are logged.
- Unit gate: a median uACR outside 2-20 mg/g is treated as a unit problem and aborts the run rather than emitting suspect numbers.
- Output: results/p3_knh_validation.json
Usage
  python scripts/p3_knh_validation.py --knh "" [--dry-run]
"""
import argparse, glob, io, json, os, re, sys, zipfile, tempfile
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repository root (runs from a fresh clone)
DATA, OUT = os.path.join(PROJ, "data"), os.path.join(PROJ, "results")
FEATURES = ["age", "sex", "race", "poverty_ratio", "education", "bmi", "sbp", "dbp", "hba1c", "diabetes", "total_cholesterol", "hdl"]
RNG, YTHR = 2026, 0.1602
PAT = {
    "age": r"^age$|^AGE$", "sex": r"^sex$|^SEX$", "education": r"^edu$|^EDU$", "poverty_ratio": r"incm|INCM|income",
    "bmi": r"^BMI$|^bmi$", "sbp": r"BPXSY1|^sbp$|^SBP$", "dbp": r"BPXDI1|^dbp$|^DBP$", "hba1c": r"HbA1c|^hba1c$",
    "total_cholesterol": r"^chol$|^TCHO$|^chol_total$", "hdl": r"^HDL|^hdl", "diabetes": r"dg|DG|diag",
    "creatinine": r"crea|CREA", "ualb": r"Ualb|UALB|alb", "ucrea": r"Ucrea|UCREA", "pregnant": r"preg|PREG",
    "wt": r"^wt_|^WT_", "id": r"^ID$|^id$",
}

def load_dir(d):
    files = []
    for ext in ("*.sas7bdat", "*.sav", "*.xpt", "*.zip"):
        files += glob.glob(os.path.join(d, "**", ext), recursive=True)
    ok = []
    for f in files:
        try:
            if f.lower().endswith(".zip"):
                z = zipfile.ZipFile(f); ex = tempfile.mkdtemp()
                z.extractall(ex)
                for inner in glob.glob(os.path.join(ex, "**", "*"), recursive=True):
                    if inner.lower().endswith((".sas7bdat", ".xpt", ".sav")): ok.append(inner)
            else:
                ok.append(f)
        except Exception as e:
            print(f"  ⚠️ failed to unpack or read {os.path.basename(f)}: {e}")
    return ok

def read_any(f):
    if f.lower().endswith((".sas7bdat", ".xpt")):
        return pd.read_sas(f, format="xport" if f.lower().endswith(".xpt") else None, encoding="latin1")
    try:
        import pyreadstat
        df, _ = pyreadstat.read_sav(f)
        return df
    except Exception as e:
        print(f"  ⚠️ pyreadstat is required to read .sav files（{e}）-> use the SAS transport format instead")
        return None

def pick(df, key):
    if df is None: return None
    if key == "id":
        for c in df.columns:
            if re.match(PAT["id"], str(c)): return c
        return None
    cand = [c for c in df.columns if re.search(PAT[key], str(c), re.I)]
    return cand[0] if cand else None

def _dry_run(DATA):
    """--dry-run: no KNHANES data required; reproduce the frozen pipeline gate and print the input checklist."""
    import pandas as pd
    from xgboost import XGBClassifier
    from sklearn.metrics import roc_auc_score
    print("== input readiness check (--dry-run) ==")
    feats = ["age", "sex", "race", "poverty_ratio", "education", "bmi", "sbp", "dbp", "hba1c", "diabetes", "total_cholesterol", "hdl"]
    print("  KNHANES inputs required (12 features plus 2 label components):")
    for f in feats:
        print("     - %s" % f)
    print("     - creatinine (serum, for eGFR)")
    print("     - urine albumin and urine creatinine (for uACR; expected in mg/L and mg/dL, enforced by the unit gate)")
    print("  NHANES data directory: %s(currently %d  XPT files)" % (DATA, len(glob.glob(os.path.join(DATA, "*.XPT")))))
    CYC = {"G": 2011, "H": 2013, "I": 2015}
    def nhanes(cyc):
        L = lambda n: pd.read_sas(os.path.join(DATA, f"{n}_{cyc}.XPT"), format="xport")
        dem, alb, bio = L("DEMO"), L("ALB_CR"), L("BIOPRO")
        bpx, bmx, diq, ghb, tc, hd = L("BPX"), L("BMX"), L("DIQ"), L("GHB"), L("TCHOL"), L("HDL")
        x = dem[["SEQN", "RIDAGEYR", "RIAGENDR", "RIDRETH1", "INDFMPIR", "DMDEDUC2", "WTMEC2YR", "RIDEXPRG"]].copy()
        x = x.rename(columns={"RIDAGEYR": "age", "RIAGENDR": "sex", "RIDRETH1": "race", "INDFMPIR": "poverty_ratio", "DMDEDUC2": "education", "WTMEC2YR": "wt", "RIDEXPRG": "pregnant"})
        M = lambda s, c, p: x.merge(s[["SEQN"] + c].rename(columns={v: p + v for v in c}), on="SEQN", how="left")
        x = M(alb, ["URXUMA", "URXUCR"], "a_"); x = M(bio, ["LBXSCR"], "b_"); x = M(bpx, ["BPXSY1", "BPXDI1"], "c_")
        x = M(bmx, ["BMXBMI"], "d_"); x = M(diq, ["DIQ010"], "e_"); x = M(ghb, ["LBXGH"], "f_"); x = M(tc, ["LBXTC"], "g_"); x = M(hd, ["LBDHDD"], "g_")
        x["uacr"] = (x["a_URXUMA"] * 100.0 / x["a_URXUCR"]).where(x["a_URXUCR"] > 0)
        x["creatinine"] = x["b_LBXSCR"]; x["sbp"] = x["c_BPXSY1"]; x["dbp"] = x["c_BPXDI1"]
        x["bmi"] = x["d_BMXBMI"]; x["hba1c"] = x["f_LBXGH"]; x["total_cholesterol"] = x["g_LBXTC"]; x["hdl"] = x["g_LBDHDD"]
        x["diabetes"] = np.where(x["e_DIQ010"] == 1, 1.0, np.where(x["e_DIQ010"] == 2, 0.0, np.nan))
        kk = np.where(x["sex"] == 2, 0.7, 0.9); aa = np.where(x["sex"] == 2, -0.241, -0.302); ss = x["creatinine"] / kk
        x["egfr"] = (142.0 * np.minimum(ss, 1.0) ** aa * np.maximum(ss, 1.0) ** (-1.200) * 0.9938 ** x["age"] * np.where(x["sex"] == 2, 1.012, 1.0))
        x["ckd"] = ((x["egfr"] < 60) | (x["uacr"] >= 30)).astype(float)
        return x[(x["age"] >= 18) & (x["pregnant"] != 1) & x["ckd"].notna()]
    dev = pd.concat([nhanes(c) for c in CYC], ignore_index=True); val = nhanes("J")
    def enc(X):
        X = X.copy(); X["race_black"] = (X["race"] == 4).astype(float); X["race_hisp"] = X["race"].isin([1, 2]).astype(float)
        X["race_other"] = (~X["race"].isin([1, 2, 3, 4])).astype(float); X["female"] = (X["sex"] == 2).astype(float)
        return X.drop(columns=["race", "sex"])
    med2 = enc(dev[FEATURES]).median()
    F = lambda X: pd.concat([X.fillna(med2), X.isna().add_prefix("miss_")], axis=1)
    Xtr, Xv = F(enc(dev[FEATURES])), F(enc(val[FEATURES]))
    xgb = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", n_jobs=4, random_state=42).fit(Xtr, dev["ckd"].values)
    a17 = roc_auc_score(val["ckd"].values, xgb.predict_proba(Xv)[:, 1])
    print("  self-check gate: NHANES 2017-2018 XGB AUC=%.4f(expected 0.8099)%s" % (a17, "PASS" if abs(a17 - 0.8099) <= 1e-4 else "DRIFT"))
    ok = abs(a17 - 0.8099) <= 1e-4
    print("  conclusion: %s - runnable as soon as the data are in place" % ("pipeline healthy; waiting for KNHANES source files" if ok else "pipeline drift; investigate before proceeding"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--knh", required=False, default=None)
    ap.add_argument("--dry-run", action="store_true"); a = ap.parse_args()
    if a.dry_run:
        sys.exit(_dry_run(DATA))
    if not a.knh: sys.exit("❌ provide --knh <dir> (or use --dry-run for the readiness check)")
    if not os.path.isdir(a.knh): sys.exit(f"❌ directory does not exist: {a.knh}")
    files = load_dir(a.knh)
    print(f"发现 {len(files)}  data files found")
    if not files: sys.exit("❌ no .sas7bdat/.sav/.xpt/.zip found; place the files as described in DATA_AVAILABILITY.md")

    # ---- per-file scan: print the variable mapping for manual review ----
    merged, report = None, {}
    for f in files:
        df = read_any(f)
        if df is None or len(df) == 0: continue
        idc = pick(df, "id")
        if idc is None:
            print(f"  ⚠️ {os.path.basename(f)}: no ID column; skipped"); continue
        sel = {"ID": idc}
        for k in PAT:
            if k == "id": continue
            c = pick(df, k)
            if c is not None: sel[k] = c
        if len(sel) <= 1: continue
        sub = df[list(sel.values())].rename(columns={v: k for k, v in sel.items()})
        report[os.path.basename(f)] = {k: sel[k] for k in sel if k != "ID"}
        merged = sub if merged is None else merged.merge(sub, on="ID", how="outer")
    if merged is None or len(merged) == 0: sys.exit("❌ could not merge any file; check the ID column")
    print("variable mapping:"); print(json.dumps(report, ensure_ascii=False, indent=1)[:1400])
    need = ["age", "sex", "creatinine", "ualb", "ucrea"]
    miss = [k for k in need if k not in merged.columns]
    if miss: sys.exit(f"❌ missing key variables {miss} —— need the file group holding serum creatinine and urine albumin/creatinine")

    d = merged.copy()
    d = d[d["age"].notna() & (d["age"] >= 18)]
    if "pregnant" in d: d = d[d["pregnant"] != 1]
    # urine albumin mg/L and creatinine mg/dL -> uACR(mg/g) = Ualb * 100 / Ucrea
    # (identical to the NHANES analysis; a different unit in any wave trips the median gate instead of producing suspect values)
    d["uacr"] = (d["ualb"] * 100.0 / d["ucrea"]).where(d["ualb"].notna() & d["ucrea"].notna() & (d["ucrea"] > 0))
    med = float(np.nanmedian(d["uacr"]))
    print(f"  median uACR = {med:.2f} mg/g(unit gate window 2-20)")
    if not (2.0 <= med <= 20.0):
        sys.exit(f"❌ uACR 中位数 {med:.2f} out of range: unit mismatch. Currently computed as Ualb(mg/L) * 100 / Ucrea(mg/dL);"
                 f"check the official codebook (if Ucrea is in mg/L, use Ualb/Ucrea); the script emits nothing rather than a suspect value.")
    k = np.where(d["sex"] == 2, 0.7, 0.9); al = np.where(d["sex"] == 2, -0.241, -0.302)
    s = d["creatinine"] / k
    d["egfr"] = (142.0 * np.minimum(s, 1.0) ** al * np.maximum(s, 1.0) ** (-1.200) * 0.9938 ** d["age"] * np.where(d["sex"] == 2, 1.012, 1.0)).where(d["creatinine"].notna())
    d["ckd"] = ((d["egfr"] < 60) | (d["uacr"] >= 30)).astype(float)
    d = d[d["ckd"].notna()]
    print(f"  KNHANES cohort n={len(d)} CKD={int(d['ckd'].sum())} ({100*d['ckd'].mean():.2f}%)")

    # ---- NHANES development set (same pipeline) and self-check ----
    CYC = {"G": 2011, "H": 2013, "I": 2015}
    def nhanes(cyc):
        L = lambda n: pd.read_sas(os.path.join(DATA, f"{n}_{cyc}.XPT"), format="xport")
        dem, alb, bio = L("DEMO"), L("ALB_CR"), L("BIOPRO")
        bpx, bmx, diq, ghb, tc, hd = L("BPX"), L("BMX"), L("DIQ"), L("GHB"), L("TCHOL"), L("HDL")
        x = dem[["SEQN", "RIDAGEYR", "RIAGENDR", "RIDRETH1", "INDFMPIR", "DMDEDUC2", "WTMEC2YR", "RIDEXPRG"]].copy()
        x = x.rename(columns={"RIDAGEYR": "age", "RIAGENDR": "sex", "RIDRETH1": "race", "INDFMPIR": "poverty_ratio", "DMDEDUC2": "education", "WTMEC2YR": "wt", "RIDEXPRG": "pregnant"})
        M = lambda s, c, p: x.merge(s[["SEQN"] + c].rename(columns={v: p + v for v in c}), on="SEQN", how="left")
        x = M(alb, ["URXUMA", "URXUCR"], "a_"); x = M(bio, ["LBXSCR"], "b_"); x = M(bpx, ["BPXSY1", "BPXDI1"], "c_")
        x = M(bmx, ["BMXBMI"], "d_"); x = M(diq, ["DIQ010"], "e_"); x = M(ghb, ["LBXGH"], "f_"); x = M(tc, ["LBXTC"], "g_"); x = M(hd, ["LBDHDD"], "g_")
        x["uacr"] = (x["a_URXUMA"] * 100.0 / x["a_URXUCR"]).where(x["a_URXUCR"] > 0)
        x["creatinine"] = x["b_LBXSCR"]; x["sbp"] = x["c_BPXSY1"]; x["dbp"] = x["c_BPXDI1"]
        x["bmi"] = x["d_BMXBMI"]; x["hba1c"] = x["f_LBXGH"]; x["total_cholesterol"] = x["g_LBXTC"]; x["hdl"] = x["g_LBDHDD"]
        x["diabetes"] = np.where(x["e_DIQ010"] == 1, 1.0, np.where(x["e_DIQ010"] == 2, 0.0, np.nan))
        kk = np.where(x["sex"] == 2, 0.7, 0.9); aa = np.where(x["sex"] == 2, -0.241, -0.302); ss = x["creatinine"] / kk
        x["egfr"] = (142.0 * np.minimum(ss, 1.0) ** aa * np.maximum(ss, 1.0) ** (-1.200) * 0.9938 ** x["age"] * np.where(x["sex"] == 2, 1.012, 1.0))
        x["ckd"] = ((x["egfr"] < 60) | (x["uacr"] >= 30)).astype(float)
        return x[(x["age"] >= 18) & (x["pregnant"] != 1) & x["ckd"].notna()]
    dev = pd.concat([nhanes(c) for c in CYC], ignore_index=True); val = nhanes("J")
    def enc(X):
        X = X.copy(); X["race_black"] = (X["race"] == 4).astype(float); X["race_hisp"] = X["race"].isin([1, 2]).astype(float)
        X["race_other"] = (~X["race"].isin([1, 2, 3, 4])).astype(float); X["female"] = (X["sex"] == 2).astype(float)
        return X.drop(columns=["race", "sex"])
    med2 = enc(dev[FEATURES]).median()
    F = lambda X: pd.concat([X.fillna(med2), X.isna().add_prefix("miss_")], axis=1)
    Xtr, Xv = F(enc(dev[FEATURES])), F(enc(val[FEATURES]))
    xgb = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", n_jobs=4, random_state=42).fit(Xtr, dev["ckd"].values)
    a17 = roc_auc_score(val["ckd"].values, xgb.predict_proba(Xv)[:, 1])
    print(f"  self-check gate: NHANES 2017-2018 XGB AUC={a17:.4f}(expected 0.8099)")
    if abs(a17 - 0.8099) > 1e-4: sys.exit("self-check failed: pipeline drift, aborting")

    # ---- external evaluation on KNHANES (median imputation plus missingness flags, as in the main analysis) ----
    for f in FEATURES:
        if f not in d.columns:
            d[f] = np.nan
            print(f"  ⚠️ KNHANES missing feature {f} -> imputed with the development-set median plus a missingness flag, noted in the results")
    Xk = F(enc(d[FEATURES])); pk = xgb.predict_proba(Xk)[:, 1]; yk = d["ckd"].values

    def boot(y, p, n=5000):
        r = np.random.default_rng(RNG); idx = np.arange(len(y)); o = []
        for _ in range(n):
            i = r.choice(idx, len(idx), replace=True)
            if len(np.unique(y[i])) > 1: o.append(roc_auc_score(y[i], p[i]))
        a = np.array(o); return [round(float(np.percentile(a, 2.5)), 4), round(float(np.percentile(a, 97.5)), 4)]
    q = np.quantile(pk, np.linspace(0, 1, 11)); q[0], q[-1] = -np.inf, np.inf
    mp, ob = [], []
    for i in range(10):
        m = (pk > q[i]) & (pk <= q[i + 1])
        if m.sum(): mp.append(pk[m].mean()); ob.append(yk[m].mean())
    c = np.polyfit(mp, ob, 1); ece = float(np.mean(np.abs(np.array(mp) - np.array(ob))))
    pred = pk >= YTHR; tp = int((pred & (yk == 1)).sum()); fp = int((pred & (yk == 0)).sum()); fn_ = int((~pred & (yk == 1)).sum()); tn = int((~pred & (yk == 0)).sum())
    nb = lambda t: float((pk >= t).astype(int).dot(yk) / len(yk) - ((pk >= t).astype(int).dot(1 - yk) / len(yk)) * (t / (1 - t)))
    out = {
        "_note": "Cross-system external validation (KNHANES, Korea; different health system, country and ancestry). Frozen models/threshold from NHANES development (2011-2016); internal consistency gate reproduced the published 2017-2018 AUC before evaluation.",
        "cohort": {"n": int(len(d)), "events": int(yk.sum()), "prevalence_pct": round(float(100 * yk.mean()), 2),
                   "age_mean": round(float(d["age"].mean()), 1), "female_pct": round(float(100 * (d["sex"] == 2).mean()), 1),
                   "uacr_median": round(med, 2), "egfr_median": round(float(d["egfr"].median()), 1),
                   "features_imputed": [f for f in FEATURES if d[f].isna().mean() > 0.2]},
        "discrimination": {"xgb": round(float(roc_auc_score(yk, pk)), 4), "xgb_ci": boot(yk, pk)},
        "calibration": {"slope": round(float(c[0]), 3), "intercept": round(float(c[1]), 4), "ece": round(ece, 4)},
        "operating": {"threshold": YTHR, "sensitivity": round(tp / max(tp + fn_, 1e-9), 4), "specificity": round(tn / max(tn + fp, 1e-9), 4),
                      "ppv": round(tp / max(tp + fp, 1e-9), 4), "flagged_per1000": round(1000 * (tp + fp) / len(yk), 1)},
        "decision_curve": {"at_10pct": {"nb_model": round(nb(0.10), 4), "nb_screen_all": round(float(yk.mean() - (1 - yk.mean()) * (0.10 / 0.90)), 4)},
                           "at_20pct": {"nb_model": round(nb(0.20), 4), "nb_screen_all": round(float(yk.mean() - (1 - yk.mean()) * (0.20 / 0.80)), 4)}},
    }
    json.dump(out, open(os.path.join(OUT, "p3_knh_validation.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=1))
    print("\nwrote results/p3_knh_validation.json")

if __name__ == "__main__":
    main()
