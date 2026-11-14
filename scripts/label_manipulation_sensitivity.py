# -*- coding: utf-8 -*-
"""Label-manipulation sensitivity analysis (pre-registered in analysis_plan/labelmanip_prereg.json).
Two stages:
  --stage labels  power diagnostics only, without training: change rate, prevalence and severity composition under age-free labels
  --stage run     2x2 retraining (label x feature set) with a paired bootstrap for the blind-stratum delta AUC, judged against the pre-registered rule
  --stage both    diagnostics then training (recommended)
Evidence discipline: cohort.parquet is never modified; model recipe, seed and B match the main analysis; every CI comes from actual resampling.
"""
import os, sys, json, time, argparse
import numpy as np, pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTD = os.path.join(BASE, "results/m2/labelmanip")
FEATURES = ["age", "sex", "race", "poverty_ratio", "education", "bmi", "sbp", "dbp",
            "hba1c", "diabetes", "total_cholesterol", "hdl"]
B, SEED = 2000, 20261014
RNG = np.random.default_rng(SEED)
D_ORIG = 0.0258          # pre-registered baseline (results/m2/ci_bootstrap_3items.json)


def ad(X):
    X = X.copy()
    if "race" in X.columns:
        X["race_black"] = (X["race"] == 4).astype(float)
        X["race_hisp"] = (X["race"].isin([1, 2])).astype(float)
        X["race_other"] = (~X["race"].isin([3, 4, 1, 2])).astype(float)
        X = X.drop(columns=["race"])
    if "sex" in X.columns:
        X["female"] = (X["sex"] == 2).astype(float)
        X = X.drop(columns=["sex"])
    return X


def prep_fit(X):
    med = X.median()
    def tf(X_):
        X_ = X_.copy(); m = X_.isna()
        return pd.concat([X_.fillna(med), m.add_prefix("miss_")], axis=1)
    return tf


def ci(a):
    a = np.asarray(a, float)
    return [round(float(np.percentile(a, 2.5)), 4), round(float(np.percentile(a, 97.5)), 4)]


def add_label(df):
    df = df.copy()
    df["egfr_nf"] = df["egfr"] / (0.9938 ** df["age"])
    df["ckd_nf"] = ((df["egfr_nf"] < 60) | (df["uacr"] >= 30)).astype(float)
    return df


def stage_labels(df):
    te = df[df["year"] == 2017]
    teb = te[te["uacr"] < 30]
    res = {"n_all": int(len(df)), "n_test": int(len(te)),
           "prev_orig": round(float(df["ckd"].mean()), 4), "prev_nf": round(float(df["ckd_nf"].mean()), 4),
           "test_prev_orig": round(float(te["ckd"].mean()), 4), "test_prev_nf": round(float(te["ckd_nf"].mean()), 4)}
    res["chg_all"] = round(float((df["ckd"] != df["ckd_nf"]).mean()), 4)
    res["chg_blind"] = round(float((teb["ckd"] != teb["ckd_nf"]).mean()), 4)
    res["blind_n_orig"] = int(teb["ckd"].sum()); res["blind_n_nf"] = int(teb["ckd_nf"].sum())
    res["by_age"] = {}
    for lab, lo, hi in [("18-44", 18, 44), ("45-64", 45, 64), ("65+", 65, 200)]:
        s = df[(df["age"] >= lo) & (df["age"] <= hi)]
        sb = teb[(teb["age"] >= lo) & (teb["age"] <= hi)]
        res["by_age"][lab] = {"n": int(len(s)),
                              "prev_orig": round(float(s["ckd"].mean()), 4),
                              "prev_nf": round(float(s["ckd_nf"].mean()), 4),
                              "chg": round(float((s["ckd"] != s["ckd_nf"]).mean()), 4),
                              "blind_chg": round(float((sb["ckd"] != sb["ckd_nf"]).mean()), 4) if len(sb) else None}
    res["power_ok"] = bool(res["chg_blind"] >= 0.05)
    res["verdict"] = ("manipulation has power; proceed to run" if res["power_ok"]
                      else "INCONCLUSIVE (blind-stratum change rate below 5%, no power; not to be read as a negative result)")
    return res


def fit_arm(tr, te, label, feats):
    Xtr, Xte = ad(tr[feats]), ad(te[feats])
    f = prep_fit(Xtr)
    m = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                      colsample_bytree=0.8, eval_metric="logloss", n_jobs=4, random_state=42)
    m.fit(f(Xtr), tr[label].values)
    return m.predict_proba(f(Xte))[:, 1]


def stage_run(df):
    out = {"when": time.strftime("%Y-%m-%dT%H:%M:%S"), "B": B, "seed": SEED,
           "label_definition": "ckd_nf = (egfr/0.9938^age < 60) | (uacr >= 30)", "cells": {}}
    tr = df[df["year"].isin([2011, 2013, 2015])]
    te = df[df["year"] == 2017]
    for label in ("ckd", "ckd_nf"):
        trb = tr[tr["uacr"] < 30].reset_index(drop=True)
        teb = te[te["uacr"] < 30].reset_index(drop=True)
        y = teb[label].values
        if y.min() == y.max():
            out["cells"][label] = {"error": "single-class blind stratum; cannot be evaluated"}; continue
        feats_nf = [f for f in FEATURES if f != "age"]
        preds = {"age-only": fit_arm(trb, teb, label, ["age"]),
                 "full": fit_arm(trb, teb, label, FEATURES),
                 "full-no-age": fit_arm(trb, teb, label, feats_nf)}
        auc = {k: round(float(roc_auc_score(y, v)), 4) for k, v in preds.items()}
        n = len(y); boot = {k: [] for k in preds}
        for _ in range(B):
            idx = RNG.integers(0, n, n); yy = y[idx]
            if yy.min() == yy.max():
                continue
            for k, p in preds.items():
                boot[k].append(roc_auc_score(yy, p[idx]))
        cell = {"n_blind": int(n), "n_case": int(y.sum()), "auc": auc, "ci95": {}}
        for k in preds:
            cell["ci95"][k] = ci(boot[k])
        for cmp_ in ("full", "full-no-age"):
            d = np.array(boot[cmp_]) - np.array(boot["age-only"])
            cell["delta_" + cmp_] = {"delta": round(auc[cmp_] - auc["age-only"], 4), "ci95": ci(d),
                                     "n_boot_used": len(d),
                                     "ci_excludes_zero": bool(np.percentile(d, 2.5) > 0 or np.percentile(d, 97.5) < 0)}
        out["cells"][label] = cell
        print("[%s] blind n=%d cases=%d | AUC %s | delta full %s %s | delta no-age %s %s"
              % (label, n, int(y.sum()), auc, cell["delta_full"]["delta"], cell["delta_full"]["ci95"],
                 cell["delta_full_no-age"]["delta"] if False else cell["delta_full-no-age"]["delta"],
                 cell["delta_full-no-age"]["ci95"]), flush=True)
    # ---- apply the pre-registered decision rule (cells with n_case < 50 are excluded from testing) ----
    MIN_CASES = 50
    a, b = out["cells"].get("ckd"), out["cells"].get("ckd_nf")
    if a and b and "delta_full" in a and "delta_full" in b:
        d0, d1 = a["delta_full"]["delta"], b["delta_full"]["delta"]
        ex = b["delta_full"]["ci_excludes_zero"]
        powered = b["n_case"] >= MIN_CASES
        if not powered:
            v = ("INCONCLUSIVE_POWER: age-free cell n_case=%d < %d and CI width %.3f, so the cell has no power; "
                 "report descriptive values only, without deciding H1/H2/H3" % (b["n_case"], MIN_CASES,
                                                   b["delta_full"]["ci95"][1] - b["delta_full"]["ci95"][0]))
        elif d1 <= 0.5 * D_ORIG and not ex:
            v = "H1_definition_dominant: definitional contribution >= 50%, increment not significant under age-free labels"
        elif ex and d1 >= 0.8 * D_ORIG:
            v = "H2_increment_genuine: the advantage is not definitional, so the current manuscript wording must be downgraded"
        elif ex:
            v = "H3_partial: definitional share = %.0f%% (1 - %.4f/%.4f)" % (100 * (1 - d1 / d0), d1, d0)
        else:
            v = "grey zone: increment not significant and reduction below 50%; report values without a strong conclusion"
        out["verdict"] = {"rule": v, "powered": powered, "min_cases": MIN_CASES,
                          "delta_orig": d0, "delta_nf": d1,
                          "cidiff_excludes_zero": ex, "prereg_baseline": D_ORIG}
        print("\ndecision:", v)
    else:
        out["verdict"] = {"rule": "INCOMPLETE (missing cell)"}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["labels", "run", "both"], default="both")
    a = ap.parse_args()
    os.makedirs(OUTD, exist_ok=True)
    df = add_label(pd.read_parquet(os.path.join(BASE, "results/cohort.parquet")))
    if a.stage in ("labels", "both"):
        r = stage_labels(df)
        json.dump(r, open(os.path.join(OUTD, "labels_diagnostic.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        print("label diagnostics: overall change %.2f%% | blind stratum %.2f%% | blind cases %d -> %d | %s"
              % (100 * r["chg_all"], 100 * r["chg_blind"], r["blind_n_orig"], r["blind_n_nf"], r["verdict"]))
        for k, v in r["by_age"].items():
            print("   %-6s n=%5d prev %.3f→%.3f chg %.1f%%" % (k, v["n"], v["prev_orig"], v["prev_nf"], 100 * v["chg"]))
        if not r["power_ok"]:
            print("\n[gate] blind-stratum change rate below 5%%: pre-registered INCONCLUSIVE, training stopped.")
            sys.exit(0)
    if a.stage in ("run", "both"):
        o = stage_run(df)
        json.dump(o, open(os.path.join(OUTD, "labelmanip_results.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        print("saved:", os.path.join(OUTD, "labelmanip_results.json"))


if __name__ == "__main__":
    main()
