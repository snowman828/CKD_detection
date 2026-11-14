# -*- coding: utf-8 -*-
"""#4 标签操纵敏感性（预注册：投稿包/_qc/02_npjDM_Followup/labelmanip_prereg.json）
两阶段：
  --stage labels  仅做功效诊断（不训练）：去年龄标签的改变率/患病率/严重度构成
  --stage run     2×2（标签×特征）重训 + 盲区层 ΔAUC 配对 bootstrap，按预注册判据出判决
  --stage both    先诊断后训练（推荐）
证据纪律：不覆盖 cohort.parquet；模型配方/seed/B 与主分析一致；CI 一律来自实际重抽。
"""
import os, sys, json, time, argparse
import numpy as np, pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

BASE = r"G:/项目文件/CKD多模态AI早诊_本项目专属归档/09_无湿实验执行"
OUTD = os.path.join(BASE, "results/m2/labelmanip")
FEATURES = ["age", "sex", "race", "poverty_ratio", "education", "bmi", "sbp", "dbp",
            "hba1c", "diabetes", "total_cholesterol", "hdl"]
B, SEED = 2000, 20261014
RNG = np.random.default_rng(SEED)
D_ORIG = 0.0258          # 预注册基线（results/m2/ci_bootstrap_3items.json）


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
    res["verdict"] = ("操纵有功效，可进入 run" if res["power_ok"]
                      else "INCONCLUSIVE（盲区层改变率<5%，操纵无功效；不得解释为阴性）")
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
            out["cells"][label] = {"error": "盲区层单一类别，无法评估"}; continue
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
        print("[%s] 盲区 n=%d case=%d | AUC %s | Δfull %s %s | Δno-age %s %s"
              % (label, n, int(y.sum()), auc, cell["delta_full"]["delta"], cell["delta_full"]["ci95"],
                 cell["delta_full_no-age"]["delta"] if False else cell["delta_full-no-age"]["delta"],
                 cell["delta_full-no-age"]["ci95"]), flush=True)
    # ---- 按预注册判据出判决（含功效下限：n_case<50 的格不参与假设检验）----
    MIN_CASES = 50
    a, b = out["cells"].get("ckd"), out["cells"].get("ckd_nf")
    if a and b and "delta_full" in a and "delta_full" in b:
        d0, d1 = a["delta_full"]["delta"], b["delta_full"]["delta"]
        ex = b["delta_full"]["ci_excludes_zero"]
        powered = b["n_case"] >= MIN_CASES
        if not powered:
            v = ("INCONCLUSIVE_POWER：去年龄格 n_case=%d < %d，CI 宽 %.3f ⇒ 不具检验力；"
                 "只报描述值，不得据以判 H1/H2/H3" % (b["n_case"], MIN_CASES,
                                                   b["delta_full"]["ci95"][1] - b["delta_full"]["ci95"][0]))
        elif d1 <= 0.5 * D_ORIG and not ex:
            v = "H1_definition_dominant：定义性贡献≥50%，去年龄标签下增量不显著"
        elif ex and d1 >= 0.8 * D_ORIG:
            v = "H2_increment_genuine：优势非定义性 ⇒ 必须降级现稿表述"
        elif ex:
            v = "H3_partial：定义性占比 = %.0f%%（1 − %.4f/%.4f）" % (100 * (1 - d1 / d0), d1, d0)
        else:
            v = "灰度：增量不显著但降幅<50%，报告数值不下强结论"
        out["verdict"] = {"rule": v, "powered": powered, "min_cases": MIN_CASES,
                          "delta_orig": d0, "delta_nf": d1,
                          "cidiff_excludes_zero": ex, "prereg_baseline": D_ORIG}
        print("\n判决:", v)
    else:
        out["verdict"] = {"rule": "INCOMPLETE（缺单元格）"}
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
        print("标签诊断: 全体改变率 %.2f%% | 盲区层 %.2f%% | 盲区病例 %d→%d | %s"
              % (100 * r["chg_all"], 100 * r["chg_blind"], r["blind_n_orig"], r["blind_n_nf"], r["verdict"]))
        for k, v in r["by_age"].items():
            print("   %-6s n=%5d prev %.3f→%.3f chg %.1f%%" % (k, v["n"], v["prev_orig"], v["prev_nf"], 100 * v["chg"]))
        if not r["power_ok"]:
            print("\n[闸] 盲区改变率<5%% ⇒ 按预注册判 INCONCLUSIVE，停止训练。")
            sys.exit(0)
    if a.stage in ("run", "both"):
        o = stage_run(df)
        json.dump(o, open(os.path.join(OUTD, "labelmanip_results.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        print("saved:", os.path.join(OUTD, "labelmanip_results.json"))


if __name__ == "__main__":
    main()
