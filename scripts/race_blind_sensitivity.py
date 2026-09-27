# -*- coding: utf-8 -*-
"""race_blind_sensitivity.py — 跨体系适用性敏感性分析。

目的：本文模型含 race/ethnicity 指示列，跨医疗体系（韩国 KNHANES / 中国 CHARLS）无法移植。
本脚本真重训两种变体并给 bootstrap CI：
  (A) 完整模型（含 race_black/hisp/other）—— 自检须复现 0.8099
  (B) race-blind（去 3 个 race 指示列，保留 female）
另报单列消融（已存在于 p1_ml.json，此处仅复核）。

输出：results/race_blind_sensitivity.json
"""
import os, sys, json, argparse
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
import numpy as np, pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

FEATURES = ["age", "sex", "race", "poverty_ratio", "education", "bmi", "sbp", "dbp",
            "hba1c", "diabetes", "total_cholesterol", "hdl"]
CYC = {"G": 2011, "H": 2013, "I": 2015}
HYP = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
           colsample_bytree=0.8, eval_metric="logloss", n_jobs=4, random_state=42)


def load(DATA):
    def nhanes(cyc):
        L = lambda n: pd.read_sas(os.path.join(DATA, f"{n}_{cyc}.XPT"), format="xport")
        dem, alb, bio = L("DEMO"), L("ALB_CR"), L("BIOPRO")
        bpx, bmx, diq, ghb, tc, hd = L("BPX"), L("BMX"), L("DIQ"), L("GHB"), L("TCHOL"), L("HDL")
        x = dem[["SEQN", "RIDAGEYR", "RIAGENDR", "RIDRETH1", "INDFMPIR", "DMDEDUC2",
                 "WTMEC2YR", "RIDEXPRG"]].copy()
        x = x.rename(columns={"RIDAGEYR": "age", "RIAGENDR": "sex", "RIDRETH1": "race",
                              "INDFMPIR": "poverty_ratio", "DMDEDUC2": "education",
                              "WTMEC2YR": "wt", "RIDEXPRG": "pregnant"})
        M = lambda s, c, p: x.merge(s[["SEQN"] + c].rename(columns={v: p + v for v in c}),
                                    on="SEQN", how="left")
        x = M(alb, ["URXUMA", "URXUCR"], "a_"); x = M(bio, ["LBXSCR"], "b_")
        x = M(bpx, ["BPXSY1", "BPXDI1"], "c_"); x = M(bmx, ["BMXBMI"], "d_")
        x = M(diq, ["DIQ010"], "e_"); x = M(ghb, ["LBXGH"], "f_")
        x = M(tc, ["LBXTC"], "g_"); x = M(hd, ["LBDHDD"], "g_")
        x["uacr"] = (x["a_URXUMA"] * 100.0 / x["a_URXUCR"]).where(x["a_URXUCR"] > 0)
        x["creatinine"], x["sbp"], x["dbp"] = x["b_LBXSCR"], x["c_BPXSY1"], x["c_BPXDI1"]
        x["bmi"], x["hba1c"] = x["d_BMXBMI"], x["f_LBXGH"]
        x["total_cholesterol"], x["hdl"] = x["g_LBXTC"], x["g_LBDHDD"]
        x["diabetes"] = np.where(x["e_DIQ010"] == 1, 1.0, np.where(x["e_DIQ010"] == 2, 0.0, np.nan))
        kk = np.where(x["sex"] == 2, 0.7, 0.9); aa = np.where(x["sex"] == 2, -0.241, -0.302)
        ss = x["creatinine"] / kk
        x["egfr"] = (142.0 * np.minimum(ss, 1.0) ** aa * np.maximum(ss, 1.0) ** (-1.200)
                     * 0.9938 ** x["age"] * np.where(x["sex"] == 2, 1.012, 1.0))
        x["ckd"] = ((x["egfr"] < 60) | (x["uacr"] >= 30)).astype(float)
        return x[(x["age"] >= 18) & (x["pregnant"] != 1) & x["ckd"].notna()]
    dev = pd.concat([nhanes(c) for c in CYC], ignore_index=True)
    val = nhanes("J")
    return dev, val


def enc(X, with_race=True):
    X = X.copy()
    if with_race:
        X["race_black"] = (X["race"] == 4).astype(float)
        X["race_hisp"] = X["race"].isin([1, 2]).astype(float)
        X["race_other"] = (~X["race"].isin([1, 2, 3, 4])).astype(float)
    X["female"] = (X["sex"] == 2).astype(float)
    return X.drop(columns=["race", "sex"])


def design(dev, val, with_race=True):
    d, v = enc(dev[FEATURES], with_race), enc(val[FEATURES], with_race)
    med = d.median()
    F = lambda X: pd.concat([X.fillna(med), X.isna().add_prefix("miss_")], axis=1)
    return F(d), F(v)


def boot(p, y, n=1000, seed=20260927):
    rng = np.random.default_rng(seed); out = []
    idx = np.arange(len(y))
    for _ in range(n):
        s = rng.choice(idx, len(idx), replace=True)
        if y[s].sum() in (0, len(s)):
            continue
        out.append(roc_auc_score(y[s], p[s]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev, val = load(a.data)
    y = val["ckd"].values
    res = {}

    # (A) 完整模型（自检门）
    Xtr, Xv = design(dev, val, True)
    mA = XGBClassifier(**HYP).fit(Xtr, dev["ckd"].values)
    pA = mA.predict_proba(Xv)[:, 1]
    aucA = roc_auc_score(y, pA); loA, hiA = boot(pA, y)
    print("(A) 完整模型   AUC=%.4f (%.4f–%.4f)  列数=%d" % (aucA, loA, hiA, Xtr.shape[1]))
    ok = abs(aucA - 0.8099) <= 1e-4
    print("    自检门 AUC==0.8099 ? %s" % ("✅" if ok else "❌ 漂移 → 中止"))
    if not ok:
        sys.exit("❌ 自检门未过，拒绝写入结果（避免污染台账）")
    res["full_model"] = {"auc": round(aucA, 4), "ci": [round(loA, 4), round(hiA, 4)],
                         "n_features_matrix": int(Xtr.shape[1]), "n_test": int(len(y)),
                         "events_test": int(y.sum())}

    # (B) race-blind
    Xtr2, Xv2 = design(dev, val, False)
    mB = XGBClassifier(**HYP).fit(Xtr2, dev["ckd"].values)
    pB = mB.predict_proba(Xv2)[:, 1]
    aucB = roc_auc_score(y, pB); loB, hiB = boot(pB, y)
    print("(B) race-blind AUC=%.4f (%.4f–%.4f)  列数=%d" % (aucB, loB, hiB, Xtr2.shape[1]))
    res["race_blind"] = {"auc": round(aucB, 4), "ci": [round(loB, 4), round(hiB, 4)],
                         "n_features_matrix": int(Xtr2.shape[1])}

    # ΔAUC（配对 bootstrap）
    rng = np.random.default_rng(20260927); deltas = []
    idx = np.arange(len(y))
    for _ in range(1000):
        s = rng.choice(idx, len(idx), replace=True)
        if y[s].sum() in (0, len(s)):
            continue
        deltas.append(roc_auc_score(y[s], pB[s]) - roc_auc_score(y[s], pA[s]))
    d = float(aucB - aucA)
    dlo, dhi = float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))
    print("    ΔAUC(race-blind − full) = %+.4f (%.4f to %+.4f)" % (d, dlo, dhi))
    res["delta_auc_race_blind_minus_full"] = {"value": round(d, 4),
                                              "ci": [round(dlo, 4), round(dhi, 4)]}

    # 单列消融复核（与 p1_ml.json 对账）
    _repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _cand = [os.path.join(_repo, "results", "p1_ml.json"),
             os.path.join(os.path.dirname(os.path.abspath(a.out)), "p1_ml.json")]
    ref = None
    for _p in _cand:
        if os.path.exists(_p):
            ref = json.load(open(_p, encoding="utf-8")); break
    if ref is None:
        print("   ⚠️ 未找到 p1_ml.json（对账项跳过，不影响主结果）：%s" % _cand)
        res["single_column_ablation_reference"] = None
    else:
        ab = ref["results"]["missingness_perturbation"]["single_feature_ablation_delta_auc"]
        res["single_column_ablation_reference"] = {k: ab[k] for k in ("race_black", "race_hisp", "race_other")}
        print("    p1_ml.json 单列消融参照：%s" % res["single_column_ablation_reference"])
        print("    参照合计（单列分别消融之和）=%+.4f → 与联合去种族 %+.4f 的差异说明两者口径不同"
              % (sum(res["single_column_ablation_reference"].values()), d))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("✅ 写出 %s" % a.out)


if __name__ == "__main__":
    main()
