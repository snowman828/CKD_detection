"""S1: dose-response of the age term in the CKD-EPI 2021 equation.

Recompute eGFR while varying only the age exponent, rebuild the CKD label
(eGFR < 60 or uACR >= 30) and count how many albuminuria-blind cases exist
solely because of eGFR. Validates the published 253 -> 25 result first.
"""
import json
import os

import numpy as np
import pandas as pd

BASE = r"G:\项目文件\CKD多模态AI早诊_本项目专属归档\09_无湿实验执行"
c = pd.read_parquet(os.path.join(BASE, "results", "cohort.parquet"))
te = c[c["year"] == 2017].copy()

scr = te["creatinine"].values.astype(float)
age = te["age"].values.astype(float)
female = (te["sex"] == 2).values
uacr = te["uacr"].values.astype(float)


def egfr_2021(exp_age):
    k = np.where(female, 0.7, 0.9)
    a = np.where(female, -0.241, -0.302)
    ratio = scr / k
    return (142.0
            * np.minimum(ratio, 1.0) ** a
            * np.maximum(ratio, 1.0) ** -1.200
            * (exp_age ** age)
            * np.where(female, 1.012, 1.0))


def blind_cases(exp_age):
    e = egfr_2021(exp_age)
    ckd = (e < 60) | (uacr >= 30)
    blind_ckd = ckd & (uacr < 30)
    return {
        "age_exponent": exp_age,
        "n_ckd_total": int(ckd.sum()),
        "n_blind_ckd": int(blind_ckd.sum()),
        "n_ckd_from_egfr_only": int(((e < 60) & (uacr < 30)).sum()),
        "median_egfr_blind_cases": float(np.median(e[blind_ckd])) if blind_ckd.any() else None,
    }


grid = [1.0000, 0.9980, 0.9950, 0.9938, 0.9920, 0.9900, 0.9850]
rows = [blind_cases(x) for x in grid]

# validation against the published analysis
ref = {"n_blind_ckd_2021": 253, "n_blind_ckd_no_age": 25}
row_2021 = next(r for r in rows if abs(r["age_exponent"] - 0.9938) < 1e-9)
row_none = next(r for r in rows if abs(r["age_exponent"] - 1.0) < 1e-9)
chk = {"reproduced_2021": row_2021["n_blind_ckd"] == ref["n_blind_ckd_2021"],
       "reproduced_no_age": row_none["n_blind_ckd"] == ref["n_blind_ckd_no_age"],
       "observed_2021": row_2021["n_blind_ckd"], "observed_no_age": row_none["n_blind_ckd"]}

out = {"design": "test set (2017-2018); label = eGFR<60 or uACR>=30; only the age exponent varies",
       "check": chk, "rows": rows}
path = os.path.join(BASE, "results", "m2", "age_term_dose_response.json")
json.dump(out, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

print("validation:", chk)
print("\nexponent | total CKD | blind cases | median eGFR of blind cases")
for r in rows:
    print("  %.4f | %5d | %5d | %s" % (r["age_exponent"], r["n_ckd_total"],
                                       r["n_blind_ckd"], r["median_egfr_blind_cases"]))
print("\nwritten:", path)
