"""M9 Table 1 —— 基线特征表（CKD vs 非 CKD，含 SMD，NHANES 2017-18 测试集）
"""
import pandas as pd, numpy as np, os, json

OUT = r"G:/项目文件/CKD多模态AI早诊_本项目专属归档/09_无湿实验执行/results/m2"
cohort = pd.read_parquet(r"G:/项目文件/CKD多模态AI早诊_本项目专属归档/09_无湿实验执行/results/cohort.parquet")
te = cohort[cohort["year"] == 2017].reset_index(drop=True)

CONT = ["age","bmi","sbp","dbp","hba1c","total_cholesterol","hdl","poverty_ratio"]
CAT = [("sex", {1: "Male", 2: "Female"}), ("race", {1: "Mexican American", 2: "Other Hispanic",
        3: "Non-Hispanic White", 4: "Non-Hispanic Black", 5: "Other"}),
       ("education", {0: "<9th grade", 1: "9–11th grade", 2: "HS/GED", 3: "Some college", 4: "College+"}),
       ("diabetes", {0: "No", 1: "Yes"})]

def smd_cont(a, b):
    m1, s1, m2, s2 = a.mean(), a.std(ddof=1), b.mean(), b.std(ddof=1)
    return (m1 - m2) / np.sqrt((s1**2 + s2**2) / 2)

rows = []; results = {"n_ckd": int((te["ckd"] == 1).sum()), "n_no_ckd": int((te["ckd"] == 0).sum())}
print(f"n(CKD)={results['n_ckd']} | n(no CKD)={results['n_no_ckd']}")
print(f"{'Variable':<22}{'CKD':>14}{'No CKD':>14}{'SMD':>8}")

for c in CONT:
    a = te.loc[te["ckd"] == 1, c]; b = te.loc[te["ckd"] == 0, c]
    s = smd_cont(a, b)
    rows.append([c, f"{a.mean():.1f} ({a.std(ddof=1):.1f})", f"{b.mean():.1f} ({b.std(ddof=1):.1f})", f"{s:+.3f}"])
    print(f"{c:<22}{a.mean():>8.1f} ({a.std(ddof=1):.1f}){b.mean():>8.1f} ({b.std(ddof=1):.1f}){s:>+8.3f}")

for col, lab in CAT:
    a = te.loc[te["ckd"] == 1, col]; b = te.loc[te["ckd"] == 0, col]
    for k, v in lab.items():
        pa, pb = (a == k).mean(), (b == k).mean()
        s = (pa - pb) / np.sqrt((pa*(1-pa) + pb*(1-pb)) / 2)
        nm = f"{col}={v}"
        rows.append([nm, f"{pa:.1%}", f"{pb:.1%}", f"{s:+.3f}"])
        print(f"{nm:<22}{pa:>13.1%}{pb:>13.1%}{s:>+8.3f}")

pd.DataFrame(rows, columns=["Variable", "CKD (n=%d)" % results["n_ckd"], "No CKD (n=%d)" % results["n_no_ckd"], "SMD"]).to_csv(
    os.path.join(OUT, "table1_baseline.csv"), index=False)
with open(os.path.join(OUT, "table1_baseline.json"), "w", encoding="utf-8") as f:
    json.dump({"rows": rows, **results}, f, ensure_ascii=False, indent=2)
print("已保存: table1_baseline.csv / .json")
