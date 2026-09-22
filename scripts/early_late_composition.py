# -*- coding: utf-8 -*-
"""「早期 vs 晚期」CKD 病例构成分析（研究必要性补强）
检测到的现患 CKD 中，多少是「早期可干预」（仅白蛋白尿、肾功能尚可 A2+），多少是「较晚」（eGFR<60）。
这是回应「检测已发生的 CKD 有无临床价值」的关键论证。
"""
import pandas as pd, os

BASE = r"G:/项目文件/CKD多模态AI早诊_本项目专属归档/09_无湿实验执行"
cohort = pd.read_parquet(os.path.join(BASE, "results/cohort.parquet"))
test = cohort[cohort["year"] == 2017]
ckd = test["ckd"] == 1
egfr_low = test["egfr"] < 60
uacr_high = test["uacr"] >= 30

n_ckd = int(ckd.sum())
n_alb_only = int((ckd & uacr_high & ~egfr_low).sum())   # 仅白蛋白尿（肾功能尚可）
n_egfr_only = int((ckd & egfr_low & ~uacr_high).sum())  # 仅 eGFR<60
n_both = int((ckd & egfr_low & uacr_high).sum())         # 两者都有
n_test = len(test)

print(f"test n={n_test}, CKD n={n_ckd} ({n_ckd/n_test*100:.1f}%)")
print(f"  仅白蛋白尿型（A2+，eGFR≥60，早期可干预）: {n_alb_only} ({n_alb_only/n_ckd*100:.1f}%)")
print(f"  仅 eGFR 型（G3+，uACR<30，较晚）: {n_egfr_only} ({n_egfr_only/n_ckd*100:.1f}%)")
print(f"  两者都有（A2+ 且 G3+）: {n_both} ({n_both/n_ckd*100:.1f}%)")
print(f"  校验: {n_alb_only}+{n_egfr_only}+{n_both}={n_alb_only+n_egfr_only+n_both} (应={n_ckd})")

# 早期可干预（含白蛋白尿的所有，即 A2+，无论 eGFR）= 仅白蛋白尿 + 两者
n_early = n_alb_only + n_both
print(f"\n  含白蛋白尿（A2+）的所有病例（早期损伤标志）: {n_early} ({n_early/n_ckd*100:.1f}%)")
print(f"  → 检测到的 CKD 中 {n_early/n_ckd*100:.0f}% 携带白蛋白尿（A2+），这是可干预的早期损伤标志")
