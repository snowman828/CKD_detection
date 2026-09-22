# -*- coding: utf-8 -*-
"""无湿实验优化补全 · 侦察 + NNS 计算"""
import json, os

# 1. 库可用性
libs = {}
for name in ["lifelines", "sksurv"]:
    try:
        m = __import__(name)
        libs[name] = getattr(m, "__version__", "OK")
    except ImportError:
        libs[name] = None
print("=== 库可用性 ===")
for k, v in libs.items():
    print(f"  {k}: {v}")

# 2. NNS 计算（真源数字）
BASE = r"G:/项目文件/CKD多模态AI早诊_本项目专属归档/09_无湿实验执行"
d = json.load(open(os.path.join(BASE, "results/model_results.json"), encoding="utf-8"))
xgb = d["XGB"]
sens, spec = xgb["sens"], xgb["spec"]
prev = xgb["n_test"] and (1017 / 5801)  # test CKD 1017 / 5801
# 更严谨：从 cohort 拿 test 患病率
import pandas as pd
cohort = pd.read_parquet(os.path.join(BASE, "results/cohort.parquet"))
test = cohort[cohort["year"] == 2017]
prev = float(test["ckd"].mean())
n_test = len(test)

per1000 = 1000
tp = sens * prev * per1000
fp = (1 - spec) * (1 - prev) * per1000
ppv = tp / (tp + fp)
nns = per1000 / tp
ratio = fp / tp

print("\n=== NNS 计算（XGB, Youden 阈值）===")
print(f"  sens={sens:.4f}, spec={spec:.4f}, 患病率={prev:.4f} ({prev*100:.1f}%), n_test={n_test}")
print(f"  每 1000 人筛查: 真阳性={tp:.1f}, 假阳性={fp:.1f}, 阳性总数={tp+fp:.1f}")
print(f"  PPV={ppv:.3f} (校验, 应≈0.374)")
print(f"  NNS(检出 1 真阳性) = {nns:.1f} 人")
print(f"  假阳性:真阳性比 = {ratio:.2f}")
