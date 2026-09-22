"""M8 Figure 3 —— 病例严重度谱构成堆叠图（子研究③ 结构证据）
每年龄组病例 100% 堆叠：albuminuria-only / g3a(45-60) / g3b(<45)
"""
import pandas as pd, numpy as np, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = r"G:/项目文件/CKD多模态AI早诊_本项目专属归档/09_无湿实验执行/results/m2"
cohort = pd.read_parquet(r"G:/项目文件/CKD多模态AI早诊_本项目专属归档/09_无湿实验执行/results/cohort.parquet")
test = cohort[cohort["year"] == 2017].reset_index(drop=True)
test["sev"] = np.where(test["egfr"] < 45, "g3b (<45)",
             np.where(test["egfr"] < 60, "g3a (45–60)",
             np.where(test["uacr"] >= 30, "albuminuria-only", "none")))

groups = [("18–44", test["age"] < 45),
          ("45–64", (test["age"] >= 45) & (test["age"] < 65)),
          ("≥65", test["age"] >= 65)]
cats = ["albuminuria-only", "g3a (45–60)", "g3b (<45)"]
COLORS = {"albuminuria-only": "#F5B942", "g3a (45–60)": "#E8890C", "g3b (<45)": "#C0392B"}

props = []
for name, mask in groups:
    cases = test.loc[mask.values & (test["ckd"] == 1)]
    tot = len(cases)
    row = [ (cases["sev"] == c).sum() / tot if tot else 0 for c in cats ]
    props.append(row)
    print(f"{name}: n_cases={tot} | " + ", ".join(f"{c}={row[i]:.2f}" for i, c in enumerate(cats)))

props = np.array(props)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.7})
fig, ax = plt.subplots(figsize=(4.4, 3.2))
x = np.arange(len(groups))
bottom = np.zeros(len(groups))
for ci, c in enumerate(cats):
    ax.bar(x, props[:, ci], bottom=bottom, color=COLORS[c], width=0.55,
           label=c, edgecolor="white", linewidth=0.6)
    for xi, v in enumerate(props[:, ci]):
        if v > 0.03:
            ax.text(xi, bottom[xi] + v / 2, f"{v:.0%}", ha="center", va="center",
                    fontsize=7.5, color="white", fontweight="bold")
    bottom += props[:, ci]
ax.set_xticks(x); ax.set_xticklabels([g[0] for g in groups])
ax.set_xlabel("Age group (NHANES 2017–18 CKD cases)")
ax.set_ylabel("Proportion of CKD cases")
ax.set_ylim(0, 1.02)
ax.legend(loc="upper right", frameon=False, fontsize=7, title="Case severity spectrum", title_fontsize=7)
ax.set_title("Case severity composition by age — 91% of young-adult CKD is albuminuria-only", loc="left", fontsize=8.5, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "figure3_spectrum.png"), dpi=300, bbox_inches="tight")
fig.savefig(os.path.join(OUT, "figure3_spectrum.pdf"), bbox_inches="tight")
print("已保存: figure3_spectrum.png / .pdf")
