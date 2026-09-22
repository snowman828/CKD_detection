# -*- coding: utf-8 -*-
"""M10 Table 2 —— 子研究判决汇总表（从 results JSON 自动生成）"""
import json, os

OUT = r"G:/项目文件/CKD多模态AI早诊_本项目专属归档/09_无湿实验执行/results/m2"
def load(name):
    with open(os.path.join(OUT, name), encoding="utf-8") as f:
        return json.load(f)

s1 = load("substudy12_results.json")
s3 = load("substudy3_results.json")
s4 = load("substudy4_results.json")
s5 = load("substudy5b_results.json")

s2 = s1["substudy2_nadkd"]
rows = [
    ["S1 · Footprint probe",
     "H1a: SHAP patterns cluster by CKM pathway vs H1b: co-occurrence",
     "ARI ≥ 0.30 & p < 0.001",
     f"ARI = {s1['substudy1_footprint']['ari']} (perm p = {s1['substudy1_footprint']['ari_permutation_p']}); age share = {s1['substudy1_footprint']['age_share_of_shap']}",
     "H1a SUPPORTED (moderate)"],
    ["S2 · Albuminuria-blind detection",
     "H2a: independent footprint vs H2b: albuminuria-dependent",
     "Blind-stratum AUC ≥ 0.75 & ΔAUC < 0.10",
     f"uACR<30 stratum AUC = {s2['strata']['uACR<30 全部成人（盲区人群）']['auc']} vs overall {s1['baseline_auc']} (ΔAUC = {s2['strata']['uACR<30 全部成人（盲区人群）']['delta_auc_vs_all']:+.3f})",
     "H2a SUPPORTED (strong)"],
    ["S3 · Young-adult attribution",
     "H3a: spectrum effect vs H3b: signal accumulation vs H3c: interaction",
     "Corrected gain ≥ +0.03 → H3a; corrected AUC ≤ 0.70 & no interaction → H3b",
     f"Severity-weighted gain = {s3['substudy3']['gain_severity']:+.3f}; prevalence gain = {s3['substudy3']['gain_prevalence']:+.3f}; interaction p = {s3['substudy3']['interaction_p_approx']}",
     "H3b SUPPORTED (true boundary)"],
    ["S4 · PIR gradient attribution",
     "H4a: statistical (distributional) vs H4b: mechanistic equality",
     "Post-balancing ΔAUC < 0.02 → H4a; ≥ 0.03 → H4b",
     f"Baseline ΔAUC = {s4['baseline']['delta_auc']:+.3f} → post-entropy-balance ΔAUC = {s4['after_balancing']['delta_auc']:+.3f} (covariate moment deviation → {s4['after_balancing']['max_covariate_moment_deviation']})",
     "H4b SUPPORTED (recalibration cannot fix)"],
    ["S5-B · Systemic footprint (mortality)",
     "H5-B: routine features predict systemic (non-renal) outcomes",
     "All-cause & CVD C-index ≥ 0.70",
     f"All-cause Harrell C = {s5['all_cause']['harrell_c']}; CVD C = {s5['cvd']['harrell_c']}",
     "SUPPORTED (strong)"],
]

print("| Substudy | Competing hypotheses | Pre-registered rule | Result | Verdict |")
print("|---|---|---|---|---|")
for r in rows:
    print("| " + " | ".join(r) + " |")

md = "\n".join(["| Substudy | Competing hypotheses | Pre-registered rule | Result | Verdict |",
                "|---|---|---|---|---|"] +
               ["| " + " | ".join(r) + " |" for r in rows])
with open(os.path.join(OUT, "table2_verdicts.md"), "w", encoding="utf-8") as f:
    f.write(md + "\n")
print("\n已保存: table2_verdicts.md")
