# -*- coding: utf-8 -*-
"""m13 —— 投稿版清理：剥离草稿注记 + Table 1 从 CSV 程序化生成（红线：禁止手工抄写）
输出：MACKI_Followup_Mechanistic_CLEAN.md → 供 md2docx 转换
"""
import pandas as pd, re, os

BASE = r"G:/项目文件/CKD多模态AI早诊_本项目专属归档"
MD = os.path.join(BASE, "09_无湿实验执行/论文/MACKI_Followup_Mechanistic_DRAFT_v1.md")
OUT_MD = os.path.join(BASE, "09_无湿实验执行/论文/MACKI_Followup_Mechanistic_CLEAN.md")
CSV = os.path.join(BASE, "09_无湿实验执行/results/m2/table1_baseline.csv")

# ---- Table 1 从 CSV 程序化生成 ----
t1 = pd.read_csv(CSV)
header = list(t1.columns)
md_rows = ["| " + " | ".join(header) + " |",
           "|" + "---|" * len(header)]
for _, r in t1.iterrows():
    md_rows.append("| " + " | ".join(str(r[c]) for c in header) + " |")
table1_md = "\n".join(md_rows)
print(f"Table 1 生成: {len(t1)} 行")

# ---- 读 md，清理 ----
text = open(MD, encoding="utf-8").read()
lines = text.split("\n")

# 1) 删除顶部 draft 注记（Draft v1.0 / Status 行 + 它们的引用行）
clean = []
skip_next_ref = False
for ln in lines:
    if ln.startswith("**Draft v1.0") or ln.startswith("**Status"):
        continue
    if ln.strip().startswith("> 用途") or ln.strip().startswith("> 日期") or \
       ln.strip().startswith("> 关联") or ln.strip().startswith("> 数据") or \
       ln.strip().startswith("> 判定") or ln.strip().startswith("> 全指标"):
        continue
    if "Draft v1.0" in ln or "graphical abstract pending" in ln.lower():
        continue
    clean.append(ln)
text = "\n".join(clean)

# 2) Table 1 段替换为真实表
pat = re.compile(r"### Table 1\..*?(?=\n### |\n---|\Z)", re.S)
if not pat.search(text):
    print("WARN: 未找到 Table 1 段——检查 md 结构")
else:
    text = pat.sub(f"### Table 1. Baseline characteristics (NHANES 2017\u201318 test set; n=5,801)\n\n{table1_md}\n", text)

# 3) 更新顶部状态行（投稿版元信息）
text = text.replace("# Can Non-Kidney-Specific Routine Clinical Data Encode an Interpretable CKM Footprint?",
                    "# Can Non-Kidney-Specific Routine Clinical Data Encode an Interpretable CKM Footprint?\n\n**Submission draft v1 — 2026-08-29**")
# 4) 移除 [SOURCE]/[INFERENCE]/[UNVERIFIED] 内部标注（投稿版不留协议标签，保留 DOI 引用）
text = re.sub(r"\[SOURCE,?\s*", "[", text)
text = re.sub(r"\[INFERENCE,?\s*", "[", text)
text = re.sub(r"\s*\[SOURCE\]", "", text)
text = re.sub(r"\s*\[INFERENCE\]", "", text)
text = text.replace("synthesised from 10.1038/s41467-025-62273-0, 10.1038/s43856-023-00278-w", "synthesised from refs 5, 6")

with open(OUT_MD, "w", encoding="utf-8") as f:
    f.write(text)
print(f"已保存: {OUT_MD} | {os.path.getsize(OUT_MD)} bytes")
# 校验 Table 1 已插入
print("Table 1 在位:", "Baseline characteristics" in text and "SMD" in text)
