"""引用编号最终修复（2026-08-28 v2）
目标 Vancouver 顺序（正文首次出现）：1=Jager 2=GBD 3=WHA 4=ZhangL 5=Kovesdy 6=Inker
  7=KDIGO 8=Shi 9=Chu 10=Sabanayagam 11=ZhangK 12=KIDS 13=DeepDKD 14=Holmstrom 15=TRIPOD 16=CDC
当前（被 m² 误伤偏移）：Jager=2 GBD=3 WHA=1 ZhangL=4 Kovesdy=5 6-14 正确 TRIPOD=15 CDC=17
"""
SRC = r"G:/项目文件/CKD多模态AI早诊_本项目专属归档/09_无湿实验执行/论文/MACKI_Dry_Manuscript_FINAL_投稿完整版.md"
md = open(SRC, encoding='utf-8').read()
head, refs_part = md.split('## References')

# 保护单位符号
head = head.replace('m²', '@@M2@@').replace('m³', '@@M3@@')

# 先保护所有两位上标（¹⁰-¹⁷），防止单字符替换拆解（2026-08-28 教训：¹³ 被拆成 ³²）
for n in range(10, 18):
    sup = ''.join({'0':'⁰','1':'¹','2':'²','3':'³','4':'⁴','5':'⁵','6':'⁶','7':'⁷','8':'⁸','9':'⁹'}[c] for c in str(n))
    head = head.replace(sup, f'@@T{n}@@')

# 占位符替换（防链式冲突）——此时仅剩单位数上标
head = head.replace('¹', '@@WHA@@')    # WHA（当前 1）
head = head.replace('²', '@@JAG@@')    # Jager（当前 2）
head = head.replace('³', '@@GBD@@')    # GBD（当前 3）
# 还原为目标编号
head = head.replace('@@JAG@@', '¹')    # Jager → 1
head = head.replace('@@GBD@@', '²')    # GBD → 2
head = head.replace('@@WHA@@', '³')    # WHA → 3
for n in range(10, 18):
    sup = ''.join({'0':'⁰','1':'¹','2':'²','3':'³','4':'⁴','5':'⁵','6':'⁶','7':'⁷','8':'⁸','9':'⁹'}[c] for c in str(n))
    head = head.replace(f'@@T{n}@@', sup)  # 10-15 原样；17→16 在下一步
head = head.replace('¹⁷', '¹⁶')   # CDC → 16
# 恢复单位
head = head.replace('@@M2@@', 'm²').replace('@@M3@@', 'm³')

# References 重建（16 条目标顺序）
REFS = [
 "Jager KJ, Kovesdy C, Langham R, et al. A single number for advocacy and communication—worldwide more than 850 million individuals have kidney diseases. *Kidney Int.* 2019;96(5):1048–1050. doi:10.1016/j.kint.2019.07.012",
 "GBD Chronic Kidney Disease Collaboration. Global, regional, and national burden of chronic kidney disease, 1990–2017: a systematic analysis for the Global Burden of Disease Study 2017. *Lancet.* 2020;395(10225):709–733. doi:10.1016/S0140-6736(20)30045-3",
 "World Health Organization. Resolution WHA78.6: Chronic kidney disease as a global non-communicable disease priority. Seventy-eighth World Health Assembly, Geneva; May 2025.",
 "Zhang L, Wang F, Wang L, et al. Prevalence of chronic kidney disease in China: a cross-sectional survey. *Lancet.* 2012;379(9818):815–822. doi:10.1016/S0140-6736(12)60033-6",
 "Kovesdy CP. Epidemiology of chronic kidney disease: an update 2022. *Kidney Int Suppl.* 2022;12(1):7–11. doi:10.1016/j.kisu.2021.11.003",
 "Inker LA, Eneanya ND, Coresh J, et al. New creatinine- and cystatin C–based equations to estimate GFR without race. *N Engl J Med.* 2021;385(19):1737–1749. doi:10.1056/NEJMoa2102953",
 "Kidney Disease: Improving Global Outcomes (KDIGO) CKD Work Group. KDIGO 2024 clinical practice guideline for the evaluation and management of chronic kidney disease. *Kidney Int.* 2024;105(4S):S117–S314. doi:10.1016/j.kint.2023.10.018",
 "Shi S, Gao L, Wu X. Comparison of nonalbuminuric and albuminuric diabetic kidney disease among patients with type 2 diabetes: a systematic review and meta-analysis. *Front Endocrinol (Lausanne).* 2022;13:871272. doi:10.3389/fendo.2022.871272",
 "Chu CD, Xia F, Du Y, et al. Estimated prevalence and testing for albuminuria in US adults at risk for chronic kidney disease. *JAMA Netw Open.* 2023;6(7):e2326230. doi:10.1001/jamanetworkopen.2023.26230",
 "Sabanayagam C, Xu D, Ting DSW, et al. A deep learning algorithm to detect chronic kidney disease from retinal photographs in community-based populations. *Lancet Digit Health.* 2020;2(6):e295–e302. doi:10.1016/S2589-7500(20)30063-7",
 "Zhang K, Liu X, Wang G, et al. Deep-learning models for the detection and incidence prediction of chronic kidney disease and type 2 diabetes from retinal fundus images. *Nat Biomed Eng.* 2021;5(6):570–580. doi:10.1038/s41551-021-00745-6",
 "Wu Q, Li J, Zhao L, et al. A noninvasive model for chronic kidney disease screening and common pathological type identification from retinal images. *Nat Commun.* 2025;16:6962. doi:10.1038/s41467-025-62273-0",
 "Meng Z, Guan Z, Yu S, et al. Non-invasive biopsy diagnosis of diabetic kidney disease via deep learning applied to retinal images: a population-based study. *Lancet Digit Health.* 2025;7(5):e100868. doi:10.1016/j.landig.2025.02.008",
 "Holmstrom L, Christensen M, Yuan N, et al. Deep learning-based electrocardiographic screening for chronic kidney disease. *Commun Med (Lond).* 2023;3:73. doi:10.1038/s43856-023-00278-w",
 "Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. *BMJ.* 2024;385:q902. doi:10.1136/bmj.q902",
 "Centers for Disease Control and Prevention. Chronic Kidney Disease in the United States, 2023. Atlanta, GA: US Department of Health and Human Services, CDC; 2023.",
]
new_refs = "## References\n\n" + "\n".join(f"{i+1}. {t}" for i, t in enumerate(REFS)) + "\n"

open(SRC, 'w', encoding='utf-8').write(head + new_refs)

# 验证
SUP = {'⁰':'0','¹':'1','²':'2','³':'3','⁴':'4','⁵':'5','⁶':'6','⁷':'7','⁸':'8','⁹':'9'}
def sup2num(s): return int(''.join(SUP[c] for c in s))
seen = []
for m in __import__('re').finditer(r'[⁰¹²³⁴⁵⁶⁷⁸⁹]+', head):
    n = sup2num(m.group(0))
    if n not in seen: seen.append(n)
print('正文首次出现顺序:', seen)
assert seen == list(range(1, 17)), "编号仍不正确！"
assert 'm¹' not in head and 'm²' in head, "单位符号未恢复！"
assert 'Sabanayagam¹⁰' in head, "Sabanayagam 编号错误！"
print('✅ 全部通过：16 条引用 Vancouver 顺序 1→16，单位符号完好，Sabanayagam=10，CDC=16')
