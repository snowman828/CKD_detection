"""引用核验（reference-verification 技能落地）：CrossRef 按题名检索，输出核验报告
每条引用 → query.bibliographic 检索 → 最佳匹配比对（题名/作者/期刊/年份/DOI）
"""
import json, time, urllib.request, urllib.parse

REFS = [
    {"id": 1,  "title": "Global regional and national burden of chronic kidney disease 1990 to 2021", "note": "GBD CKD 2024 Lancet, 8.5亿"},
    {"id": 3,  "title": "Prevalence of chronic kidney disease in China a cross-sectional survey", "note": "Zhang L Lancet 2012"},
    {"id": 4,  "title": "New creatinine- and cystatin C-based equations to estimate GFR without race", "note": "Inker NEJM 2021"},
    {"id": 5,  "title": "KDIGO 2024 clinical practice guideline for the evaluation and management of chronic kidney disease", "note": "Kidney Int 2024"},
    {"id": 6,  "title": "A noninvasive model for chronic kidney disease screening and common pathological type identification from retinal images", "note": "KIDS Nat Commun 2025"},
    {"id": 7,  "title": "Non-invasive biopsy diagnosis of diabetic kidney disease via deep learning applied to retinal images", "note": "DeepDKD Lancet Digit Health 2025"},
    {"id": 8,  "title": "A deep learning algorithm to detect chronic kidney disease from retinal photographs in community-based populations", "note": "Sabanayagam 2020"},
    {"id": 9,  "title": "Deep-learning models for the detection of chronic kidney disease from retinal photographs", "note": "Zhang K Nat Biomed Eng 2021"},
    {"id": 10, "title": "12-lead electrocardiogram deep learning chronic kidney disease screening", "note": "Holmstrom Commun Med 2023"},
    {"id": 11, "title": "A clinically applicable approach to continuous prediction of future acute kidney injury", "note": "Tomasev Nature 2019"},
    {"id": 12, "title": "TRIPOD+AI statement updated guidance for reporting clinical prediction models", "note": "Collins BMJ 2024"},
    {"id": 14, "title": "non-albuminuric diabetic kidney disease prevalence meta-analysis", "note": "NADKD 荟萃 45.6%"},
    {"id": 15, "title": "urine albumin-to-creatinine ratio testing rates primary care chronic kidney disease", "note": "uACR 检测率"},
]

def crossref_query(q, rows=3):
    url = "https://api.crossref.org/works?" + urllib.parse.urlencode(
        {"query.bibliographic": q, "rows": rows, "select": "title,author,container-title,issued,DOI,volume,page,type"})
    req = urllib.request.Request(url, headers={"User-Agent": "MACKI-RefVerifier/1.0 (mailto:research@example.org)"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)["message"]["items"]

report = []
for ref in REFS:
    try:
        items = crossref_query(ref["title"])
    except Exception as e:
        report.append({"id": ref["id"], "query": ref["title"], "error": str(e)})
        continue
    top = []
    for it in items[:3]:
        title = (it.get("title") or ["N/A"])[0]
        authors = ", ".join(f"{a.get('given','')} {a.get('family','')}".strip()
                            for a in it.get("author", [])[:4]) or "N/A"
        year = None
        for f in ["published-print", "published-online", "issued"]:
            dp = it.get(f, {}).get("date-parts", [[]])
            if dp and dp[0]: year = dp[0][0]; break
        top.append({"title": title[:110], "authors": authors, "journal": (it.get("container-title") or ["N/A"])[0],
                    "year": year, "doi": it.get("DOI"), "volume": it.get("volume"), "page": it.get("page")})
    report.append({"id": ref["id"], "query": ref["title"], "note": ref["note"], "candidates": top})
    time.sleep(0.4)

print(json.dumps(report, ensure_ascii=False, indent=1))
with open(r"G:/项目文件/CKD多模态AI早诊_本项目专属归档/09_无湿实验执行/results/ref_verification_raw.json", "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=1)
