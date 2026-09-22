# -*- coding: utf-8 -*-
"""M12 文献元数据补全 —— CrossRef(DOI) + PubMed esummary(PMID)
目标：论文底稿 References 5-6（KIDS/DeepDKD，有 DOI）与 13-16（4 条 PMID）
输出：Vancouver 完整格式（完整标题 + 作者前3+et al. + 期刊 + 卷(期):页码 + DOI/PMID）
"""
import json, urllib.request, time

DOIS = ["10.1038/s41467-025-62273-0", "10.1016/j.landig.2025.02.008"]
PMIDS = ["41938379", "40359732", "27334381", "27401013"]

def crossref(doi):
    req = urllib.request.Request(f"https://api.crossref.org/works/{doi}",
                                 headers={"User-Agent": "macki-followup/1.0 (mailto:research@example.org)"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                m = json.loads(r.read())["message"]
            auth = m.get("author", [])
            authors = ", ".join([f"{a.get('family','')} {a.get('given','')}" for a in auth[:3]]) + (" et al." if len(auth) > 3 else "")
            title = (m.get("title") or [""])[0]
            jr = (m.get("container-title") or [""])[0]
            vol, iss, page = m.get("volume",""), m.get("issue",""), m.get("page","")
            year = (m.get("issued", {}).get("date-parts", [[None]])[0][0])
            return {"doi": doi, "authors": authors, "title": title, "journal": jr,
                    "volume": vol, "issue": iss, "pages": page, "year": year}
        except Exception as e:
            if attempt == 2: return {"doi": doi, "error": str(e)[:120]}
            time.sleep(3)

def pubmed(pid):
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    try:
        with urllib.request.urlopen(f"{base}/esummary.fcgi?db=pubmed&id={pid}&retmode=json", timeout=30) as r:
            d = json.loads(r.read())["result"][pid]
        auth = d.get("authors", [])
        authors = ", ".join([a["name"] for a in auth[:3]]) + (" et al." if len(auth) > 3 else "")
        title = d.get("title", "")
        jr = d.get("fulljournalname", "")
        vol, iss, pg, yr = d.get("volume",""), d.get("issue",""), d.get("pages",""), d.get("pubdate","")[:4]
        return {"pmid": pid, "authors": authors, "title": title, "journal": jr,
                "volume": vol, "issue": iss, "pages": pg, "year": yr}
    except Exception as e:
        return {"pmid": pid, "error": str(e)[:120]}

out = {"crossref": [crossref(d) for d in DOIS], "pubmed": [pubmed(p) for p in PMIDS]}
print("=== CrossRef（DOI）===")
for r in out["crossref"]:
    print(json.dumps(r, ensure_ascii=False))
print("=== PubMed（PMID）===")
for r in out["pubmed"]:
    print(json.dumps(r, ensure_ascii=False))

print("\n=== Vancouver 完整格式 ===")
for r in out["crossref"] + out["pubmed"]:
    if "error" in r: print("ERR:", r); continue
    if "doi" in r:
        line = (f"{r['authors']}. {r['title']}. {r['journal']} {r['year']}"
                + (f";{r['volume']}" if r['volume'] else "")
                + (f"({r['issue']})" if r['issue'] else "")
                + (f":{r['pages']}" if r['pages'] else "")
                + f". doi:{r['doi']}")
    else:
        line = (f"{r['authors']}. {r['title']}. {r['journal']} {r['year']}"
                + (f";{r['volume']}" if r['volume'] else "")
                + (f"({r['issue']})" if r['issue'] else "")
                + (f":{r['pages']}" if r['pages'] else "")
                + f". PMID:{r['pmid']}")
    print(line)

with open(r"G:/项目文件/CKD多模态AI早诊_本项目专属归档/09_无湿实验执行/results/m2/refs_metadata.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print("\n已保存: refs_metadata.json")
