# -*- coding: utf-8 -*-
"""Graphical Abstract 生成（agnes-image-2.1-flash via agnes 聚合 API）
工程级提示词：逐元素布局 + CKM 通路色值 + 负向禁令
"""
import yaml, os, json, base64, urllib.request, sys

cfg = yaml.safe_load(open(os.path.expanduser('~/AppData/Local/hermes/config.yaml'), encoding='utf-8'))
a = cfg['providers']['agnes']
KEY, BASE = a['api_key'], a['api']

PROMPT = """Professional medical graphical abstract for a top-tier journal (Nature-series style), landscape orientation, left-to-right narrative flow, flat 2.5D vector illustration, clean white background with very light cool-gray gradient, no clutter.

LEFT THIRD — INPUT: non-kidney-specific routine clinical data. A subtle human silhouette in light gray outline. Four floating source icons arranged vertically around it: (1) blood pressure cuff with small heart pulse line [color #D62728 red], (2) glucose meter with droplet [color #FF7F0E orange], (3) lipid panel tube with cholesterol molecule [color #1F77B4 blue], (4) demographic identity card with person glyph [color #7F7F7F gray]. A thin label-less arrow streams these icons rightward into the model.

CENTER THIRD — MODEL: a minimal decision-tree ensemble / neural network node cluster in cool white-blue (#EAF2FA), with an abstract fingerprint motif emerging from it: a rounded fingerprint outline whose ridge loops are colored in the four pathway colors (red hemodynamic, orange glycotoxic, blue metabolic, gray demographic), representing the CKM (cardiovascular-kidney-metabolic) pathophysiological footprint encoded from routine data.

RIGHT THIRD — OUTCOME: split into two connected panels. (a) Kidney: left kidney healthy smooth surface, right kidney with subtle granular texture and small warning dots, separated by a thin vertical divider; above them a small detection score badge showing "AUC 0.865" in dark gray with a green up-arrow (albuminuria-blind population outperforms). (b) Heart: clean heart icon with small ECG line, above it a badge "C-index 0.85" with a heart-rate symbol. A soft rounded bracket links kidney and heart panels labeled implicitly by a small CKM acronym badge (cardiovascular-kidney-metabolic) in muted teal.

STYLE: flat professional medical illustration, consistent soft shadows, muted academic palette (red #D62728, orange #FF7F0E, blue #1F77B4, gray #7F7F7F, teal #2CA6A4, white background), high resolution, crisp edges, publication quality.

NEGATIVE PROHIBITIONS: absolutely no readable text except the two numeric badges; no garbled letters; no handwriting; no cartoon faces; no comic style; no photorealism; no 3D render gloss; no drop-shadow text; no red-alert alarm aesthetics; no medical crosses; no clutter; no watermark; no logo."""

payload = {"model": "agnes-image-2.1-flash", "prompt": PROMPT, "n": 1,
           "size": "1536x1024", "response_format": "b64_json"}
req = urllib.request.Request(BASE + "/images/generations",
                             data=json.dumps(payload).encode(),
                             headers={"Authorization": "Bearer " + KEY,
                                      "Content-Type": "application/json"})
try:
    with urllib.request.urlopen(req, timeout=600) as r:
        data = json.loads(r.read())
    b64 = data["data"][0]["b64_json"]
    out = r"G:/项目文件/CKD多模态AI早诊_本项目专属归档/09_无湿实验执行/results/m2/graphical_abstract_v1.png"
    with open(out, "wb") as f:
        f.write(base64.b64decode(b64))
    print("SAVED:", out, os.path.getsize(out), "bytes")
except Exception as e:
    print("ERR:", e)
    sys.exit(1)
