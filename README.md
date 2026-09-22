# MACKI-Dry — reproducible analysis pipeline

**Detection of chronic kidney disease from routine, non-kidney-specific clinical data (NHANES 2011–2018):
development and external temporal validation of machine-learning models.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Public repository**: https://github.com/snowman828/CKD_detection

This repository contains the complete, reproducible analysis pipeline for the study. The corresponding manuscript is
in submission; the citation will be completed on publication (see `CITATION.cff`).

**Key result**: XGBoost achieved AUC 0.810 (95% CI 0.796–0.823) for CKD detection in a completely unseen NHANES cycle
(2017–2018; n=5,801), using only non-kidney-specific routine clinical variables; the development cycles were
2011–2016 (n=17,777).

## What this repository provides (CJASN data/code expectation)
1. **Novel computer script** — the full pipeline (`scripts/`).
2. **Trained machine-learning models** — **not committed as binaries**: models are re-fitted deterministically by
   `scripts/modeling.py` (fixed seeds are set in the scripts). The repository therefore provides the exact code path
   that produces the model, not a pre-trained binary.
3. **Source data used for performance evaluation** — public-use NHANES files; see `DATA_AVAILABILITY.md` for the
   official download locations (this repository: https://github.com/snowman828/CKD_detection) and for the scripts that rebuild the analysis cohort.

## Pipeline
```bash
python scripts/download_nhanes.py    # NHANES download (4 cycles + mortality linkage)
python scripts/build_cohort.py       # cohort construction, CKD-EPI 2021 eGFR, KDIGO labels, anti-leakage feature set
python scripts/modeling.py           # logistic regression / XGBoost / MLP + external temporal validation
python scripts/m9_table1.py          # Table 1 (cohort characteristics)
python scripts/m10_table2.py         # Table 2 (model performance)
python scripts/m2_substudy12.py      # sub-studies S1–S2
python scripts/m3_substudy3.py       # sub-study S3 (spectrum attribution)
python scripts/m5_substudy4.py       # sub-study S4 (poverty-income gradient, entropy balancing)
python scripts/m4_substudy5b.py      # sub-study S5-B (mortality, exploratory)
python scripts/m6_figure1.py        # Figure 1
python scripts/m7_figure2.py        # Figure 2
python scripts/m8_figure3.py        # Figure 3
```

## Environment
```bash
python -m venv .venv && source .venv/Scripts/activate   # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```
`requirements.txt` was derived from the actual imports in `scripts/` (see the file header).

## Repository layout
```
scripts/          analysis pipeline (31 Python files)
results/          machine-readable result summaries (JSON); parquet data files are git-ignored
analysis_plan/    pre-specified analysis plan / pre-registration material
requirements.txt  dependencies (derived from imports)
DATA_AVAILABILITY.md  data sources, ethics facts, what is not redistributed
PUSH_GUIDE.md     (Chinese) how the author pushes this local repository to GitHub
CITATION.cff      citation metadata (fill in ORCID/DOI/repository URL)
LICENSE           MIT
```

## Ethics
NHANES protocols were reviewed and approved by the NCHS Research Ethics Review Board (renamed the NCHS Ethics Review
Board in 2018; Protocols #2011-17 and #2018-01) and documented informed consent was obtained from all participants.
This secondary analysis used de-identified public-use files and required no additional ethics approval.

## Licence
MIT (see `LICENSE`).
