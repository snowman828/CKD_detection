# CKD-detection — reproducible analysis pipeline

**Detection of chronic kidney disease from routine, non-kidney-specific clinical data
(NHANES 2011–2018): development and external temporal validation of machine-learning models.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
**Public repository**: https://github.com/snowman828/CKD_detection

**Key result**: XGBoost AUC 0.810 (95% CI 0.796–0.823) for CKD detection in a completely
unseen NHANES cycle (2017–2018; n=5,801), using only non-kidney-specific routine clinical
variables. Development set: NHANES 2011–2016 (n=17,777).

---

## 1. Repository layout

| Path | Contents |
|---|---|
| `scripts/` | 41 Python scripts — full analysis pipeline, sub-studies, external validation, figure/table generation |
| `results/*.json` | 19 machine-readable result files (AUC, CIs, verdicts, participant flow, sub-study summaries) |
| `results/figures_cjasn_v2/` | CJASN main-text Figure 1–5 (PDF + PNG) — *produced by* `make_cjasn_figures.py`, not committed |
| `models/` | Trained model artifacts: `xgb_model.json`, `lr_model.joblib`, `mlp_model.joblib`, `preprocessing.json`, `model_metadata.json` |
| `models/README.md` | Load/apply snippets for each model artifact |
| `analysis_plan/` | Pre-specified analysis protocol (the study was **not** registered) |
| `DATA_AVAILABILITY.md` | Data sources, ethics, and what is intentionally **not** redistributed |
| `CITATION.cff` | Citation metadata (author ORCIDs filled; DOI added on publication) |
| `LICENSE` | MIT |

**Not in the repository** (public NHANES data is fetched, not committed — see `.gitignore`):

- `results/*.npy`, `results/*.parquet` — intermediate test-set predictions and cohort matrix
- `data/` — raw NHANES cycle files (`.XPT`) and LMF mortality-linkage files
- `results/figures_cjasn_v2/` — generated figure output

---

## 2. How to run

All paths below are relative to the repository root. Install once:

```bash
python -m venv .venv
source .venv/Scripts/activate        # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

### 2a. Runs immediately from a fresh checkout (no data rebuild needed)

These two scripts read **only** committed `results/*.json` and complete without any
`data/` or `results/*.parquet` present:

```bash
# Main-text Figure 1 (participant flow) + Figure 5 (subgroup AUC forest)
python scripts/make_cjasn_figures.py
#   → results/figures_cjasn_v2/Figure1.pdf, Figure5.pdf
#   If .npy/.parquet are absent it prints a rebuild guide and skips Fig 2–4 (no traceback)

# CrossRef per-citation reference verification
python scripts/verify_refs.py
```

Figures 2–4 (ROC curves, calibration, decision-curve) additionally need the test-set
`test_proba_*.npy` / `test_set.parquet` / `cohort.parquet` — rebuild those via §2b.

### 2b. Full pipeline (rebuilds intermediate artifacts, then regenerates everything)

```bash
# 1. Download public NHANES clinical files
python scripts/download_nhanes.py
#    ↓  NOTE — cycle H (2013–2014) is required by build_cohort.py but is NOT in
#    download_nhanes.py CYCLE_FILES.  Download H manually from the CDC NHANES site:
#    https://wwwn.cdc.gov/nchs/nhanes/ContinuousNhanes/2013-2014/  (Demo_H + 6 companion files)

# 2. Build analysis cohort  →  results/cohort.parquet
python scripts/build_cohort.py

# 3. Train + externally validate LR / XGBoost / MLP  →  results/model_results.json
#    + test_proba_{lr,xgb,mlp}.npy + test_set.parquet
python scripts/modeling.py

# 4. Persist trained models + verify against published values (|ΔAUC| ≤ 1e-4)
python scripts/export_models.py
#    → models/ (xgb_model.json, lr/mlp joblib, preprocessing.json, model_metadata.json)

# 5. Now all five main-text figures generate
python scripts/make_cjasn_figures.py

# 6. Tables and sub-studies
python scripts/m9_table1.py                        # Table 1 (cohort characteristics)
python scripts/m10_table2.py                        # Table 2 (sub-study verdicts)
python scripts/m2_substudy12.py                     # sub-studies S1–S2
python scripts/m3_substudy3.py                      # sub-study S3 (spectrum attribution)
python scripts/m5_substudy4.py                     # sub-study S4 (PIR gradient, entropy balancing)
python scripts/m4_substudy5b.py                     # sub-study S5-B (LMF mortality; reads data/lmf/)
python scripts/cox_s5b.py                           # S5-B Cox survival analysis
python scripts/component_and_subgroup_analysis.py   # component AUCs + subgroup calibration
python scripts/subgroup_dca_incremental.py          # subgroup DCA + incremental analysis
python scripts/gen_supplementary.py                 # Supplementary Information (S1–S3 + methods)
```

External-validation scripts (frozen development model applied to independent eras / systems):

```bash
python scripts/p2_independent_era.py    # NHANES 2007–2010 (cycles E+F); needs cohort.parquet
python scripts/p3_knh_validation.py    # KNHANES cross-system (frozen model; uACR unit gate)
python scripts/knh2022_cross_system.py # KNHANES 2022 (frozen model + race-blind pre-spec)
python scripts/knh_multiwave_cross_system.py  # KNHANES 2022/2023/2024 (3-wave consistency)
```

`p3_knh_validation.py` and `knh*_cross_system.py` also `import pyreadstat` — install it
explicitly if running them: `pip install pyreadstat`.

### 2c. Data dependency map (which files each script reads)

| Script | Reads (committed?) | Produced by |
|---|---|---|
| `make_cjasn_figures.py` Fig 1, 5 | `results/participant_flow.json` · `results/supplementary_results.json` · `results/subgroup_dca_incremental.json` *(committed ✓)* | — |
| `make_cjasn_figures.py` Fig 2–4 | `results/test_proba_{xgb,lr,mlp}.npy` · `results/test_y.npy` · `results/test_set.parquet` · `results/cohort.parquet` *(not committed)* | `modeling.py` / `build_cohort.py` |
| `export_models.py` | `results/cohort.parquet` · `results/model_results.json` · `results/test_proba_*.npy` | `build_cohort.py` · `modeling.py` |
| `modeling.py` | `results/cohort.parquet` | `build_cohort.py` |
| `check_nns.py` | `results/cohort.parquet` · `results/model_results.json` | as above |
| `m9_table1.py` / `m10_table2.py` | `results/cohort.parquet` · `results/m2/*.json` | `build_cohort.py` · sub-study scripts |
| `cox_s5b.py` / `m4_substudy5b.py` | `data/lmf/*.XPT` *(manual download — mortality linkage)* + `results/cohort.parquet` | — · `build_cohort.py` |
| `p3_knh_validation.py` | KNHANES source files under `data/` (obtained via KDCA / KISTI — human download) | — |

---

## 3. Citing results and model artifacts

All quantitative values in the manuscript trace back to committed `results/*.json`:

| Metric | File | Key path |
|---|---|---|
| Model test-set AUC / sens / spec | `results/model_results.json` | `XGB.auc_test`, `LR.auc_test`, `MLP.auc_test` (keys are uppercase) |
| Subgroup AUC + 95% CI (10 subgroups) | `results/supplementary_results.json` | `subgroup_auc_ci_xgb` — dict: `age_18_44` / `sex_M` / `race_black` / `pov_lt1.3` / … → `{auc, ci95}` |
| Subgroup DCA net benefit | `results/subgroup_dca_incremental.json` | `overall.auc_xgb` + `subgroups` |
| Participant-flow counts | `results/participant_flow.json` | `n_screened`, `n_analytic`, `n_ckd` |
| Independent-era (NHANES 2007–2010) | `results/p2_independent_era.json` | `discrimination.xgb` = 0.8131, `discrimination.lr`, `discrimination.age_only` |
| KNHANES cross-system | `results/knh2022_cross_system.json` | `auc` + `self_check_nhanes_2017_2018_auc` (guard-rail), `knh_cohort` |
| Trained models (load directly) | `models/xgb_model.json` · `models/lr_model.joblib` · `models/mlp_model.joblib` | see `models/README.md` |
| Preprocessing (imputation medians + column order) | `models/preprocessing.json` | — |

To apply a trained model to new data:

```python
import joblib, json, pandas as pd
prep  = json.load(open("models/preprocessing.json"))
model = joblib.load("models/lr_model.joblib")       # or xgboost booster (see models/README.md)
p = model.predict_proba(encode(df))[:, 1]            # encode() snippet in models/README.md
```

---

## 4. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `make_cjasn_figures.py` outputs only Figure 1 + 5 and prints a "missing files" guide | `.npy/.parquet` intermediate artifacts absent (fresh clone) | Run §2b steps 1–3, then re-run; Fig 2–4 will generate |
| `export_models.py` exits `❌ 缺少必需输入` (exit 1) | `results/cohort.parquet` or `results/model_results.json` absent | Run §2b steps 1–3 first |
| `build_cohort.py` fails on cycle H | H (2013–2014) files not in `download_nhanes.py` | Download H files manually from the CDC NHANES 2013–2014 page (URL in §2b step 1) |
| `No module named 'pyreadstat'` | KNHANES scripts require `pyreadstat`, listed under "optional" in `requirements.txt` | `pip install pyreadstat` |
| `No module named 'lifelines'` | `cox_s5b.py` / `check_nns.py` need `lifelines` | `pip install lifelines==0.30.3` (already in requirements.txt) |
| `p2_independent_era.py` → `ValueError: All objects passed were None` | `cohort.parquet` absent or cycle E/F rows not present | Rebuild cohort via §2b; ensure NHANES 2007–2010 files are in `data/` |
| `ModuleNotFoundError` after `pip install -r requirements.txt` | Check "optional" block entries in `requirements.txt` | `pymupdf` (font self-check only), `pyreadstat`, `seaborn` (m6_figure1 only) |

**One-shot patching scripts** (`fix_refs_final.py`, `fix_vancouver_final.py`, `vancouver_renumber.py`,
`m12_refs.py`, `m13_clean4submission.py`, `m14_review_fixes.py`) are retained for the audit trail
only — **do not re-run** them on the finalised manuscript; they perform destructive substitutions
designed for an intermediate state.

---

## Ethics

NHANES protocols were reviewed and approved by the NCHS Research Ethics Review Board
(Protocols #2011-17 and #2018-01); documented informed consent was obtained from all
participants. This secondary analysis uses de-identified public-use files and required no
additional ethics approval. KNHANES data were obtained under KDCA public-use terms
(cross-system validation only; no re-identification or cross-linkage to NHANES).

## Licence

MIT — see `LICENSE`.


## Reproducibility

Every reported number is regenerated by `python scripts/rerun_all.py --check`; see `REPRODUCE.md` for stages, expected outputs and the hard self-check gates (temporal AUC 0.8099; race-blind 0.8065).
