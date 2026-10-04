# Reproduce every reported number

This repository contains the full analysis pipeline for the manuscript. All reported statistics are
regenerated from public data by the scripts below; no numbers are entered by hand.

## One command

```bash
python scripts/rerun_all.py --check
```

Stages (each in `scripts/`):

| Stage | Script | Output |
|---|---|---|
| Cohort construction (label, weights, anti-leakage feature set) | `build_cohort.py` | `results/cohort.parquet`, `results/cohort_summary.json` |
| Three models + temporal validation | `modeling.py` | `results/model_results.json`, `results/test_proba_*.npy` |
| Bootstrap differences, calibration, DCA, subgroups | `supplementary_analysis.py` | `results/supplementary_results.json`, figures |
| Linked-mortality prognostic specificity (post hoc) | `m16_mortality_a2.py` | `results/m16_mortality_a2.json` |
| Transport recalibration ladder (held-out) + local sample-size rule | `m17_transport_ladder.py` | `results/m17_transport_ladder.json` |
| Cross-system validation (KNHANES) | `knh2022_cross_system.py`, `knh_multiwave_cross_system.py` | `results/knh*.json` |

`--check` prints the registered key values next to the recomputed ones and writes
`results/_rerun_log_<timestamp>.json` (per-stage exit codes, durations, tails).

## Hard self-check gates (the run aborts if they fail)

* Full model, temporal validation AUC must equal **0.8099** (tolerance 1e-4).
* Race-blind variant, temporal validation AUC must equal **0.8065** (tolerance 1e-4).

These gates exist so that a silent protocol drift cannot pass unnoticed: if the pipeline changes, the
run stops instead of producing plausible-looking new numbers.

## Environment

`requirements.txt` (Python 3.11; pandas, NumPy, scikit-learn, XGBoost, SciPy, lifelines, pyreadstat).
The cross-system scripts read KNHANES `.sas7bdat` files, which are **not redistributed here** — see
`DATA_AVAILABILITY.md` for how to obtain them.

## Working directory

Run the pipeline **from the analysis root** (the directory that contains `data/` and `scripts/`): the cross-system scripts read NHANES files through the relative path `data/*.XPT`, so a different working directory makes them fail with `FileNotFoundError: data\DEMO_G.XPT`.
