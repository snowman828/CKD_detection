# Figure provenance

Every main-text figure is produced by a script in this repository from analysis outputs committed
under `results/`. No figure is drawn from hand-entered numbers.

| Figure | Script | Primary inputs (committed) | Output |
|---|---|---|---|
| Figure 1 — SHAP effect-pattern clustering against the cardiovascular–kidney–metabolic pathway map | `scripts/m6_figure1.py` | `models/xgb_model.json`, `results/model_results.json`, `results/component_subgroup_results.json` | `figures/Figure1.{png,pdf}` |
| Figure 2 — stratified AUC forest plot | `scripts/m7_figure2.py` | `results/subgroup_dca_incremental.json`, `results/model_results.json` | `figures/Figure2.{png,pdf}` |
| Figure 3 — case severity composition by age group | `scripts/m8_figure3.py` | `results/cohort_summary.json`, `results/participant_flow.json` | `figures/Figure3.{png,pdf}` |

## Regenerating the figures

```bash
python scripts/build_cohort.py          # or: python scripts/rerun_all.py --check
python scripts/modeling.py
python scripts/m6_figure1.py
python scripts/m7_figure2.py
python scripts/m8_figure3.py
```

Print-ready derivatives are produced from the same vector canvases:

```bash
python scripts/figures_to_print_formats.py --dir figures --dpi 300
```

`figures_to_print_formats.py` reads the vector PDFs as input, so raster and print variants cannot
drift from the vector set. Ghostscript and an ICC CMYK profile are optional; when either is missing
the script prints the exact command it would run instead of failing silently.

## Conventions

- Vector PDFs embed their fonts (`pdf.fonttype = 42`); minima of 8.5 pt apply to every text span.
- Figure 2 plots stratum AUCs with 95% bootstrap confidence intervals (2,000 resamples); the point
  estimates printed beside each row also appear in `results/subgroup_dca_incremental.json`.
- Figure 1 counts and pathway assignments follow the SHAP clustering output in
  `results/model_results.json`; the adjusted Rand index is reported in the manuscript.
- Figure 3 uses the same cohort definition as Table 1 and `results/participant_flow.json`.

## Boundaries

`results/test_proba_*.npy`, `results/test_y.npy`, `results/test_set.parquet` and `results/cohort.parquet`
are pipeline outputs rather than committed data: `build_cohort.py` and `modeling.py` regenerate them
from public NHANES files (see `DATA_AVAILABILITY.md`), and the AUC gates in `rerun_all.py --check`
stop the run if they drift. A fresh end-to-end re-execution may differ in sub-point spacing because
layout depends on the installed matplotlib and font versions; the committed PDFs embed their fonts
and are therefore stable.
