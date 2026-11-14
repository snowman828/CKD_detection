# requirements.txt provenance

Every pinned package was derived from the actual imports in `scripts/*.py` (import-graph scan),
not copied from a template.

| Package | Version | Source |
|---|---|---|
| Pillow | 12.3.0 | imported by the figure-format helper |
| PySocks | (unpinned) | imported indirectly via the HTTP proxy client |
| PyYAML | 6.0.3 | imported by the metadata writer |
| lifelines | 0.30.3 | survival models (Cox, C-index) |
| matplotlib | 3.11.1 | all figures |
| pandas | 2.3.3 | tabular processing |
| scikit-learn | 1.9.0 | logistic regression, MLP, metrics |
| scipy | 1.17.1 | statistical tests |
| seaborn | (unpinned) | figure styling |
| xgboost | 3.2.0 | primary model |

Versions were read from the environment used to produce the reported results
(`uv pip list --python <interpreter>`). Packages without a pinned version were present
in that environment but are not imported directly by any analysis script.
