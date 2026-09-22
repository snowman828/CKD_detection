# OSF Preregistration — MACKI-Dry (Detection manuscript)

> This file is the ready-to-paste content for the OSF preregistration. Paste the sections below into an OSF "OSF Preregistration" form (https://osf.io/prereg/), then cite the resulting DOI in the manuscript.

## Study Information

**Title**: Detection of chronic kidney disease from routine clinical data in the absence of kidney-specific testing: development and external temporal validation of machine-learning models in NHANES 2011–2018

**Research question**: Can chronic kidney disease (CKD; eGFR <60 mL/min/1.73 m² or uACR ≥30 mg/g, CKD-EPI 2021 without the race coefficient) be detected from non-kidney-specific routine clinical features alone (demographics, socioeconomic indicators, BMI, blood pressure, glycaemic status, diabetes history, lipids), with no imaging and no kidney-specific laboratory values?

## Hypotheses

- H1: A machine-learning model trained on non-kidney-specific routine features achieves external temporal-validation AUC ≥0.75 for prevalent CKD.
- H2: The full feature set adds discriminative value over age alone (ΔAUC >0, quantified against an age-only benchmark).
- H3: The full feature set adds value over a minimal clinical rule (age + diabetes + systolic blood pressure).

## Design Plan

- **Data**: NHANES 2011–2012, 2013–2014, 2015–2016 (development, n=17,777); 2017–2018 (external temporal validation, n=5,801). 2019–2020 excluded (COVID-19 suspension; not nationally representative).
- **Outcome**: prevalent CKD (eGFR <60 or uACR ≥30, KDIGO).
- **Features (anti-leakage)**: all label-defining measurements (creatinine, cystatin C, urine albumin/creatinine) explicitly excluded; 12 clinical features + 12 missingness indicators.
- **Models**: logistic regression, XGBoost, multilayer perceptron (fixed hyperparameters).
- **Thresholds and imputation fitted on the development set only** (no test-set leakage).

## Analysis Plan

- **Discrimination**: AUC with 95% CI (2,000 bootstrap resamples); pairwise model differences via bootstrap paired difference (5,000 resamples).
- **Clinical performance**: sensitivity/specificity/PPV/NPV at the development-derived Youden threshold.
- **Calibration**: slope/intercept (linear fit across deciles) and expected calibration error (ECE).
- **Clinical utility**: decision-curve analysis (threshold probabilities 5–25%).
- **Fairness audit**: subgroup AUCs by age, sex, race/ethnicity, PIR (exploratory; no multiplicity adjustment).
- **Robustness/sensitivity**: survey-weight-adjusted AUC; cycle-weighted prevalence; within-age-group discrimination; seed sensitivity (≥3 seeds); race-blind sensitivity (race features removed).

## Known Deviations

The companion mechanistic manuscript (MACKI-Followup) was conceived after the initial analysis and is registered separately; its five substudies (S1 footprint, S2 blind zone, S3 spectrum attribution, S4 PIR attribution, S5-B mortality) are hypothesis-driven and exploratory.
