# Data availability

Analysis code and the pre-specified analysis plan: **https://github.com/snowman828/CKD_detection** (MIT).

## Source data (public, no restrictions)
All input data are public-use files from the **National Health and Nutrition Examination Survey (NHANES)**,
US Centers for Disease Control and Prevention (CDC) / National Center for Health Statistics (NCHS):

- NHANES survey files (cycles used: 2011-2012, 2013-2014, 2015-2016, 2017-2018): https://wwwn.cdc.gov/nchs/nhanes/
- NHANES Public-Use Linked Mortality Files: https://www.cdc.gov/nchs/data-linkage/mortality-public.htm

NHANES protocols were reviewed and approved by the NCHS Research Ethics Review Board (renamed the NCHS Ethics
Review Board in 2018); documented informed consent was obtained from all participants. This analysis uses
de-identified public-use files only and required no additional ethics approval.

## What is NOT in this repository
- No participant-level data files are redistributed here (`*.parquet` is git-ignored). Run the download and
  cohort-construction scripts to regenerate them from the public sources above.
- No trained model binaries are committed. Models are re-fitted by the scripts (see README, "Reproducing results").

## How to regenerate the analysis data
```bash
python scripts/download_nhanes.py    # fetch the public NHANES cycle files
python scripts/build_cohort.py       # build the analysis cohort (CKD-EPI 2021 eGFR, KDIGO labels, anti-leakage features)
```


## Frozen release of record (this analysis)

Version **1.0.0** is archived and citable via its DOI:

- Version DOI: [10.5281/zenodo.23357060](https://doi.org/10.5281/zenodo.23357060)  
- Concept DOI (all versions): 10.5281/zenodo.23357059  
- Corresponding GitHub tag: [v1.0.0](https://github.com/snowman828/CKD_detection/releases/tag/v1.0.0)
