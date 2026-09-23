# How Data-Quality Defects Affect Credit-Risk Models and Their Explanations

**Author:** Divine Ineza · Independent research project, 2026

## Research question

When credit data contains missing values, duplicates, inconsistent records, or distribution shift, how much do (a) predictive accuracy and calibration and (b) model explanations degrade? Do explanations become unreliable *before* accuracy visibly drops?

## Why it matters

Financial institutions increasingly rely on explanations to justify credit decisions. If poor data quality silently distorts those explanations while accuracy still looks acceptable, a model can appear trustworthy when it is not.

## Data

*Default of Credit Card Clients* (Yeh & Lien, 2009), UCI Machine Learning Repository, id 350: 30,000 credit-card clients in Taiwan, 23 features, binary default target.

## Method (planned)

| Step | Description |
|---|---|
| Models | Logistic regression (transparent baseline) and gradient boosting |
| Defects injected | Missing values (5–30%), duplicates, inconsistent records, distribution shift |
| Metrics | AUC (accuracy), Brier score (calibration), SHAP top-k overlap and rank correlation (explanation stability) |
| Robustness | Each experiment repeated over 5 random seeds |

## Progress

- [x] **Week 1:** data loading, exploration, documentation of existing quality issues, clean baseline
- [ ] Week 2: baseline models and SHAP explanations
- [ ] Week 3: missing-values experiment
- [ ] Week 4: distribution-shift experiment
- [ ] Week 5: duplicates and inconsistent records
- [ ] Week 6: repeated runs and summary results
- [ ] Week 7: written report

## Repository structure

```
src/data.py                         loading and cleaning (reused every week)
notebooks/01_data_exploration.ipynb Week 1
data/raw/                           raw data (not committed)
data/processed/                     clean baseline (not committed)
reports/figures/                    saved charts
```

## How to run

```bash
pip install -r requirements.txt
jupyter notebook notebooks/01_data_exploration.ipynb
```

The notebook downloads the data automatically. If that fails, download the `.xls` file from the UCI page and place it in `data/raw/`.

## Early finding (Week 1)

Although this dataset is widely used as a clean benchmark, several columns contain category codes not defined in its documentation (EDUCATION, MARRIAGE, and the repayment-status columns). This supports the project's premise: data-quality problems exist even in "clean" research data.
