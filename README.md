# Pneumonitis-ML: A Reproducible Pipeline for Predicting ≥ G2 Radiation Pneumonitis

This repository contains all code and synthetic data for the study

> **“XXX”**

The workflow covers every step from raw Excel exports to publication-ready figures:

1. **ETL** – clean & harmonise the clinical / dosimetric spreadsheets  
2. **Synthetic data** – generate realistic or “easy” toy data for testing  
3. **Nested cross-validation** – feature selection + hyper-parameter tuning  
4. **Analysis** – bootstrap CIs, calibration, decision curves, plots & tables  
5. **Packaging** – scripts, conda environment, and a lightweight Python API

All real patient data stay local (the `data/raw/` folder is in `.gitignore`).  
The repo ships with purely synthetic datasets so anyone can run the pipeline end-to-end.

## Quick start

```bash
# 1. create and activate the conda env
conda env create -f environment.yml
conda activate pneumonitis_ml

# 2. (optional) generate fresh synthetic data
python scripts/02_generate_synthetic_data.py --analysis 1 --easy

# 3. run nested CV on the synthetic single-row dataset
python scripts/03_run_nested_cv.py \
       data/simulation/data_synthetic_analysis_1_easy_single.xlsx \
       --analysis 1 \
       --use-stability-selection \
       --experiment analysis_using_synthetic_data

# 4. analyse the raw CV output
python scripts/04_analyze_results.py \
       results/raw_cv_results/analysis_using_synthetic_data_<datetime_stamp>
```

Full command-line examples are embedded in each script’s help text (`-h`).

## Repository layout

```
config/                     YAML lists of candidate predictors
data/
  raw/                      ⬅ real (not tracked)
  processed/                cleaned & anonymised versions
  simulation/               synthetic datasets (kept in git)
notebooks/                  exploratory Jupyter notebooks
pneumonitis_ml/             reusable library code
scripts/                    command-line entry points
results/
  raw_cv_results/           folds, predictions, metadata (csv/-json)
  analyzed_cv_results/      tables, figures, summaries
environment.yml             conda spec (Python 3.11)
```

## Key design choices

| Step                      | Rationale                                                                                |
| ------------------------- | ---------------------------------------------------------------------------------------- |
| **Model**                 | L2-regularised logistic regression – low variance, interpretable with small event counts |
| **Feature pre-selection** | Manual clinical judgement (12 candidates) + optional stability selection / optional univariate selection |
| **Outer CV**              | Stratified *Group* K-fold (patients as groups) to avoid leakage                          |
| **Inner CV**              | Repeated 3-fold for λ grid search                                                        |
| **Metrics**               | AUROC, AUPRC, bootstrap 95 % CI, calibration slope/intercept, decision curve             |
| **Sensitivity analyses**  | (i) lesion- vs patient-level splitting, (ii) alternative feature-selection strategies    |

All settings can be overridden from the CLI; defaults follow TRIPOD-AI guidance.

## How to cite

XXX

Please also cite the original TRIPOD-AI and Stability Selection papers when appropriate.

## License

Code is released under the MIT License.
**No patient-identifiable information is included in this repository.**