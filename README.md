# Pneumonitis-ML

End-to-end pipeline for predicting ≥G2 radiation pneumonitis  
Paper: “…” (submitted to …)

## Quick start
```bash
conda env create -f environment.yml
conda activate pneumonitis-ml
python scripts/02_generate_synthetic_data.py --analysis 1
python scripts/03_run_nested_cv.py data/processed/analysis_1_one_treatment_per_patient.xlsx --analysis 1
python scripts/04_analyze_results.py results/raw_cv_results/analysis_1_main_YYYYMMDD_HHMMSS
```

## Repository layout
.
├── pneumonitis_ml/         # reusable package code
├── scripts/                # CLI entry-points
├── data/                   # processed & simulation data (no raw PHI)
├── config/                 # YAML predictor lists
└── results/                # (empty in git; released via Zenodo)
