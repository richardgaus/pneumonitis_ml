# run all nested-CV variants (analysis 1)
python scripts/03_run_nested_cv.py data/processed/analysis_1_one_treatment_per_patient.xlsx \
       --analysis 1 --experiment analysis_1_main --n-splits 5 --n-repeats 5 \
       --inner-cv-splits 3 --inner-cv-repeats 5 --model random-forest
python scripts/03_run_nested_cv.py data/processed/analysis_1_one_treatment_per_patient.xlsx \
       --analysis 1 --experiment analysis_1_stability --n-splits 5 --n-repeats 5 \
       --inner-cv-splits 3 --inner-cv-repeats 5 --model random-forest \
       --use-stability-selection 
python scripts/03_run_nested_cv.py data/processed/analysis_1_one_treatment_per_patient.xlsx \
       --analysis 1 --experiment analysis_1_univariate --n-splits 5 --n-repeats 5 \
       --inner-cv-splits 3 --inner-cv-repeats 5 --model random-forest \
       --use-univariate-selection 
python scripts/03_run_nested_cv.py data/processed/analysis_1_full.xlsx \
       --analysis 1 --experiment analysis_1_all_treatments --n-splits 5 --n-repeats 5 \
       --inner-cv-splits 3 --inner-cv-repeats 5 --model random-forest
python scripts/03_run_nested_cv.py data/processed/analysis_1_one_treatment_per_patient.xlsx \
       --analysis 1 --experiment analysis_1_logistic_regression --n-splits 5 --n-repeats 5 \
       --inner-cv-splits 3 --inner-cv-repeats 5 --model logistic-regression

# run all nested-CV variants (analysis 2)
python scripts/03_run_nested_cv.py data/processed/analysis_2_one_treatment_per_patient.xlsx \
       --analysis 2 --experiment analysis_2_main --n-splits 5 --n-repeats 5 \
       --inner-cv-splits 3 --inner-cv-repeats 5 --model random-forest
python scripts/03_run_nested_cv.py data/processed/analysis_2_one_treatment_per_patient.xlsx \
       --analysis 2 --experiment analysis_2_stability --n-splits 5 --n-repeats 5 \
       --inner-cv-splits 3 --inner-cv-repeats 5 --model random-forest \
       --use-stability-selection 
python scripts/03_run_nested_cv.py data/processed/analysis_2_one_treatment_per_patient.xlsx \
       --analysis 2 --experiment analysis_2_univariate --n-splits 5 --n-repeats 5 \
       --inner-cv-splits 3 --inner-cv-repeats 5 --model random-forest \
       --use-univariate-selection 
python scripts/03_run_nested_cv.py data/processed/analysis_2_full.xlsx \
       --analysis 2 --experiment analysis_2_all_treatments --n-splits 5 --n-repeats 5 \
       --inner-cv-splits 3 --inner-cv-repeats 5 --model random-forest
python scripts/03_run_nested_cv.py data/processed/analysis_2_one_treatment_per_patient.xlsx \
       --analysis 2 --experiment analysis_2_logistic_regression --n-splits 5 --n-repeats 5 \
       --inner-cv-splits 3 --inner-cv-repeats 5 --model logistic-regression

# run all nested-CV variants (synthetic)
python scripts/03_run_nested_cv.py data/simulation/data_synthetic_analysis_1_easy_single.xlsx \
       --analysis 1 --experiment analysis_1_synthetic --n-splits 5 --n-repeats 5 \
       --inner-cv-splits 3 --inner-cv-repeats 5 --model random-forest
python scripts/03_run_nested_cv.py data/simulation/data_synthetic_analysis_2_easy_single.xlsx \
       --analysis 2 --experiment analysis_2_synthetic --n-splits 5 --n-repeats 5 \
       --inner-cv-splits 3 --inner-cv-repeats 5 --model random-forest


# analyse all CV results (writes to results/analyzed_cv_results/)
for base in results/raw_cv_results/*/; do
    base_name="${base%_metadata.json}"
    python scripts/04_analyze_results.py "${base_name}"
done