# analyse all CV results (writes to results/analyzed_cv_results/)
for base in results/raw_cv_results/*/; do
    base_name="${base%_metadata.json}"
    python scripts/04_analyze_results.py "${base_name}"
done