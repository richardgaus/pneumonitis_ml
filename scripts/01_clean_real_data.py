"""
01_clean_real_data.py

Script to clean and process real clinical data for pneumonitis analysis.
Handles both analysis 1 and analysis 2 datasets.

Usage:
    python 01_clean_real_data.py <clinical_data_file> --analysis {1,2} [options]

Example:
    python 01_clean_real_data.py data/raw/clinical_analysis_1.xlsx --analysis 1
    python 01_clean_real_data.py data/raw/clinical_analysis_2.xlsx --analysis 2 --output-dir data/processed
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Import required functions and config
from pneumonitis_ml.config import (
    DATA_RAW, DATA_PROCESSED, REPO_ROOT,
    load_predictor_set, load_endpoint
)
from pneumonitis_ml.etl.data_cleaning import (
    trim_dataframe_at_empty_index, 
    select_one_treatment_per_patient, 
    extract_id_columns
)


def load_and_process_analysis_data(clinical_file_path, analysis_number, output_dir):
    """
    Load and process clinical data for specified analysis.
    
    Parameters:
    -----------
    clinical_file_path : str or Path
        Path to the clinical data Excel file
    analysis_number : int
        Analysis number (1 or 2)
    output_dir : str or Path
        Directory to save processed files
        
    Returns:
    --------
    tuple : (df_full, df_final) - Full dataset and one-treatment-per-patient dataset
    """
    
    print(f"Processing Analysis {analysis_number} data...")
    print(f"Input file: {clinical_file_path}")
    
    # Load predictor sets and endpoints for validation purposes
    predictors = load_predictor_set(which=f"analysis_{analysis_number}")
    endpoint = load_endpoint(which=f"analysis_{analysis_number}")
    
    print(f"Analysis configuration: {len(predictors)} predictors, endpoint: {endpoint}")
    print("Note: Saving FULL dataset with all columns (not just selected predictors)")
    
    # Load clinical data
    print("Loading clinical data...")
    if analysis_number == 1:
        df_clinical_data = pd.read_excel(
            clinical_file_path,
            sheet_name="clinical_data",
            header=0,
            dtype={'Pat': str}  # Force Pat column to string
        )
    else:  # analysis_number == 2
        df_clinical_data = pd.read_excel(
            clinical_file_path,
            sheet_name="clinical_data",
            header=0
        )
    
    print(f"Clinical data shape: {df_clinical_data.shape}")
    
    # Load dosimetric data
    print("Loading dosimetric data...")
    if analysis_number == 1:
        # Analysis 1 has multi-level headers
        df_dosimetric_data = pd.read_excel(
            clinical_file_path,
            sheet_name="dosimetric_data",
            header=[0, 1],
            dtype={'Pat': str}  # Force Pat column to string
        )
        
        # Flatten multi-level column headers
        print("Flattening multi-level column headers...")
        new_columns = []
        for col in df_dosimetric_data.columns:
            if pd.isna(col[0]) or col[0] == '' or col[0].startswith("Unnamed:"):
                # If first level is empty/unnamed, use only second level
                new_columns.append(col[1])
            elif pd.isna(col[1]) or col[1] == '':
                # If second level is empty, use only first level
                new_columns.append(col[0])
            else:
                # Combine both levels with underscore
                new_columns.append(f"{col[0]}_{col[1]}")
        
        df_dosimetric_data.columns = new_columns
        
    else:  # analysis_number == 2
        # Analysis 2 has single-level headers
        df_dosimetric_data = pd.read_excel(
            clinical_file_path,
            sheet_name="dosimetric_data ",  # Note the space in sheet name
            header=0
        )
    
    print(f"Dosimetric data shape: {df_dosimetric_data.shape}")
    
    # Trim dataframes at empty patient IDs
    print("Trimming dataframes at empty patient IDs...")
    df_clinical_data = trim_dataframe_at_empty_index(df_clinical_data, "Pat")
    df_dosimetric_data = trim_dataframe_at_empty_index(df_dosimetric_data, "Pat")
    
    print(f"After trimming - Clinical: {df_clinical_data.shape}, Dosimetric: {df_dosimetric_data.shape}")
    
    # Handle overlapping columns
    print("Handling overlapping columns...")
    overlapping_cols = set(df_clinical_data.columns).intersection(set(df_dosimetric_data.columns))
    if overlapping_cols:
        print(f"Found {len(overlapping_cols)} overlapping columns: {overlapping_cols}")
        df_dosimetric_filtered = df_dosimetric_data.drop(columns=list(overlapping_cols))
    else:
        print("No overlapping columns found")
        df_dosimetric_filtered = df_dosimetric_data
    
    # Join clinical and dosimetric data
    print("Joining clinical and dosimetric data...")
    df_full = df_clinical_data.join(df_dosimetric_filtered, on='Pat', how='left')
    print(f"Joined data shape: {df_full.shape}")
    
    # Extract ID columns
    print("Extracting patient_id, treatment_id, lesion_id...")
    df_full = extract_id_columns(df=df_full)
    
    # Select one treatment per patient
    print("Selecting one treatment per patient...")
    df_selected = select_one_treatment_per_patient(df_full, pneumonitis_col=endpoint)
    print(f"Selected data shape: {df_selected.shape}")
    
    # Validate that required columns are present (but don't filter to only these columns)
    print("Validating required columns are present...")
    id_cols = ["patient_id", "treatment_id", "lesion_id"]
    required_cols = id_cols + predictors + [endpoint]
    
    missing_cols = set(required_cols) - set(df_full.columns)
    if missing_cols:
        print(f"Warning: Missing required columns: {missing_cols}")
        print("Note: Will still save full dataset, but some predictors may be missing")
    else:
        print("All required columns are present")
    
    # Check that ID columns are present (these are essential)
    missing_id_cols = set(id_cols) - set(df_full.columns)
    if missing_id_cols:
        print(f"Error: Missing essential ID columns: {missing_id_cols}")
        raise ValueError(f"Essential ID columns missing: {missing_id_cols}")
    
    # Check that endpoint is present
    if endpoint not in df_full.columns:
        print(f"Error: Endpoint column '{endpoint}' not found")
        raise ValueError(f"Endpoint column '{endpoint}' not found")
    
    print(f"Final datasets - Full: {df_full.shape}, Selected: {df_selected.shape}")
    print(f"Columns in full dataset: {list(df_full.columns)}")
    
    # Save processed datasets (keeping ALL columns)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    full_output_path = output_dir / f"analysis_{analysis_number}_full.xlsx"
    selected_output_path = output_dir / f"analysis_{analysis_number}_one_treatment_per_patient.xlsx"
    
    print(f"Saving full dataset to: {full_output_path}")
    df_full.to_excel(full_output_path, index=False)
    
    print(f"Saving selected dataset to: {selected_output_path}")
    df_selected.to_excel(selected_output_path, index=False)
    
    # Print summary statistics
    print("\n" + "="*60)
    print("PROCESSING SUMMARY")
    print("="*60)
    print(f"Analysis: {analysis_number}")
    print(f"Input file: {clinical_file_path}")
    print(f"Output directory: {output_dir}")
    print(f"Required predictors: {len(predictors)}")
    print(f"Endpoint: {endpoint}")
    print(f"Full dataset: {df_full.shape[0]} rows, {df_full.shape[1]} columns")
    print(f"Selected dataset: {df_selected.shape[0]} rows, {df_selected.shape[1]} columns")
    
    if endpoint in df_selected.columns:
        endpoint_stats = df_selected[endpoint].value_counts()
        print(f"Endpoint distribution in selected dataset:")
        for value, count in endpoint_stats.items():
            percentage = (count / len(df_selected)) * 100
            print(f"  {value}: {count} ({percentage:.1f}%)")
    
    print(f"Unique patients in selected dataset: {df_selected['patient_id'].nunique()}")
    
    # Print available predictors vs required predictors
    available_predictors = [col for col in predictors if col in df_selected.columns]
    missing_predictors = [col for col in predictors if col not in df_selected.columns]
    
    print(f"Available predictors: {len(available_predictors)}/{len(predictors)}")
    if missing_predictors:
        print(f"Missing predictors: {missing_predictors}")
    else:
        print("All required predictors are available")
    
    print("="*60)
    
    return df_full, df_selected


def main():
    """Main function to parse arguments and run data cleaning."""
    
    parser = argparse.ArgumentParser(
        description="Clean and process real clinical data for pneumonitis analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python 01_clean_real_data.py data/raw/clinical_analysis_1.xlsx --analysis 1
  python 01_clean_real_data.py data/raw/clinical_analysis_2.xlsx --analysis 2 --output-dir data/processed
  python 01_clean_real_data.py /absolute/path/to/data.xlsx --analysis 1 --output-dir custom_output --verbose

Note: This script now saves the FULL dataset with all columns. Variable selection will be done during training.
        """
    )
    
    # Required arguments
    parser.add_argument(
        'clinical_file',
        type=str,
        help='Path to the clinical data Excel file'
    )
    
    parser.add_argument(
        '--analysis',
        type=int,
        choices=[1, 2],
        required=True,
        help='Analysis number (1 or 2)'
    )
    
    # Optional arguments
    parser.add_argument(
        '--output-dir',
        type=str,
        default=DATA_PROCESSED,
        help='Output directory for processed files (default: DATA_PROCESSED from config)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without actually processing'
    )
    
    args = parser.parse_args()
    
    # Validate input file exists
    clinical_file_path = Path(args.clinical_file)
    
    # If path is relative, resolve it relative to REPO_ROOT
    if not clinical_file_path.is_absolute():
        clinical_file_path = REPO_ROOT / clinical_file_path
    
    if not clinical_file_path.exists():
        print(f"Error: Clinical data file not found: {clinical_file_path}")
        print(f"Looking relative to repo root: {REPO_ROOT}")
        sys.exit(1)
    
    # Set output directory  
    output_dir = Path(args.output_dir)
    
    # Print configuration
    print("="*60)
    print("CLINICAL DATA CLEANING CONFIGURATION")
    print("="*60)
    print(f"Input file: {clinical_file_path}")
    print(f"Analysis: {args.analysis}")
    print(f"Output directory: {output_dir}")
    print(f"Verbose: {args.verbose}")
    print(f"Dry run: {args.dry_run}")
    print(f"Mode: Save FULL dataset (all columns)")
    print("="*60)
    
    if args.dry_run:
        print("DRY RUN MODE - No files will be created")
        print(f"Would process: {clinical_file_path}")
        print(f"Would save results to: {output_dir}")
        return
    
    try:
        # Process the data
        df_full, df_final = load_and_process_analysis_data(
            clinical_file_path=clinical_file_path,
            analysis_number=args.analysis,
            output_dir=output_dir
        )
        
        print(f"\nProcessing completed successfully!")
        print(f"Output files saved to: {output_dir}")
        print(f"Full dataset contains {df_full.shape[1]} columns (all variables)")
        print(f"Selected dataset contains {df_final.shape[1]} columns (all variables)")
        print("Variable selection will be performed during training phase")
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()