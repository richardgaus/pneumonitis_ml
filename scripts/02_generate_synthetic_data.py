#!/usr/bin/env python3
"""
02_generate_synthetic_data.py

Script to generate synthetic pneumonitis data for testing and validation.
Creates realistic or easy-to-classify synthetic datasets.
Generates both one-row-per-patient and multi-row datasets to match real data structure.

Usage:
    python 02_generate_synthetic_data.py --analysis {1,2} [options]

Examples:
    python 02_generate_synthetic_data.py --analysis 1 --easy
    python 02_generate_synthetic_data.py --analysis 2 --n-patients 200 --seed 42
    python 02_generate_synthetic_data.py --analysis 1 --output-dir data/simulation --suffix custom
    python 02_generate_synthetic_data.py --analysis 1 --single-only  # Only generate one-row-per-patient
    python 02_generate_synthetic_data.py --analysis 1 --multi-only   # Only generate multi-row dataset
"""

import argparse
import sys
from pathlib import Path

# Import required functions and config
from pneumonitis_ml.config import (
    DATA_SIMULATION, REPO_ROOT,
    load_predictor_set, load_endpoint
)
from pneumonitis_ml.simulation import generate_synthetic_pneumonitis, validate_dataset
from pneumonitis_ml.etl.data_cleaning import select_one_treatment_per_patient


def generate_synthetic_datasets(analysis_number, easy=False, n_patients=126, 
                               n_treatments=150, n_rows=167, seed=2025, 
                               target_event_rate=None, output_dir=None, 
                               output_suffix=None, generate_single=True, 
                               generate_multi=True):
    """
    Generate synthetic datasets for specified analysis.
    Creates both one-row-per-patient and multi-row datasets.
    
    Parameters:
    -----------
    analysis_number : int
        Analysis number (1 or 2)
    easy : bool
        Whether to generate easy-to-classify dataset
    n_patients : int
        Number of unique patients
    n_treatments : int
        Total number of treatments across all patients
    n_rows : int
        Total number of lesions/rows in multi-row dataset
    seed : int
        Random seed for reproducibility
    target_event_rate : float, optional
        Target event rate (uses function default if None)
    output_dir : Path
        Directory to save generated data
    output_suffix : str, optional
        Suffix to add to output filename
    generate_single : bool
        Whether to generate one-row-per-patient dataset
    generate_multi : bool
        Whether to generate multi-row dataset
        
    Returns:
    --------
    dict : Dictionary containing generated datasets
    """
    
    print(f"Generating synthetic data for Analysis {analysis_number}...")
    
    # Load predictor sets and endpoints to validate compatibility
    try:
        predictors = load_predictor_set(which=f"analysis_{analysis_number}")
        endpoint = load_endpoint(which=f"analysis_{analysis_number}")
        print(f"Target predictors: {len(predictors)} features")
        print(f"Target endpoint: {endpoint}")
    except Exception as e:
        print(f"Warning: Could not load analysis {analysis_number} configuration: {e}")
        predictors = None
        endpoint = None
    
    print(f"Generation parameters:")
    print(f"  - Patients: {n_patients}")
    print(f"  - Treatments: {n_treatments}")
    print(f"  - Rows (multi-row): {n_rows}")
    print(f"  - Seed: {seed}")
    print(f"  - Easy mode: {easy}")
    print(f"  - Target event rate: {target_event_rate or 'default'}")
    print(f"  - Generate single-row dataset: {generate_single}")
    print(f"  - Generate multi-row dataset: {generate_multi}")
    
    if easy:
        print(f"\nEasy mode differences from realistic:")
        print(f"  - Higher event rate (0.25 vs 0.12)")
        print(f"  - Large effect sizes (2.0-2.5x vs 0.35-0.55x)")
        print(f"  - Less noise in dosimetric variables")
        print(f"  - Less missing data (5% vs 10%)")
        print(f"  - Smaller lesion-level jitter")
        print(f"  - Expected AUROC > 0.9 (vs ~0.6-0.7 realistic)")
    else:
        print(f"\nRealistic mode features:")
        print(f"  - Low event rate (0.12) mimicking clinical data")
        print(f"  - Subtle effect sizes challenging for detection")
        print(f"  - Realistic noise and missing data patterns")
        print(f"  - Expected AUROC ~0.6-0.7 (clinically realistic)")
    
    results = {}
    
    # Generate multi-row dataset first (if requested)
    if generate_multi:
        print(f"\n{'='*50}")
        print("GENERATING MULTI-ROW DATASET")
        print(f"{'='*50}")
        
        df_multi = generate_synthetic_pneumonitis(
            n_patients=n_patients,
            n_treatments=n_treatments,
            n_rows=n_rows,
            seed=seed,
            target_event_rate=target_event_rate,
            easy=easy
        )
        
        print(f"Generated multi-row dataset shape: {df_multi.shape}")
        print(f"Dataset includes Date_SBRT_start column")
        
        # Validate against target configuration
        if predictors and endpoint:
            validate_dataset(df_multi, predictors, endpoint, "multi-row")
        
        # Save multi-row dataset
        difficulty = "easy" if easy else "realistic"
        base_name = f"data_synthetic_analysis_{analysis_number}_{difficulty}_multi"
        
        if output_suffix:
            base_name += f"_{output_suffix}"
        
        multi_output_file = output_dir / f"{base_name}.xlsx"
        
        print(f"Saving multi-row dataset to: {multi_output_file}")
        output_dir.mkdir(parents=True, exist_ok=True)
        df_multi.to_excel(multi_output_file, index=False)
        
        results['multi_row'] = {
            'data': df_multi,
            'file': multi_output_file,
            'shape': df_multi.shape
        }
        
        print(f"Multi-row dataset summary:")
        print(f"  - Shape: {df_multi.shape}")
        print(f"  - Unique patients: {df_multi['patient_id'].nunique()}")
        print(f"  - Unique treatments: {df_multi.groupby('patient_id')['treatment_id'].nunique().sum()}")
        print(f"  - Max treatments per patient: {df_multi.groupby('patient_id')['treatment_id'].nunique().max()}")
        print(f"  - Date range: {df_multi['Date_SBRT_start'].min()} to {df_multi['Date_SBRT_start'].max()}")
    
    # Generate single-row dataset (if requested)
    if generate_single:
        print(f"\n{'='*50}")
        print("GENERATING SINGLE-ROW DATASET")
        print(f"{'='*50}")
        
        if generate_multi and 'multi_row' in results:
            # Use the multi-row dataset to create single-row by selecting one treatment per patient
            print("Creating single-row dataset from multi-row using treatment selection...")
            df_single = select_one_treatment_per_patient(
                df=results['multi_row']['data'], 
                pneumonitis_col=endpoint or "Pneumonitis G0-1=0, G>/=2=1",
                date_col='Date_SBRT_start'
            )
        else:
            # Generate single-row dataset directly
            print("Generating single-row dataset directly...")
            df_single = generate_synthetic_pneumonitis(
                n_patients=n_patients,
                n_treatments=n_patients,  # One treatment per patient
                n_rows=n_patients,        # One row per patient
                seed=seed,
                target_event_rate=target_event_rate,
                easy=easy
            )
        
        print(f"Generated single-row dataset shape: {df_single.shape}")
        
        # Validate single-row dataset
        if predictors and endpoint:
            validate_dataset(df_single, predictors, endpoint, "single-row")
        
        # Save single-row dataset
        difficulty = "easy" if easy else "realistic"
        base_name = f"data_synthetic_analysis_{analysis_number}_{difficulty}_single"
        
        if output_suffix:
            base_name += f"_{output_suffix}"
        
        single_output_file = output_dir / f"{base_name}.xlsx"
        
        print(f"Saving single-row dataset to: {single_output_file}")
        output_dir.mkdir(parents=True, exist_ok=True)
        df_single.to_excel(single_output_file, index=False)
        
        results['single_row'] = {
            'data': df_single,
            'file': single_output_file,
            'shape': df_single.shape
        }
        
        print(f"Single-row dataset summary:")
        print(f"  - Shape: {df_single.shape}")
        print(f"  - Unique patients: {df_single['patient_id'].nunique()}")
        print(f"  - All patients have 1 treatment: {(df_single.groupby('patient_id')['treatment_id'].nunique() == 1).all()}")
        print(f"  - Date range: {df_single['Date_SBRT_start'].min()} to {df_single['Date_SBRT_start'].max()}")
    
    # Print overall summary
    print(f"\n{'='*60}")
    print("GENERATION SUMMARY")
    print(f"{'='*60}")
    print(f"Analysis: {analysis_number}")
    print(f"Difficulty: {'Easy' if easy else 'Realistic'}")
    print(f"Output directory: {output_dir}")
    
    for dataset_type, info in results.items():
        print(f"{dataset_type.replace('_', '-').title()} dataset:")
        print(f"  - File: {info['file'].name}")
        print(f"  - Shape: {info['shape']}")
        if endpoint and endpoint in info['data'].columns:
            event_rate = info['data'][endpoint].mean()
            print(f"  - Event rate: {event_rate:.3f}")
    
    return results


def main():
    """Main function to parse arguments and generate synthetic data."""
    
    parser = argparse.ArgumentParser(
        description="Generate synthetic pneumonitis data for testing and validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate both single and multi-row datasets (default)
  python 02_generate_synthetic_data.py --analysis 1

  # Generate only single-row dataset
  python 02_generate_synthetic_data.py --analysis 1 --single-only

  # Generate only multi-row dataset  
  python 02_generate_synthetic_data.py --analysis 1 --multi-only

  # Generate easy datasets with custom parameters
  python 02_generate_synthetic_data.py --analysis 2 --easy --n-patients 200 --n-rows 250

  # Custom parameters matching real data structure
  python 02_generate_synthetic_data.py --analysis 1 --n-patients 126 --n-treatments 150 --n-rows 167

  # Batch generation with suffix
  python 02_generate_synthetic_data.py --analysis 1 --suffix experiment1
  python 02_generate_synthetic_data.py --analysis 1 --easy --suffix experiment1
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--analysis',
        type=int,
        choices=[1, 2],
        required=True,
        help='Analysis number (1 or 2)'
    )
    
    # Data generation parameters
    parser.add_argument(
        '--easy',
        action='store_true',
        help='Generate easy-to-classify dataset (default: realistic dataset)'
    )
    
    parser.add_argument(
        '--n-patients',
        type=int,
        default=126,
        help='Number of unique patients (default: 126, matching real data)'
    )
    
    parser.add_argument(
        '--n-treatments',
        type=int,
        default=150,
        help='Total number of treatments for multi-row dataset (default: 150)'
    )
    
    parser.add_argument(
        '--n-rows',
        type=int,
        default=167,
        help='Total number of lesions/rows for multi-row dataset (default: 167, matching real data)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=2025,
        help='Random seed for reproducibility (default: 2025)'
    )
    
    parser.add_argument(
        '--event-rate',
        type=float,
        default=None,
        help='Target event rate (default: 0.12 for realistic, 0.25 for easy)'
    )
    
    # Dataset type selection
    parser.add_argument(
        '--single-only',
        action='store_true',
        help='Generate only single-row-per-patient dataset'
    )
    
    parser.add_argument(
        '--multi-only',
        action='store_true',
        help='Generate only multi-row dataset'
    )
    
    # Output parameters
    parser.add_argument(
        '--output-dir',
        type=str,
        default=DATA_SIMULATION,
        help='Output directory for generated files (default: DATA_SIMULATION from config)'
    )
    
    parser.add_argument(
        '--suffix',
        type=str,
        default=None,
        help='Suffix to add to output filename'
    )
    
    # Control parameters
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be generated without actually creating files'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Validate mutually exclusive options
    if args.single_only and args.multi_only:
        print("Error: Cannot specify both --single-only and --multi-only")
        sys.exit(1)
    
    # Determine what to generate
    generate_single = not args.multi_only  # Generate single unless multi-only specified
    generate_multi = not args.single_only  # Generate multi unless single-only specified
    
    # Set output directory
    output_dir = Path(args.output_dir)
    
    # If output directory path is relative, resolve it relative to REPO_ROOT
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    
    # Print configuration
    print("="*60)
    print("SYNTHETIC DATA GENERATION CONFIGURATION")
    print("="*60)
    print(f"Analysis: {args.analysis}")
    print(f"Difficulty: {'Easy' if args.easy else 'Realistic'}")
    print(f"Patients: {args.n_patients}")
    print(f"Treatments (multi-row): {args.n_treatments}")
    print(f"Rows (multi-row): {args.n_rows}")
    print(f"Seed: {args.seed}")
    print(f"Event rate: {args.event_rate or 'default'}")
    print(f"Generate single-row: {generate_single}")
    print(f"Generate multi-row: {generate_multi}")
    print(f"Output directory: {output_dir}")
    print(f"Output suffix: {args.suffix or 'none'}")
    print(f"Dry run: {args.dry_run}")
    print("="*60)
    
    if args.dry_run:
        print("DRY RUN MODE - No files will be created")
        
        # Show what filenames would be generated
        difficulty = "easy" if args.easy else "realistic"
        
        if generate_single:
            base_name = f"data_synthetic_analysis_{args.analysis}_{difficulty}_single"
            if args.suffix:
                base_name += f"_{args.suffix}"
            single_file = output_dir / f"{base_name}.xlsx"
            print(f"Would generate single-row: {single_file}")
        
        if generate_multi:
            base_name = f"data_synthetic_analysis_{args.analysis}_{difficulty}_multi"
            if args.suffix:
                base_name += f"_{args.suffix}"
            multi_file = output_dir / f"{base_name}.xlsx"
            print(f"Would generate multi-row: {multi_file}")
        
        return
    
    try:
        # Generate synthetic datasets
        results = generate_synthetic_datasets(
            analysis_number=args.analysis,
            easy=args.easy,
            n_patients=args.n_patients,
            n_treatments=args.n_treatments,
            n_rows=args.n_rows,
            seed=args.seed,
            target_event_rate=args.event_rate,
            output_dir=output_dir,
            output_suffix=args.suffix,
            generate_single=generate_single,
            generate_multi=generate_multi
        )
        
        print(f"\nSynthetic data generation completed successfully!")
        print(f"Generated {len(results)} dataset(s)")
        
        if args.verbose:
            for dataset_type, info in results.items():
                print(f"\n{dataset_type.replace('_', '-').title()} dataset preview:")
                print(info['data'].head())
                print(f"\n{dataset_type.replace('_', '-').title()} dataset info:")
                print(info['data'].info())
        
    except Exception as e:
        print(f"Error during generation: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
