#!/usr/bin/env python3
"""
04_analyze_results.py

Comprehensive analysis script for cross-validation results.
Generates publication-ready outputs including tables, figures, and summaries.

Usage:
    python 04_analyze_results.py <experiment_directory> [options]

Examples:
    python 04_analyze_results.py results/raw_cv_results/experiment1_20231201_143022
    python 04_analyze_results.py results/raw_cv_results/test_20231201_143022 --threshold 0.6 --format all
"""

import matplotlib.pyplot as plt
import warnings
from pathlib import Path
import argparse
import sys

from pneumonitis_ml.evaluation.data_collector import CVDataCollector
from pneumonitis_ml.evaluation.calculate_metrics import calculate_comprehensive_metrics
from pneumonitis_ml.evaluation.create_output import (
    create_publication_figures, create_publication_tables, generate_text_summary
)
from pneumonitis_ml.config import REPO_ROOT

warnings.filterwarnings('ignore')

# Set matplotlib style for publication-quality plots
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

# Set default output directory
DEFAULT_OUTPUT_DIR = REPO_ROOT / 'results' / 'analyzed_cv_results'

def load_cv_results_with_collector(experiment_path):
    """Load CV results using CVDataCollector from experiment directory."""
    experiment_path = Path(experiment_path)
    
    print(f"Loading CV results from experiment directory: {experiment_path}")
    
    if not experiment_path.exists():
        raise FileNotFoundError(f"Experiment directory not found: {experiment_path}")
    
    if not experiment_path.is_dir():
        raise ValueError(f"Path is not a directory: {experiment_path}")
    
    # Initialize collector and load data
    collector = CVDataCollector()
    
    # Try loading with directory path first, then with base filename
    try:
        dfs = collector.load_data(str(experiment_path))
    except Exception as e:
        # If that fails, try with the base filename (directory name)
        base_filename = experiment_path / experiment_path.name
        dfs = collector.load_data(str(base_filename))
    
    metadata = collector.metadata
    
    return collector, dfs['predictions'], dfs['coefficients'], dfs['feature_selection'], dfs['fold_summaries'], metadata


def main():
    """Main function for command line interface."""
    
    parser = argparse.ArgumentParser(
        description='Analyze CV training results and generate publication outputs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze results with all outputs (using experiment directory)
  python 04_analyze_results.py results/raw_cv_results/experiment1_20231201_143022

  # Custom threshold and output format
  python 04_analyze_results.py results/raw_cv_results/test_20231201_143022 --threshold 0.6 --format csv

  # Only generate figures
  python 04_analyze_results.py results/raw_cv_results/debug_20231201_143022 --format figures
        """
    )
    
    parser.add_argument(
        'experiment_directory',
        type=str,
        help='Path to experiment directory containing CV results'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=None,
        help='Classification threshold (default: optimal using Youden\'s J statistic)'
    )
    
    parser.add_argument(
        '--format',
        choices=['all', 'tables', 'figures', 'summary', 'csv'],
        default='all',
        help='Output format (default: all)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (default: results/analyzed_cv_results/experiment_name)'
    )
    
    args = parser.parse_args()
    
    # Resolve experiment directory path
    experiment_dir = Path(args.experiment_directory)
    if not experiment_dir.is_absolute():
        experiment_dir = REPO_ROOT / experiment_dir
    
    if not experiment_dir.exists():
        print(f"Error: Experiment directory not found: {experiment_dir}")
        sys.exit(1)
    
    if not experiment_dir.is_dir():
        print(f"Error: Path is not a directory: {experiment_dir}")
        sys.exit(1)
    
    # Extract experiment name from directory
    experiment_name = experiment_dir.name
    
    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute():
            output_dir = REPO_ROOT / output_dir
    else:
        # Default: results/analyzed_cv_results/experiment_name/
        output_dir = DEFAULT_OUTPUT_DIR / experiment_name
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("COMPREHENSIVE RESULTS ANALYSIS")
    print("="*60)
    print(f"Experiment directory: {experiment_dir}")
    print(f"Experiment name: {experiment_name}")
    print(f"Output directory: {output_dir}")
    print(f"Format: {args.format}")
    if args.threshold is not None:
        print(f"Threshold: {args.threshold}")
    else:
        print("Threshold: Optimal (Youden's J statistic)")
    print("="*60)
    
    try:
        # Load data
        print("\nLoading CV results...")
        collector, df_predictions, df_coefficients, df_feature_selection, df_fold_summaries, metadata = load_cv_results_with_collector(experiment_dir)
        
        # Validate we have data
        if len(df_predictions) == 0:
            print("Error: No predictions data found")
            sys.exit(1)
        
        if len(df_fold_summaries) == 0:
            print("Error: No fold summaries found")
            sys.exit(1)
        
        print(f"Loaded data successfully:")
        print(f"  - Predictions: {len(df_predictions)} rows")
        print(f"  - Coefficients: {len(df_coefficients)} rows")
        print(f"  - Feature selection: {len(df_feature_selection)} rows")
        print(f"  - Fold summaries: {len(df_fold_summaries)} rows")
        
        # Calculate metrics
        print("\nCalculating comprehensive metrics...")
        results = calculate_comprehensive_metrics(
            df_predictions, df_coefficients, df_feature_selection,
            df_fold_summaries, args.threshold
        )
        
        if not results:
            print("Error: Failed to calculate metrics")
            sys.exit(1)
        
        # Create tables
        print("\nGenerating publication tables...")
        tables = create_publication_tables(results, metadata)
        
        # Generate outputs based on format
        output_files = []
        
        if args.format in ['all', 'tables', 'csv']:
            print("\nGenerating CSV tables...")
            for table_name, table_df in tables.items():
                if len(table_df) > 0:
                    csv_path = output_dir / f"{experiment_name}_{table_name}.csv"
                    table_df.to_csv(csv_path, index=False)
                    output_files.append(csv_path)
                    print(f"  - {csv_path.name}")
        
        if args.format in ['all', 'figures']:
            print("\nGenerating publication figures...")
            figures = create_publication_figures(results, output_dir, experiment_name)
            output_files.extend(figures.values())
            for fig_name, fig_path in figures.items():
                print(f"  - {fig_path.name}")
        
        if args.format in ['all', 'summary']:
            print("\nGenerating text summary...")
            summary_path = output_dir / f"{experiment_name}_summary.txt"
            generate_text_summary(results, metadata, tables, summary_path)
            output_files.append(summary_path)
            print(f"  - {summary_path.name}")
        
        # Print summary to console
        print("\n" + "="*60)
        print("RESULTS SUMMARY")
        print("="*60)
        
        # Show main results
        perf = results['performance_metrics']
        print(f"AUROC: {perf['auroc']['mean']:.3f} (95% CI: {perf['auroc']['ci_lower']:.3f}-{perf['auroc']['ci_upper']:.3f})")
        print(f"AUPRC: {perf['auprc']['mean']:.3f} (95% CI: {perf['auprc']['ci_lower']:.3f}-{perf['auprc']['ci_upper']:.3f})")
        
        cm = results['classification_metrics']
        print(f"Threshold  : {cm['threshold']:.3f} ({cm['threshold_method']})")
        print(f"Youden J   : {cm['youden_score']:.3f}")
        print(f"Sensitivity: {cm['sensitivity']['mean']:.3f} "
            f"(95 % CI {cm['sensitivity']['ci_lower']:.3f}"
            f"–{cm['sensitivity']['ci_upper']:.3f})")
        print(f"Specificity: {cm['specificity']['mean']:.3f} "
            f"(95 % CI {cm['specificity']['ci_lower']:.3f}"
            f"–{cm['specificity']['ci_upper']:.3f})")
        print(f"PPV        : {cm['ppv']['mean']:.3f} "
            f"(95 % CI {cm['ppv']['ci_lower']:.3f}"
            f"–{cm['ppv']['ci_upper']:.3f})")
        print(f"NPV        : {cm['npv']['mean']:.3f} "
            f"(95 % CI {cm['npv']['ci_lower']:.3f}"
            f"–{cm['npv']['ci_upper']:.3f})")

        
        if 'feature_selection' in results:
            fold_stats = results['fold_analysis']['summary_stats']
            print(f"Features selected (mean): {fold_stats['mean_features_selected']:.1f} ± {fold_stats['std_features_selected']:.1f}")
        
        print(f"\nOutput files generated: {len(output_files)}")
        for file_path in output_files:
            print(f"  - {file_path}")
        
        print(f"\nAnalysis complete! Results saved to: {output_dir}")
        
        return {
            'collector': collector,
            'results': results,
            'tables': tables,
            'metadata': metadata,
            'output_files': output_files
        }
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()