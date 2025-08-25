#!/usr/bin/env python3
"""
05_create_figures.py

Generate Figure 1: Side-by-side AUROC and AUPRC comparison plots for two analyses.
Creates publication-ready figures comparing the performance of two main analyses.

Usage:
    python 05_create_figures.py <analysis1_dir> <analysis2_dir> [options]

Examples:
    python 05_create_figures.py results/raw_cv_results/analysis1_20231201_143022 results/raw_cv_results/analysis2_20231201_150000
    python 05_create_figures.py results/raw_cv_results/main_analysis results/raw_cv_results/sensitivity_analysis --output-dir figures --analysis1-label "Main Analysis" --analysis2-label "Sensitivity Analysis"
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import sys
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import warnings

from pneumonitis_ml.evaluation.data_collector import CVDataCollector
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

def load_analysis_data(analysis_dir):
    """Load CV results from an analysis directory."""
    analysis_path = Path(analysis_dir)
    
    if not analysis_path.is_absolute():
        analysis_path = REPO_ROOT / analysis_path
    
    if not analysis_path.exists():
        raise FileNotFoundError(f"Analysis directory not found: {analysis_path}")
    
    if not analysis_path.is_dir():
        raise ValueError(f"Path is not a directory: {analysis_path}")
    
    print(f"Loading data from: {analysis_path}")
    
    # Initialize collector and load data
    collector = CVDataCollector()
    
    try:
        dfs = collector.load_data(str(analysis_path))
    except Exception as e:
        # If that fails, try with the base filename (directory name)
        base_filename = analysis_path / analysis_path.name
        dfs = collector.load_data(str(base_filename))
    
    predictions_df = dfs['predictions']
    coefficients_df = dfs['coefficients'] 
    feature_selection_df = dfs['feature_selection']
    fold_summaries_df = dfs['fold_summaries']
    metadata = collector.metadata
    
    if len(predictions_df) == 0:
        raise ValueError(f"No predictions data found in {analysis_path}")
    
    print(f"  - Loaded {len(predictions_df)} prediction rows")
    print(f"  - Loaded {len(coefficients_df)} coefficient rows")
    print(f"  - Loaded {len(feature_selection_df)} feature selection rows") 
    print(f"  - Loaded {len(fold_summaries_df)} fold summary rows")
    
    return predictions_df, coefficients_df, feature_selection_df, fold_summaries_df, metadata

def calculate_fold_curves(predictions_df, coefficients_df, feature_selection_df, fold_summaries_df):
    """Calculate ROC and PR curves for each fold using the existing calculate_comprehensive_metrics function."""
    from pneumonitis_ml.evaluation.calculate_metrics import calculate_comprehensive_metrics
    
    # Use the existing comprehensive metrics calculation to get consistent results
    results = calculate_comprehensive_metrics(
        predictions_df, coefficients_df, feature_selection_df, fold_summaries_df
    )
    
    # Extract test data for fold-level curve calculations
    test_data = predictions_df[predictions_df['dataset'] == 'test'].copy()
    
    if len(test_data) == 0:
        raise ValueError("No test data found in predictions DataFrame")
    
    print(f"  - Using {len(test_data)} test predictions")
    print(f"  - Available columns: {list(test_data.columns)}")
    
    fold_roc_curves = {}
    fold_pr_curves = {}
    fold_aurocs = []
    fold_auprcs = []
    
    # Get unique folds
    folds = sorted(test_data['fold'].unique())
    
    for fold in folds:
        fold_data = test_data[test_data['fold'] == fold]
        
        if len(fold_data) == 0:
            continue
        
        # Aggregate to patient level for this fold (consistent with existing approach)
        patient_fold_data = (fold_data.groupby('patient_id')
                           .agg(y_true=('y_true', 'max'),
                                y_pred_proba=('y_pred_proba', 'mean'))
                           .reset_index())
        
        if len(patient_fold_data) == 0:
            continue
            
        y_true = patient_fold_data['y_true'].values
        y_prob = patient_fold_data['y_pred_proba'].values
        
        # Skip if no positive or negative cases
        if len(np.unique(y_true)) < 2:
            continue
        
        # ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        fold_auroc = auc(fpr, tpr)
        fold_roc_curves[fold] = {'fpr': fpr, 'tpr': tpr, 'auroc': fold_auroc}
        fold_aurocs.append(fold_auroc)
        
        # PR curve
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        fold_auprc = auc(recall, precision)
        fold_pr_curves[fold] = {'recall': recall, 'precision': precision, 'auprc': fold_auprc}
        fold_auprcs.append(fold_auprc)
    
    # Use the metrics from the comprehensive calculation for consistency
    performance_metrics = results['performance_metrics']
    
    return {
        'roc_curves': fold_roc_curves,
        'pr_curves': fold_pr_curves,
        'aurocs': fold_aurocs,
        'auprcs': fold_auprcs,
        'mean_auroc': performance_metrics['auroc']['mean'],
        'auroc_ci_lower': performance_metrics['auroc']['ci_lower'],
        'auroc_ci_upper': performance_metrics['auroc']['ci_upper'],
        'mean_auprc': performance_metrics['auprc']['mean'],
        'auprc_ci_lower': performance_metrics['auprc']['ci_lower'],
        'auprc_ci_upper': performance_metrics['auprc']['ci_upper'],
        'comprehensive_results': results
    }

def plot_roc_curves(ax, curves_data, label, color, alpha=0.3):
    """Plot ROC curves for all folds with mean curve."""
    roc_curves = curves_data['roc_curves']
    aurocs = curves_data['aurocs']
    mean_auroc = curves_data['mean_auroc']
    auroc_ci_lower = curves_data['auroc_ci_lower']
    auroc_ci_upper = curves_data['auroc_ci_upper']
    
    # Plot individual fold curves (lighter)
    for fold, curve_data in roc_curves.items():
        ax.plot(curve_data['fpr'], curve_data['tpr'], 
               color=color, alpha=alpha, linewidth=0.8)
    
    # Calculate mean ROC curve using interpolation
    # Create common FPR points
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    
    for fold, curve_data in roc_curves.items():
        # Interpolate TPR at common FPR points
        interp_tpr = np.interp(mean_fpr, curve_data['fpr'], curve_data['tpr'])
        interp_tpr[0] = 0.0  # Ensure it starts at (0,0)
        tprs.append(interp_tpr)
    
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0  # Ensure it ends at (1,1)
    
    # Plot mean curve (bold) with bootstrap 95% CI
    ax.plot(mean_fpr, mean_tpr, color=color, linewidth=2.5,
           label=f'{label} (AUROC = {mean_auroc:.3f}, 95% CI: {auroc_ci_lower:.3f}-{auroc_ci_upper:.3f})')
    
    # Plot diagonal reference line
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1)
    
    return ax

def plot_pr_curves(ax, curves_data, label, color, alpha=0.3):
    """Plot Precision-Recall curves for all folds with mean curve."""
    pr_curves = curves_data['pr_curves']
    auprcs = curves_data['auprcs']
    mean_auprc = curves_data['mean_auprc']
    auprc_ci_lower = curves_data['auprc_ci_lower']
    auprc_ci_upper = curves_data['auprc_ci_upper']
    
    # Plot individual fold curves (lighter)
    for fold, curve_data in pr_curves.items():
        ax.plot(curve_data['recall'], curve_data['precision'], 
               color=color, alpha=alpha, linewidth=0.8)
    
    # Calculate mean PR curve using interpolation
    # Create common recall points
    mean_recall = np.linspace(0, 1, 100)
    precisions = []
    
    for fold, curve_data in pr_curves.items():
        # Reverse arrays for proper interpolation (recall should be decreasing)
        recall_rev = curve_data['recall'][::-1]
        precision_rev = curve_data['precision'][::-1]
        
        # Interpolate precision at common recall points
        interp_precision = np.interp(mean_recall, recall_rev, precision_rev)
        precisions.append(interp_precision)
    
    mean_precision = np.mean(precisions, axis=0)
    
    # Plot mean curve (bold) with bootstrap 95% CI
    ax.plot(mean_recall, mean_precision, color=color, linewidth=2.5,
           label=f'{label} (AUPRC = {mean_auprc:.3f}, 95% CI: {auprc_ci_lower:.3f}-{auprc_ci_upper:.3f})')
    
    return ax

def create_comparison_figure(analysis1_data, analysis2_data, analysis1_label, analysis2_label, 
                           output_path, figure_size=(12, 5), dpi=300):
    """Create side-by-side AUROC and AUPRC comparison figure."""
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figure_size, dpi=dpi)
    
    # Define colors for the two analyses
    color1 = '#1f77b4'  # Blue
    color2 = '#ff7f0e'  # Orange
    
    # Plot ROC curves
    ax1 = plot_roc_curves(ax1, analysis1_data, analysis1_label, color1)
    ax1 = plot_roc_curves(ax1, analysis2_data, analysis2_label, color2)
    
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curves')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    
    # Plot PR curves
    ax2 = plot_pr_curves(ax2, analysis1_data, analysis1_label, color1)
    ax2 = plot_pr_curves(ax2, analysis2_data, analysis2_label, color2)
    
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curves')
    ax2.legend(loc='lower left')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])
    
    # Add panel labels
    ax1.text(-0.1, 1.05, 'A', transform=ax1.transAxes, fontsize=16, fontweight='bold')
    ax2.text(-0.1, 1.05, 'B', transform=ax2.transAxes, fontsize=16, fontweight='bold')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    print(f"Figure saved to: {output_path}")
    
    return fig

def print_summary_stats(analysis1_data, analysis2_data, analysis1_label, analysis2_label):
    """Print summary statistics for both analyses."""
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON SUMMARY")
    print("="*60)
    
    print(f"\n{analysis1_label}:")
    print(f"  AUROC: {analysis1_data['mean_auroc']:.3f} (95% CI: {analysis1_data['auroc_ci_lower']:.3f}-{analysis1_data['auroc_ci_upper']:.3f})")
    print(f"  AUPRC: {analysis1_data['mean_auprc']:.3f} (95% CI: {analysis1_data['auprc_ci_lower']:.3f}-{analysis1_data['auprc_ci_upper']:.3f})")
    print(f"  Number of folds: {len(analysis1_data['aurocs'])}")
    
    print(f"\n{analysis2_label}:")
    print(f"  AUROC: {analysis2_data['mean_auroc']:.3f} (95% CI: {analysis2_data['auroc_ci_lower']:.3f}-{analysis2_data['auroc_ci_upper']:.3f})")
    print(f"  AUPRC: {analysis2_data['mean_auprc']:.3f} (95% CI: {analysis2_data['auprc_ci_lower']:.3f}-{analysis2_data['auprc_ci_upper']:.3f})")
    print(f"  Number of folds: {len(analysis2_data['aurocs'])}")
    
    # Calculate differences
    auroc_diff = analysis1_data['mean_auroc'] - analysis2_data['mean_auroc']
    auprc_diff = analysis1_data['mean_auprc'] - analysis2_data['mean_auprc']
    
    print(f"\nDifferences ({analysis1_label} - {analysis2_label}):")
    print(f"  ΔAUROC: {auroc_diff:+.3f}")
    print(f"  ΔAUPRC: {auprc_diff:+.3f}")

def main():
    """Main function for command line interface."""
    
    parser = argparse.ArgumentParser(
        description='Generate Figure 1: AUROC and AUPRC comparison plots',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic comparison
  python 05_create_figures.py results/raw_cv_results/analysis1_20231201_143022 results/raw_cv_results/analysis2_20231201_150000

  # Custom labels and output directory
  python 05_create_figures.py results/raw_cv_results/main_analysis results/raw_cv_results/sensitivity_analysis --output-dir figures --analysis1-label "Main Analysis" --analysis2-label "Sensitivity Analysis"

  # High-resolution figure
  python 05_create_figures.py analysis1/ analysis2/ --dpi 600 --figure-size 14 6
        """
    )
    
    parser.add_argument(
        'analysis1_dir',
        type=str,
        help='Path to first analysis directory containing CV results'
    )
    
    parser.add_argument(
        'analysis2_dir',
        type=str,
        help='Path to second analysis directory containing CV results'
    )
    
    parser.add_argument(
        '--analysis1-label',
        type=str,
        default='Analysis 1',
        help='Label for first analysis (default: Analysis 1)'
    )
    
    parser.add_argument(
        '--analysis2-label',
        type=str,
        default='Analysis 2',
        help='Label for second analysis (default: Analysis 2)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='figures',
        help='Output directory for figure (default: figures)'
    )
    
    parser.add_argument(
        '--output-filename',
        type=str,
        default='figure1_auroc_auprc_comparison.png',
        help='Output filename (default: figure1_auroc_auprc_comparison.png)'
    )
    
    parser.add_argument(
        '--figure-size',
        type=float,
        nargs=2,
        default=[12, 5],
        help='Figure size as width height (default: 12 5)'
    )
    
    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='Figure resolution in DPI (default: 300)'
    )
    
    args = parser.parse_args()
    
    # Set up output directory
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.output_filename
    
    print("="*60)
    print("FIGURE 1: AUROC AND AUPRC COMPARISON")
    print("="*60)
    print(f"Analysis 1 directory: {args.analysis1_dir}")
    print(f"Analysis 2 directory: {args.analysis2_dir}")
    print(f"Analysis 1 label: {args.analysis1_label}")
    print(f"Analysis 2 label: {args.analysis2_label}")
    print(f"Output path: {output_path}")
    print(f"Figure size: {args.figure_size[0]} x {args.figure_size[1]} inches")
    print(f"Resolution: {args.dpi} DPI")
    print("="*60)
    
    try:
        # Load data for both analyses
        print("\nLoading analysis data...")
        
        predictions1, coefficients1, feature_selection1, fold_summaries1, metadata1 = load_analysis_data(args.analysis1_dir)
        predictions2, coefficients2, feature_selection2, fold_summaries2, metadata2 = load_analysis_data(args.analysis2_dir)
        
        # Calculate curves for both analyses
        print("\nCalculating performance curves...")
        
        analysis1_data = calculate_fold_curves(predictions1, coefficients1, feature_selection1, fold_summaries1)
        analysis2_data = calculate_fold_curves(predictions2, coefficients2, feature_selection2, fold_summaries2)
        
        # Print summary statistics
        print_summary_stats(analysis1_data, analysis2_data, 
                          args.analysis1_label, args.analysis2_label)
        
        # Create comparison figure
        print(f"\nGenerating comparison figure...")
        
        fig = create_comparison_figure(
            analysis1_data, analysis2_data,
            args.analysis1_label, args.analysis2_label,
            output_path, tuple(args.figure_size), args.dpi
        )
        
        print(f"\nFigure 1 generation complete!")
        print(f"Output saved to: {output_path}")
        
        # Show the plot if running interactively
        try:
            plt.show()
        except:
            pass  # Ignore if not in interactive mode
            
        return {
            'figure': fig,
            'analysis1_data': analysis1_data,
            'analysis2_data': analysis2_data,
            'output_path': output_path
        }
        
    except Exception as e:
        print(f"Error during figure generation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()