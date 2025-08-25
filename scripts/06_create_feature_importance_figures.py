#!/usr/bin/env python3
"""
06_create_feature_importance_figures.py

Generate Figure 2: Feature importance and selection frequency comparison plots.
Creates side-by-side horizontal bar charts for two analyses showing:
- Left: Mean selection frequency (top 10 features)
- Right: Absolute mean coefficient/importance (same 10 features)

Usage:
    python 06_create_feature_importance_figures.py <analysis1_dir> <analysis2_dir> [options]

Examples:
    python 06_create_feature_importance_figures.py results/raw_cv_results/analysis1_20231201_143022 results/raw_cv_results/analysis2_20231201_150000
    python 06_create_feature_importance_figures.py results/raw_cv_results/main_analysis results/raw_cv_results/sensitivity_analysis --analysis1-label "Main Analysis" --analysis2-label "Sensitivity Analysis"
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import sys
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
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9

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
    
    feature_selection_df = dfs['feature_selection']
    coefficients_df = dfs['coefficients']
    metadata = collector.metadata
    
    if len(feature_selection_df) == 0:
        raise ValueError(f"No feature selection data found in {analysis_path}")
    
    if len(coefficients_df) == 0:
        raise ValueError(f"No coefficients data found in {analysis_path}")
    
    print(f"  - Loaded {len(feature_selection_df)} feature selection rows")
    print(f"  - Loaded {len(coefficients_df)} coefficient rows")
    
    return feature_selection_df, coefficients_df, metadata

def calculate_feature_stats(feature_selection_df, coefficients_df):
    """Calculate selection frequency and mean coefficient/importance statistics."""
    
    # Calculate selection frequencies
    selection_stats = (feature_selection_df.groupby('feature_name')
                      .agg({
                          'selection_frequency': 'mean',
                          'selected': ['sum', 'count']
                      }))
    
    # Flatten column names
    selection_stats.columns = ['mean_selection_freq', 'times_selected', 'total_folds']
    selection_stats['selection_rate'] = selection_stats['times_selected'] / selection_stats['total_folds']
    
    # Calculate coefficient statistics (excluding intercept)
    coef_stats = (coefficients_df[coefficients_df['feature_name'] != 'intercept']
                 .groupby('feature_name')
                 .agg({
                     'coefficient': ['mean', 'std', 'count']
                 }))
    
    # Flatten column names
    coef_stats.columns = ['mean_coef', 'std_coef', 'n_models']
    coef_stats['abs_mean_coef'] = coef_stats['mean_coef'].abs()
    
    # Merge the two dataframes
    feature_stats = selection_stats.merge(coef_stats, left_index=True, right_index=True, how='inner')
    
    print(f"  - Combined statistics for {len(feature_stats)} features")
    
    return feature_stats

def create_feature_name_mapping():
    """Create mapping from database column names to descriptive variable names."""
    return {
        'num__Age_SBRT': 'Age at SBRT',
        'bin__sex': 'Sex',
        'bin__smoking': 'Former or current smoker',
        'bin__lower lobe =1, others =0': 'Lower-lobe location',
        'bin__lower lobe =1, others = 0': 'Lower-lobe location',
        'bin__peri-RT Systemic_therapy_prior_RT (3M)_yes_1_no_0': 'Peri-RT systemic therapy ≤ 3 mo',
        'num__CCI score': 'Charlson Comorbidity Index score',
        'num__Baseline_FEV1 [L]': 'Baseline FEV1 (L)',
        'num__Baseline_DLCOcSB [%]': 'Baseline DLCOcSB (%)',
        'num__Baseline_DLCOcSB [%] ': 'Baseline DLCOcSB (%)',  # Handle potential trailing space
        'num__V(PTV) [cc]': 'PTV volume (cc)',
        'num__V(PTV1) [cc]': 'PTV volume (cc)',
        'num__total_lung_V_5Gy_(%)': 'Total-lung V5 Gy (%)',
        'num__total_lung_V_10Gy_(%)': 'Total-lung V10 Gy (%)',
        'num__total_lung_V_20Gy_(%)': 'Total-lung V20 Gy (%)',
        'num__total_lung_Dmean_[Gy]_(EQD2)': 'Mean lung dose, EQD2 (Gy)',
        'num__CI50(GI)': 'Conformity index 50 % (CI50/GI)',
        'num__CI100': 'Conformity index 100 % (CI100)',
        'num__D2cm (%)': 'Gradient distance 2 cm (D2 cm, %)'
    }

def map_feature_names(feature_stats):
    """Map database column names to descriptive names."""
    name_mapping = create_feature_name_mapping()
    
    # Create new index with mapped names, keeping original if no mapping exists
    mapped_names = []
    for feature in feature_stats.index:
        mapped_name = name_mapping.get(feature, feature)
        mapped_names.append(mapped_name)
    
    # Create new dataframe with mapped index
    feature_stats_mapped = feature_stats.copy()
    feature_stats_mapped.index = mapped_names
    
    return feature_stats_mapped

def get_top_features(feature_stats, n_features=10):
    """Get top N features by selection frequency and map to descriptive names."""
    # Sort by selection frequency descending, then reverse for display
    top_features = (feature_stats.sort_values('mean_selection_freq', ascending=False)
                   .head(n_features))
    
    # Map to descriptive names
    top_features_mapped = map_feature_names(top_features)
    
    # Reverse the order so highest frequency appears at top when plotted
    top_features_mapped = top_features_mapped.iloc[::-1]
    
    print(f"  - Selected top {len(top_features_mapped)} features by selection frequency")
    print(f"  - Selection frequency range: {top_features_mapped['mean_selection_freq'].min():.3f} - {top_features_mapped['mean_selection_freq'].max():.3f}")
    
    return top_features_mapped

def create_horizontal_bar_chart(ax, data, x_col, title, color, x_label):
    """Create a horizontal bar chart."""
    y_pos = np.arange(len(data))
    
    bars = ax.barh(y_pos, data[x_col], color=color, alpha=0.8)
    
    # Use index as labels (descriptive feature names)
    feature_labels = [name[:50] + '...' if len(name) > 50 else name for name in data.index]
    
    # Customize the plot
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_labels, fontsize=9)
    ax.set_xlabel(x_label)
    ax.set_title(title, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, data[x_col])):
        ax.text(value + 0.01 * max(data[x_col]), bar.get_y() + bar.get_height()/2, 
                f'{value:.3f}', va='center', ha='left', fontsize=7)
    
    # Don't invert y-axis, so highest values are at top naturally
    # ax.invert_yaxis() - removed this line
    
    return ax

def create_feature_importance_figure(analysis1_data, analysis2_data, analysis1_label, analysis2_label,
                                   output_path, n_features=10, figure_size=(14, 10), dpi=300):
    """Create feature importance and selection frequency comparison figure."""
    
    # Use single shade of blue for all plots
    color = '#2E86AB'  # Medium blue
    
    # Create figure with 4 subplots (2x2) with shared y-axes
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figure_size, dpi=dpi, sharey='row')
    
    # Analysis 1 plots (top row)
    ax1 = create_horizontal_bar_chart(
        ax1, analysis1_data, 'mean_selection_freq',
        f'{analysis1_label}: Selection Frequency', color, 'Mean Selection Frequency'
    )
    
    ax2 = create_horizontal_bar_chart(
        ax2, analysis1_data, 'abs_mean_coef',
        f'{analysis1_label}: Feature Importance', color, 'Absolute Mean Coefficient'
    )
    
    # Analysis 2 plots (bottom row)
    ax3 = create_horizontal_bar_chart(
        ax3, analysis2_data, 'mean_selection_freq',
        f'{analysis2_label}: Selection Frequency', color, 'Mean Selection Frequency'
    )
    
    ax4 = create_horizontal_bar_chart(
        ax4, analysis2_data, 'abs_mean_coef',
        f'{analysis2_label}: Feature Importance', color, 'Absolute Mean Coefficient'
    )
    
    # Since we're sharing y-axis within rows, only show labels on left plots
    # But we need to do this after the charts are created
    ax2.tick_params(labelleft=False)
    ax4.tick_params(labelleft=False)
    
    # Add panel labels
    ax1.text(-0.15, 1.05, 'A', transform=ax1.transAxes, fontsize=16, fontweight='bold')
    ax2.text(-0.15, 1.05, 'B', transform=ax2.transAxes, fontsize=16, fontweight='bold')
    ax3.text(-0.15, 1.05, 'C', transform=ax3.transAxes, fontsize=16, fontweight='bold')
    ax4.text(-0.15, 1.05, 'D', transform=ax4.transAxes, fontsize=16, fontweight='bold')
    
    # Adjust layout
    plt.tight_layout(pad=3.0)
    
    # Save figure
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    print(f"Figure saved to: {output_path}")
    
    return fig

def print_feature_summary(analysis1_data, analysis2_data, analysis1_label, analysis2_label, n_features=10):
    """Print summary statistics for both analyses."""
    print("\n" + "="*80)
    print("FEATURE IMPORTANCE AND SELECTION SUMMARY")
    print("="*80)
    
    print(f"\n{analysis1_label} - Top {n_features} Features:")
    print(f"{'Feature':<30} {'Sel. Freq.':<12} {'|Coef|':<12} {'Mean Coef.':<12}")
    print("-" * 70)
    for feature, row in analysis1_data.iterrows():
        print(f"{feature[:28]:<30} {row['mean_selection_freq']:<12.3f} "
              f"{row['abs_mean_coef']:<12.3f} {row['mean_coef']:<12.3f}")
    
    print(f"\n{analysis2_label} - Top {n_features} Features:")
    print(f"{'Feature':<30} {'Sel. Freq.':<12} {'|Coef|':<12} {'Mean Coef.':<12}")
    print("-" * 70)
    for feature, row in analysis2_data.iterrows():
        print(f"{feature[:28]:<30} {row['mean_selection_freq']:<12.3f} "
              f"{row['abs_mean_coef']:<12.3f} {row['mean_coef']:<12.3f}")
    
    # Compare overlapping features
    common_features = set(analysis1_data.index) & set(analysis2_data.index)
    print(f"\nCommon features in top {n_features}: {len(common_features)}")
    if common_features:
        print("Overlapping features:")
        for feature in sorted(common_features):
            freq1 = analysis1_data.loc[feature, 'mean_selection_freq']
            freq2 = analysis2_data.loc[feature, 'mean_selection_freq']
            coef1 = analysis1_data.loc[feature, 'mean_coef']
            coef2 = analysis2_data.loc[feature, 'mean_coef']
            print(f"  {feature[:25]:<25} Freq: {freq1:.3f} vs {freq2:.3f}, "
                  f"Coef: {coef1:+.3f} vs {coef2:+.3f}")

def main():
    """Main function for command line interface."""
    
    parser = argparse.ArgumentParser(
        description='Generate Figure 2: Feature importance and selection frequency comparison',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic comparison
  python 06_create_feature_importance_figures.py results/raw_cv_results/analysis1_20231201_143022 results/raw_cv_results/analysis2_20231201_150000

  # Custom labels and output directory
  python 06_create_feature_importance_figures.py results/raw_cv_results/main_analysis results/raw_cv_results/sensitivity_analysis --analysis1-label "Main Analysis" --analysis2-label "Sensitivity Analysis"

  # Show top 15 features instead of 10
  python 06_create_feature_importance_figures.py analysis1/ analysis2/ --n-features 15
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
        '--n-features',
        type=int,
        default=10,
        help='Number of top features to show (default: 10)'
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
        default='figure2_feature_importance_comparison.png',
        help='Output filename (default: figure2_feature_importance_comparison.png)'
    )
    
    parser.add_argument(
        '--figure-size',
        type=float,
        nargs=2,
        default=[14, 10],
        help='Figure size as width height (default: 14 10)'
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
    
    print("="*80)
    print("FIGURE 2: FEATURE IMPORTANCE AND SELECTION FREQUENCY COMPARISON")
    print("="*80)
    print(f"Analysis 1 directory: {args.analysis1_dir}")
    print(f"Analysis 2 directory: {args.analysis2_dir}")
    print(f"Analysis 1 label: {args.analysis1_label}")
    print(f"Analysis 2 label: {args.analysis2_label}")
    print(f"Number of features to show: {args.n_features}")
    print(f"Output path: {output_path}")
    print(f"Figure size: {args.figure_size[0]} x {args.figure_size[1]} inches")
    print(f"Resolution: {args.dpi} DPI")
    print("="*80)
    
    try:
        # Load data for both analyses
        print("\nLoading analysis data...")
        
        fs1, coef1, metadata1 = load_analysis_data(args.analysis1_dir)
        fs2, coef2, metadata2 = load_analysis_data(args.analysis2_dir)
        
        # Calculate feature statistics
        print("\nCalculating feature statistics...")
        
        print(f"Analysis 1 ({args.analysis1_label}):")
        feature_stats1 = calculate_feature_stats(fs1, coef1)
        top_features1 = get_top_features(feature_stats1, args.n_features)
        
        print(f"\nAnalysis 2 ({args.analysis2_label}):")
        feature_stats2 = calculate_feature_stats(fs2, coef2)
        top_features2 = get_top_features(feature_stats2, args.n_features)
        
        # Print detailed summary
        print_feature_summary(top_features1, top_features2, 
                            args.analysis1_label, args.analysis2_label, args.n_features)
        
        # Create comparison figure
        print(f"\nGenerating feature importance comparison figure...")
        
        fig = create_feature_importance_figure(
            top_features1, top_features2,
            args.analysis1_label, args.analysis2_label,
            output_path, args.n_features, tuple(args.figure_size), args.dpi
        )
        
        print(f"\nFigure 2 generation complete!")
        print(f"Output saved to: {output_path}")
        
        # Show the plot if running interactively
        try:
            plt.show()
        except:
            pass  # Ignore if not in interactive mode
            
        return {
            'figure': fig,
            'analysis1_features': top_features1,
            'analysis2_features': top_features2,
            'output_path': output_path
        }
        
    except Exception as e:
        print(f"Error during figure generation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()