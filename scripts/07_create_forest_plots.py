#!/usr/bin/env python3
"""
07_create_forest_plots.py

Generate Figure 3: Forest plots comparing AUROC with 95% CIs across different analysis scenarios.
Creates side-by-side forest plots for two main analyses showing performance across:
- Main model
- Stability selection of predictors  
- Univariate selection of predictors
- All treatments and lesions
- Logistic regression
- Synthetic data (Analysis 1 only)

Usage:
    python 07_create_forest_plots.py <analysis1_scenarios> <analysis2_scenarios> [options]

Examples:
    python 07_create_forest_plots.py analysis1_scenarios.csv analysis2_scenarios.csv
    python 07_create_forest_plots.py results/analysis1/ results/analysis2/ --analysis1-label "Analysis 1" --analysis2-label "Analysis 2"
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import sys
import warnings

from pneumonitis_ml.evaluation.data_collector import CVDataCollector
from pneumonitis_ml.evaluation.calculate_metrics import calculate_comprehensive_metrics
from pneumonitis_ml.config import REPO_ROOT

warnings.filterwarnings('ignore')

# Set matplotlib style for publication-quality plots
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

def load_scenario_data(scenario_dir):
    """Load data for all scenarios from pre-calculated results CSV files."""
    scenario_path = Path(scenario_dir)
    
    if not scenario_path.is_absolute():
        scenario_path = REPO_ROOT / scenario_path
    
    if not scenario_path.exists():
        raise FileNotFoundError(f"Scenario directory not found: {scenario_path}")
    
    print(f"Loading scenario data from: {scenario_path}")
    
    # Define scenario mapping (subdirectory names to descriptive labels)
    scenario_mapping = {
        'main_model': 'Main model',
        'stability_selection': 'Stability selection of predictors',
        'univariate_selection': 'Univariate selection of predictors', 
        'all_treatments_lesions': 'All treatments and lesions',
        'logistic_regression': 'Logistic regression',
        'synthetic_data': 'Synthetic data'
    }
    
    scenarios_data = {}
    
    # Look for subdirectories and their main_results.csv files
    for subdir in scenario_path.iterdir():
        if not subdir.is_dir():
            continue
            
        # Find the main results CSV file
        results_file = None
        for csv_file in subdir.glob("*_main_results.csv"):
            results_file = csv_file
            break
        
        if not results_file:
            print(f"  Warning: No main_results.csv found in {subdir}")
            continue
            
        # Determine scenario label
        scenario_key = subdir.name.lower()
        scenario_label = scenario_mapping.get(scenario_key, None)
        
        # Try fuzzy matching if exact match not found
        if not scenario_label:
            for key, label in scenario_mapping.items():
                if key in scenario_key or any(word in scenario_key for word in key.split('_')):
                    scenario_label = label
                    break
        
        # Use directory name if no mapping found
        if not scenario_label:
            scenario_label = subdir.name.replace('_', ' ').title()
        
        try:
            print(f"  Loading {scenario_label} from {results_file.name}")
            
            # Read the CSV file
            results_df = pd.read_csv(results_file)
            
            # Extract AUROC row
            auroc_row = results_df[results_df['Metric'] == 'AUROC']
            if len(auroc_row) == 0:
                print(f"    Warning: No AUROC metric found in {results_file}")
                continue
                
            # Parse AUROC value and CI
            auroc_value = float(auroc_row['Value'].iloc[0])
            ci_string = auroc_row['CI_95'].iloc[0]
            
            # Parse CI string like "(0.427-0.693)" 
            ci_clean = ci_string.strip('()')
            if '-' in ci_clean:
                ci_parts = ci_clean.split('-')
                ci_lower = float(ci_parts[0])
                ci_upper = float(ci_parts[1])
            else:
                print(f"    Warning: Could not parse CI string '{ci_string}' in {results_file}")
                continue
            
            scenarios_data[scenario_label] = {
                'auroc_mean': auroc_value,
                'auroc_ci_lower': ci_lower,
                'auroc_ci_upper': ci_upper,
                'source_file': results_file
            }
            
            print(f"    AUROC: {auroc_value:.3f} (95% CI: {ci_lower:.3f}-{ci_upper:.3f})")
            
        except Exception as e:
            print(f"    Warning: Could not load {scenario_label} from {results_file}: {e}")
            continue
    
    if not scenarios_data:
        raise ValueError(f"No valid scenarios found in {scenario_path}")
    
    print(f"  Loaded {len(scenarios_data)} scenarios")
    return scenarios_data

def create_forest_plot(ax, scenarios_data, analysis_label, color='#2E86AB'):
    """Create a forest plot for one analysis."""
    
    # Define the order of scenarios (bottom to top, will be reversed for top-to-bottom display)
    scenario_order = [
        'Synthetic data',
        'Logistic regression',
        'All treatments and lesions', 
        'Univariate selection of predictors',
        'Stability selection of predictors',
        'Main model'
    ]
    
    print(f"  Creating forest plot for {analysis_label}")
    print(f"  Available scenarios: {list(scenarios_data.keys())}")
    
    # Filter to available scenarios and get their positions in the complete order
    available_scenarios = []
    y_positions = []
    
    for i, scenario in enumerate(scenario_order):
        if scenario in scenarios_data:
            available_scenarios.append(scenario)
            y_positions.append(i)  # Use the position from the complete order
    
    print(f"  Scenarios to plot: {available_scenarios}")
    print(f"  Y positions: {y_positions}")
    
    # Extract data for plotting
    auroc_means = []
    ci_lowers = []
    ci_uppers = []
    labels = []
    
    for scenario in available_scenarios:
        data = scenarios_data[scenario]
        auroc_means.append(data['auroc_mean'])
        ci_lowers.append(data['auroc_ci_lower'])
        ci_uppers.append(data['auroc_ci_upper'])
        labels.append(scenario)
    
    print(f"  Y-axis labels: {labels}")
    
    # Create error bars (CI ranges)
    ci_lower_errors = np.array(auroc_means) - np.array(ci_lowers)
    ci_upper_errors = np.array(ci_uppers) - np.array(auroc_means)
    
    # Plot points and error bars
    ax.errorbar(auroc_means, y_positions, 
                xerr=[ci_lower_errors, ci_upper_errors],
                fmt='o', color=color, capsize=5, capthick=2, 
                markersize=8, linewidth=2, markerfacecolor=color, 
                markeredgecolor='white', markeredgewidth=1)
    
    # Customize the plot
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.set_xlabel('AUROC (95% CI)')
    ax.set_title(analysis_label, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add finer vertical grid lines at 0.1 intervals
    ax.set_xticks(np.arange(0, 1.1, 0.1), minor=True)
    ax.grid(True, alpha=0.2, axis='x', which='minor')
    
    # Invert y-axis so Main model appears at top
    ax.invert_yaxis()
    
    # Add vertical reference line at 0.5
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
    
    # Set fixed x-axis limits 0-1
    ax.set_xlim([0.0, 1.0])
    
    # Add value labels above the points (no boxes)
    for i, (mean_val, lower_val, upper_val, y_pos) in enumerate(zip(auroc_means, ci_lowers, ci_uppers, y_positions)):
        # Position text above the point
        ax.text(mean_val, y_pos + 0.1, f'{mean_val:.3f}\n({lower_val:.3f}-{upper_val:.3f})', 
                va='bottom', ha='center', fontsize=9)
    
    return ax

def create_forest_comparison_figure(analysis1_data, analysis2_data, analysis1_label, analysis2_label,
                                  output_path, figure_size=(16, 8), dpi=300):
    """Create side-by-side forest plots comparing two analyses."""
    
    # Use blue color
    color = '#2E86AB'
    
    # Create the complete scenario order for consistent y-axis
    all_scenario_order = [
        'Main model',
        'Stability selection of predictors',
        'Univariate selection of predictors',
        'All treatments and lesions', 
        'Logistic regression',
        'Synthetic data'
    ]
    
    # Create figure with two subplots side by side, but don't share y-axis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figure_size, dpi=dpi)
    
    # Create forest plots
    ax1 = create_forest_plot(ax1, analysis1_data, analysis1_label, color)
    ax2 = create_forest_plot(ax2, analysis2_data, analysis2_label, color)
    
    # Manually ensure both plots have the same y-axis structure
    # Set the same y-limits and ticks for both plots
    max_scenarios = max(len(analysis1_data), len(analysis2_data))
    
    # Set consistent y-axis range
    ax1.set_ylim(-0.5, max_scenarios - 0.5)
    ax2.set_ylim(-0.5, max_scenarios - 0.5)
    
    # Remove y-axis labels from right plot 
    ax2.set_yticklabels([])
    
    # Add panel labels
    ax1.text(-0.1, 1.05, 'A', transform=ax1.transAxes, fontsize=16, fontweight='bold')
    ax2.text(-0.1, 1.05, 'B', transform=ax2.transAxes, fontsize=16, fontweight='bold')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    print(f"Forest plot saved to: {output_path}")
    
    return fig

def print_forest_summary(analysis1_data, analysis2_data, analysis1_label, analysis2_label):
    """Print summary of forest plot data."""
    print("\n" + "="*80)
    print("FOREST PLOT COMPARISON SUMMARY") 
    print("="*80)
    
    print(f"\n{analysis1_label}:")
    print(f"{'Scenario':<35} {'AUROC':<8} {'95% CI'}")
    print("-" * 60)
    for scenario, data in analysis1_data.items():
        mean_val = data['auroc_mean']
        ci_lower = data['auroc_ci_lower'] 
        ci_upper = data['auroc_ci_upper']
        print(f"{scenario:<35} {mean_val:.3f}    ({ci_lower:.3f}-{ci_upper:.3f})")
    
    print(f"\n{analysis2_label}:")
    print(f"{'Scenario':<35} {'AUROC':<8} {'95% CI'}")
    print("-" * 60)
    for scenario, data in analysis2_data.items():
        mean_val = data['auroc_mean']
        ci_lower = data['auroc_ci_lower']
        ci_upper = data['auroc_ci_upper']
        print(f"{scenario:<35} {mean_val:.3f}    ({ci_lower:.3f}-{ci_upper:.3f})")
    
    # Compare common scenarios
    common_scenarios = set(analysis1_data.keys()) & set(analysis2_data.keys())
    if common_scenarios:
        print(f"\nComparison of common scenarios:")
        print(f"{'Scenario':<35} {analysis1_label:<15} {analysis2_label:<15} {'Difference'}")
        print("-" * 80)
        for scenario in sorted(common_scenarios):
            auroc1 = analysis1_data[scenario]['auroc_mean']
            auroc2 = analysis2_data[scenario]['auroc_mean'] 
            diff = auroc1 - auroc2
            print(f"{scenario:<35} {auroc1:.3f}           {auroc2:.3f}           {diff:+.3f}")

def main():
    """Main function for command line interface."""
    
    parser = argparse.ArgumentParser(
        description='Generate Figure 3: Forest plots comparing AUROC across analysis scenarios',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic comparison using directories containing scenario subdirectories
  python 07_create_forest_plots.py results/analysis1_scenarios/ results/analysis2_scenarios/

  # Custom labels
  python 07_create_forest_plots.py results/analysis1/ results/analysis2/ --analysis1-label "Primary Analysis" --analysis2-label "Sensitivity Analysis"

  # Custom output
  python 07_create_forest_plots.py scenarios1/ scenarios2/ --output-filename forest_comparison.png --dpi 600
        """
    )
    
    parser.add_argument(
        'analysis1_scenarios',
        type=str,
        help='Path to directory containing Analysis 1 scenario subdirectories'
    )
    
    parser.add_argument(
        'analysis2_scenarios', 
        type=str,
        help='Path to directory containing Analysis 2 scenario subdirectories'
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
        default='figure3_forest_plot_comparison.png',
        help='Output filename (default: figure3_forest_plot_comparison.png)'
    )
    
    parser.add_argument(
        '--figure-size',
        type=float,
        nargs=2,
        default=[12, 6],
        help='Figure size as width height (default: 16 8)'
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
    print("FIGURE 3: FOREST PLOT COMPARISON")
    print("="*80)
    print(f"Analysis 1 scenarios: {args.analysis1_scenarios}")
    print(f"Analysis 2 scenarios: {args.analysis2_scenarios}")
    print(f"Analysis 1 label: {args.analysis1_label}")
    print(f"Analysis 2 label: {args.analysis2_label}")
    print(f"Output path: {output_path}")
    print(f"Figure size: {args.figure_size[0]} x {args.figure_size[1]} inches")
    print(f"Resolution: {args.dpi} DPI")
    print("="*80)
    
    try:
        # Load scenario data for both analyses
        print("\nLoading scenario data...")
        
        analysis1_data = load_scenario_data(args.analysis1_scenarios)
        analysis2_data = load_scenario_data(args.analysis2_scenarios)
        
        # Print detailed summary
        print_forest_summary(analysis1_data, analysis2_data,
                            args.analysis1_label, args.analysis2_label)
        
        # Create forest plot comparison figure
        print(f"\nGenerating forest plot comparison figure...")
        
        fig = create_forest_comparison_figure(
            analysis1_data, analysis2_data,
            args.analysis1_label, args.analysis2_label,
            output_path, tuple(args.figure_size), args.dpi
        )
        
        print(f"\nFigure 3 generation complete!")
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
        print(f"Error during forest plot generation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()