import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve
from pathlib import Path
from datetime import datetime


def create_publication_tables(results, metadata):
    """Create publication-ready tables."""
    tables = {}
    
    # Table 1: Main Results (for paper)
    perf = results['performance_metrics']
    calib = results['calibration']
    cm = results['classification_metrics']
    
    tables['main_results'] = pd.DataFrame({
        'Metric': [
            'AUROC',
            'AUPRC', 
            'Sensitivity',
            'Specificity',
            'PPV',
            'NPV',
            'Threshold',
            'Youden Score',
            'Brier Score',
            'Calibration Slope'
        ],
        'Value': [
            f"{perf['auroc']['mean']:.3f}",
            f"{perf['auprc']['mean']:.3f}",
            f"{cm['sensitivity']:.3f}",
            f"{cm['specificity']:.3f}",
            f"{cm['ppv']:.3f}",
            f"{cm['npv']:.3f}",
            f"{cm['threshold']:.3f}",
            f"{cm['youden_score']:.3f}",
            f"{calib['brier_score']:.3f}",
            f"{calib['calibration_slope']:.3f}"
        ],
        'CI_95': [
            f"({perf['auroc']['ci_lower']:.3f}-{perf['auroc']['ci_upper']:.3f})",
            f"({perf['auprc']['ci_lower']:.3f}-{perf['auprc']['ci_upper']:.3f})",
            "-", "-", "-", "-", "-", "-", "-", "-"
        ],
        'Notes': [
            "Bootstrap CI",
            "Bootstrap CI", 
            f"At {cm['threshold_method']} threshold",
            f"At {cm['threshold_method']} threshold",
            f"At {cm['threshold_method']} threshold",
            f"At {cm['threshold_method']} threshold",
            f"{cm['threshold_method'].title()} optimal",
            "Sensitivity + Specificity - 1",
            "Lower is better",
            "Should be close to 1"
        ]
    })
    
    # Table 2: Top Features (for paper)
    if 'feature_selection' in results:
        feat_sel = results['feature_selection']['summary']
        coef_sum = results['coefficients']['summary']
        
        combined_features = feat_sel.join(coef_sum[['mean_coef', 'std_coef']], how='outer').fillna(0)
        top_features = combined_features.sort_values('mean_selection_freq', ascending=False).head(10)
        
        tables['top_features'] = pd.DataFrame({
            'Feature': top_features.index,
            'Selection_Rate': [f"{x:.3f}" for x in top_features['selection_rate']],
            'Mean_Coefficient': [f"{x:.3f}" for x in top_features['mean_coef']],
            'Std_Coefficient': [f"{x:.3f}" for x in top_features['std_coef']]
        })
    
    # Table 3: Experiment Configuration (for methods/supplementary)
    if 'experiment_info' in metadata:
        exp_info = metadata['experiment_info']
        
        config_data = []
        key_params = [
            ('Analysis', 'analysis_number'),
            ('CV Folds', 'n_splits'),
            ('CV Repeats', 'n_repeats'),
            ('Stability Selection', 'use_stability_selection'),
            ('Stability Threshold', 'stability_threshold'),
            ('Lambda Values', 'lambda_values'),
            ('Random Seed', 'random_seed'),
            ('Total Samples', 'total_samples'),
            ('Total Patients', 'total_patients'),
            ('Target Prevalence', 'target_prevalence')
        ]
        
        for param_name, key in key_params:
            if key in exp_info:
                value = exp_info[key]
                if isinstance(value, list):
                    value = ', '.join(map(str, value))
                elif isinstance(value, float):
                    value = f"{value:.3f}"
                config_data.append({'Parameter': param_name, 'Value': str(value)})
        
        tables['experiment_config'] = pd.DataFrame(config_data)
    
    # Table 4: Detailed Performance by Fold (for supplementary)
    fold_perf = results['fold_analysis']['fold_performance']
    if len(fold_perf) > 0:
        tables['fold_performance'] = fold_perf.copy()
    
    return tables

def create_publication_figures(results, output_dir, experiment_name):
    """Create publication-quality figures."""
    figures = {}
    output_dir = Path(output_dir)
    
    # Figure 1: Main Performance Plot (2x2 grid)
    fig1, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig1.suptitle(f'Model Performance: {experiment_name}', fontsize=16, fontweight='bold')
    
    # ROC Curve
    ax1 = axes[0, 0]
    test_preds = results['test_predictions']
    fpr, tpr, _ = roc_curve(test_preds['y_true'], test_preds['y_pred_proba'])
    auroc = results['performance_metrics']['auroc']['mean']
    auroc_ci = results['performance_metrics']['auroc']
    
    ax1.plot(fpr, tpr, linewidth=2, label=f'AUROC = {auroc:.3f}\n(95% CI: {auroc_ci["ci_lower"]:.3f}-{auroc_ci["ci_upper"]:.3f})')
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curve')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Precision-Recall Curve
    ax2 = axes[0, 1]
    precision, recall, _ = precision_recall_curve(test_preds['y_true'], test_preds['y_pred_proba'])
    auprc = results['performance_metrics']['auprc']['mean']
    auprc_ci = results['performance_metrics']['auprc']
    
    ax2.plot(recall, precision, linewidth=2, label=f'AUPRC = {auprc:.3f}\n(95% CI: {auprc_ci["ci_lower"]:.3f}-{auprc_ci["ci_upper"]:.3f})')
    ax2.axhline(y=results['decision_curve']['prevalence'], color='k', linestyle='--', alpha=0.5, label='Baseline')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Calibration Plot
    ax3 = axes[1, 0]
    calib_data = results['calibration']['calibration_curve']
    ax3.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')
    ax3.plot(calib_data['mean_predicted_value'], calib_data['fraction_of_positives'], 
             'o-', linewidth=2, markersize=8, label='Model')
    ax3.set_xlabel('Mean Predicted Probability')
    ax3.set_ylabel('Fraction of Positives')
    ax3.set_title(f'Calibration Plot\nBrier Score: {results["calibration"]["brier_score"]:.3f}')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Decision Curve Analysis
    ax4 = axes[1, 1]
    dca_data = results['decision_curve']
    ax4.plot(dca_data['thresholds'], dca_data['net_benefits'], 'b-', linewidth=2, label='Model')
    ax4.plot(dca_data['thresholds'], dca_data['treat_all_nb'], 'r--', alpha=0.7, label='Treat All')
    ax4.plot(dca_data['thresholds'], dca_data['treat_none_nb'], 'k--', alpha=0.7, label='Treat None')
    ax4.set_xlabel('Risk Threshold')
    ax4.set_ylabel('Net Benefit')
    ax4.set_title('Decision Curve Analysis')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 1)
    
    plt.tight_layout()
    fig1_path = output_dir / f'{experiment_name}_main_performance.png'
    fig1.savefig(fig1_path, dpi=300, bbox_inches='tight')
    figures['main_performance'] = fig1_path
    plt.close(fig1)
    
    # Figure 2: Feature Importance
    if 'feature_selection' in results:
        fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig2.suptitle(f'Feature Analysis: {experiment_name}', fontsize=16, fontweight='bold')
        
        # Selection Frequencies
        feat_data = results['feature_selection']['summary'].head(10)
        y_pos = np.arange(len(feat_data))
        ax1.barh(y_pos, feat_data['mean_selection_freq'], alpha=0.7)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels([name[:25] + '...' if len(name) > 25 else name for name in feat_data.index], fontsize=10)
        ax1.set_xlabel('Mean Selection Frequency')
        ax1.set_title('Top 10 Features by Selection Frequency')
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Coefficient Magnitudes
        coef_data = results['coefficients']['summary']
        coef_viz = coef_data[coef_data.index != 'intercept'].head(10)
        y_pos = np.arange(len(coef_viz))
        colors = ['red' if x < 0 else 'blue' for x in coef_viz['mean_coef']]
        ax2.barh(y_pos, coef_viz['mean_coef'].abs(), alpha=0.7, color=colors)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels([name[:25] + '...' if len(name) > 25 else name for name in coef_viz.index], fontsize=10)
        ax2.set_xlabel('|Mean Coefficient|')
        ax2.set_title('Top 10 Features by Coefficient Magnitude')
        ax2.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        fig2_path = output_dir / f'{experiment_name}_feature_analysis.png'
        fig2.savefig(fig2_path, dpi=300, bbox_inches='tight')
        figures['feature_analysis'] = fig2_path
        plt.close(fig2)
    
    return figures

def generate_text_summary(results, metadata, tables, output_file):
    """Generate a comprehensive text summary for the paper."""
    
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("COMPREHENSIVE CROSS-VALIDATION RESULTS SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        # Experiment Information
        if 'experiment_info' in metadata:
            exp_info = metadata['experiment_info']
            f.write("EXPERIMENT CONFIGURATION\n")
            f.write("-" * 40 + "\n")
            f.write(f"Experiment Name: {exp_info.get('name', 'Unknown')}\n")
            f.write(f"Analysis: {exp_info.get('analysis_number', 'Unknown')}\n")
            f.write(f"Dataset: {Path(exp_info.get('dataset_file', 'Unknown')).name}\n")
            f.write(f"Cross-Validation: {exp_info.get('n_repeats', 'Unknown')} repeats × {exp_info.get('n_splits', 'Unknown')} folds\n")
            f.write(f"Stability Selection: {exp_info.get('use_stability_selection', 'Unknown')}\n")
            if exp_info.get('use_stability_selection'):
                f.write(f"  - Threshold: {exp_info.get('stability_threshold', 'Unknown')}\n")
                f.write(f"  - Iterations: {exp_info.get('stability_iterations', 'Unknown')}\n")
            f.write(f"Total Samples: {exp_info.get('total_samples', 'Unknown')}\n")
            f.write(f"Total Patients: {exp_info.get('total_patients', 'Unknown')}\n")
            f.write(f"Target Prevalence: {exp_info.get('target_prevalence', 'Unknown'):.3f}\n")
            f.write(f"Random Seed: {exp_info.get('random_seed', 'Unknown')}\n\n")
        
        # Main Results (for paper text)
        perf = results['performance_metrics']
        calib = results['calibration']
        cm = results['classification_metrics']
        
        f.write("MAIN RESULTS\n")
        f.write("-" * 40 + "\n")
        f.write(f"AUROC: {perf['auroc']['mean']:.3f} (95% CI: {perf['auroc']['ci_lower']:.3f}-{perf['auroc']['ci_upper']:.3f})\n")
        f.write(f"AUPRC: {perf['auprc']['mean']:.3f} (95% CI: {perf['auprc']['ci_lower']:.3f}-{perf['auprc']['ci_upper']:.3f})\n\n")
        
        f.write("CLASSIFICATION PERFORMANCE (YOUDEN-OPTIMAL THRESHOLD)\n")
        f.write(f"Optimal Threshold: {cm['threshold']:.3f} ({cm['threshold_method']} method)\n")
        f.write(f"Youden Score (J): {cm['youden_score']:.3f}\n")
        f.write(f"Sensitivity: {cm['sensitivity']:.3f}\n")
        f.write(f"Specificity: {cm['specificity']:.3f}\n")
        f.write(f"PPV: {cm['ppv']:.3f}\n")
        f.write(f"NPV: {cm['npv']:.3f}\n\n")
        
        f.write("CALIBRATION METRICS\n")
        f.write(f"Brier Score: {calib['brier_score']:.3f}\n")
        f.write(f"Calibration Slope: {calib['calibration_slope']:.3f}\n")
        f.write(f"Calibration Intercept: {calib['calibration_intercept']:.3f}\n\n")
        
        # Feature Selection Summary
        if 'feature_selection' in results:
            fold_stats = results['fold_analysis']['summary_stats']
            f.write("FEATURE SELECTION SUMMARY\n")
            f.write("-" * 40 + "\n")
            f.write(f"Valid CV Folds: {fold_stats['n_valid_folds']}/{fold_stats['n_total_folds']}\n")
            f.write(f"Mean Features Selected: {fold_stats['mean_features_selected']:.1f} ± {fold_stats['std_features_selected']:.1f}\n\n")
            
            # Top features
            feat_sel = results['feature_selection']['summary']
            top_features = feat_sel.sort_values('mean_selection_freq', ascending=False).head(5)
            f.write("TOP 5 SELECTED FEATURES:\n")
            for i, (feature, row) in enumerate(top_features.iterrows(), 1):
                f.write(f"{i}. {feature} (selected {row['selection_rate']:.1%} of folds)\n")
            f.write("\n")
        
        # Performance by Fold (brief summary)
        fold_perf = results['fold_analysis']['fold_performance']
        if len(fold_perf) > 0:
            f.write("CROSS-VALIDATION STABILITY\n")
            f.write("-" * 40 + "\n")
            f.write(f"All {len(fold_perf)} folds completed successfully\n")
            f.write(f"Feature selection range: {fold_perf['n_features_selected'].min()}-{fold_perf['n_features_selected'].max()} features\n\n")
        
        f.write("="*80 + "\n")
        f.write(f"Summary generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n")