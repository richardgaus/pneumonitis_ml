import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    roc_curve, confusion_matrix
)
from sklearn.calibration import calibration_curve


def bootstrap_ci(data, statistic_func, n_bootstrap=1000, confidence=0.95, random_state=42):
    """Calculate bootstrap confidence intervals for a statistic."""
    np.random.seed(random_state)
    bootstrap_stats = []
    
    for _ in range(n_bootstrap):
        indices = np.random.choice(len(data), size=len(data), replace=True)
        bootstrap_sample = data.iloc[indices]
        bootstrap_stats.append(statistic_func(bootstrap_sample))
    
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_stats, (alpha/2) * 100)
    upper = np.percentile(bootstrap_stats, (1 - alpha/2) * 100)
    
    return np.array(bootstrap_stats), lower, upper

def find_optimal_threshold(y_true, y_pred_proba, method='youden'):
    """Find optimal threshold using different methods."""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    
    if method == 'youden':
        # Youden's J statistic (sensitivity + specificity - 1)
        youden_scores = tpr - fpr
        optimal_idx = np.argmax(youden_scores)
        optimal_threshold = thresholds[optimal_idx]
        youden_score = youden_scores[optimal_idx]
    elif method == 'closest_topleft':
        # Point closest to top-left corner (0, 1)
        distances = np.sqrt((fpr - 0)**2 + (tpr - 1)**2)
        optimal_idx = np.argmin(distances)
        optimal_threshold = thresholds[optimal_idx]
        youden_score = tpr[optimal_idx] - fpr[optimal_idx]
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return optimal_threshold, youden_score

def calculate_comprehensive_metrics(df_predictions, df_coefficients, df_feature_selection, 
                                  df_fold_summaries, optimal_threshold=None):
    """Calculate comprehensive metrics from raw CV results."""
    
    results = {}
    # Collapse to patient level once, reuse everywhere
    pat = (df_predictions[df_predictions['dataset']=='test']
        .groupby('patient_id')
        .agg(y_true=('y_true','max'),
             y_pred_proba=('y_pred_proba','max'))
        .reset_index())
    test_preds = pat     # downstream code now uses patient-level
    
    if len(test_preds) == 0:
        print("No test predictions found!")
        return results
    
    print(f"\nCalculating metrics from {len(test_preds)} test patients")
    results['test_predictions'] = test_preds
    
    # AUROC & AUPRC with 95% CIs
    print("Computing AUROC and AUPRC with bootstrap CIs...")
    
    def auroc_func(data):
        return roc_auc_score(data['y_true'], data['y_pred_proba'])
    
    def auprc_func(data):
        return average_precision_score(data['y_true'], data['y_pred_proba'])
    
    auroc_bootstrap, auroc_lower, auroc_upper = bootstrap_ci(test_preds, auroc_func)
    auroc_mean = np.mean(auroc_bootstrap)
    
    auprc_bootstrap, auprc_lower, auprc_upper = bootstrap_ci(test_preds, auprc_func)
    auprc_mean = np.mean(auprc_bootstrap)
    
    results['performance_metrics'] = {
        'auroc': {'mean': auroc_mean, 'ci_lower': auroc_lower, 'ci_upper': auroc_upper, 'bootstrap_values': auroc_bootstrap},
        'auprc': {'mean': auprc_mean, 'ci_lower': auprc_lower, 'ci_upper': auprc_upper, 'bootstrap_values': auprc_bootstrap}
    }
    
    print(f"AUROC: {auroc_mean:.4f} (95% CI: {auroc_lower:.4f}-{auroc_upper:.4f})")
    print(f"AUPRC: {auprc_mean:.4f} (95% CI: {auprc_lower:.4f}-{auprc_upper:.4f})")
    
    # Find optimal threshold if not provided
    if optimal_threshold is None:
        optimal_threshold, youden_score = find_optimal_threshold(test_preds['y_true'], test_preds['y_pred_proba'], method='youden')
        threshold_method = 'youden'
        print(f"Optimal threshold (Youden): {optimal_threshold:.4f} (J = {youden_score:.4f})")
    else:
        # Calculate Youden score for the given threshold
        test_preds_thresh = (test_preds['y_pred_proba'] >= optimal_threshold).astype(int)
        cm_temp = confusion_matrix(test_preds['y_true'], test_preds_thresh)
        tn, fp, fn, tp = cm_temp.ravel()
        sensitivity_temp = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity_temp = tn / (tn + fp) if (tn + fp) > 0 else 0
        youden_score = sensitivity_temp + specificity_temp - 1
        threshold_method = 'manual'
        print(f"Using manual threshold: {optimal_threshold:.4f} (J = {youden_score:.4f})")
    
    # Calibration Metrics
    print("Computing calibration metrics...")
    
    fraction_of_positives, mean_predicted_value = calibration_curve(
        test_preds['y_true'], test_preds['y_pred_proba'], n_bins=10
    )
    
    from sklearn.linear_model import LogisticRegression
    logit_preds = np.log(test_preds['y_pred_proba'] / (1 - test_preds['y_pred_proba'] + 1e-15))
    calib_model = LogisticRegression(fit_intercept=True)
    calib_model.fit(logit_preds.values.reshape(-1, 1), test_preds['y_true'])
    
    calibration_intercept = calib_model.intercept_[0]
    calibration_slope = calib_model.coef_[0][0]
    brier_score = brier_score_loss(test_preds['y_true'], test_preds['y_pred_proba'])
    
    results['calibration'] = {
        'brier_score': brier_score,
        'calibration_intercept': calibration_intercept,
        'calibration_slope': calibration_slope,
        'calibration_curve': {
            'fraction_of_positives': fraction_of_positives,
            'mean_predicted_value': mean_predicted_value
        }
    }
    
    # Decision Curve Analysis
    print("Computing decision curve analysis...")
    
    def net_benefit(y_true, y_pred_proba, threshold):
        y_pred_binary = (y_pred_proba >= threshold).astype(int)
        tp = np.sum((y_true == 1) & (y_pred_binary == 1))
        fp = np.sum((y_true == 0) & (y_pred_binary == 1))
        n = len(y_true)
        return (tp / n) - (fp / n) * (threshold / (1 - threshold))
    
    thresholds = np.linspace(0.01, 0.99, 99)
    net_benefits = [net_benefit(test_preds['y_true'], test_preds['y_pred_proba'], t) for t in thresholds]
    
    prevalence = test_preds['y_true'].mean()
    treat_all_nb = [prevalence - (1 - prevalence) * (t / (1 - t)) for t in thresholds]
    treat_none_nb = [0] * len(thresholds)
    
    results['decision_curve'] = {
        'thresholds': thresholds,
        'net_benefits': net_benefits,
        'treat_all_nb': treat_all_nb,
        'treat_none_nb': treat_none_nb,
        'prevalence': prevalence
    }
    
    # Classification Metrics at Threshold
    print(f"Computing classification metrics at threshold {optimal_threshold}...")
    
    test_preds['y_pred_thresh'] = (test_preds['y_pred_proba'] >= optimal_threshold).astype(int)
    cm = confusion_matrix(test_preds['y_true'], test_preds['y_pred_thresh'])
    tn, fp, fn, tp = cm.ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    results['classification_metrics'] = {
        'threshold': optimal_threshold,
        'threshold_method': threshold_method,
        'youden_score': youden_score,
        'confusion_matrix': cm,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'ppv': ppv,
        'npv': npv,
        'accuracy': accuracy,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
    }
    
    # Coefficient Analysis
    print("Analyzing model coefficients...")
    
    coef_summary = (df_coefficients
                   .groupby('feature_name')
                   .agg({'coefficient': ['mean', 'std', 'median', 'min', 'max', 'count']})
                   .round(4))
    
    coef_summary.columns = ['mean_coef', 'std_coef', 'median_coef', 'min_coef', 'max_coef', 'n_models']
    coef_summary = coef_summary.sort_values('mean_coef', key=abs, ascending=False)
    
    results['coefficients'] = {
        'summary': coef_summary,
        'raw_data': df_coefficients
    }
    
    # Feature Selection Analysis
    print("Analyzing feature selection patterns...")
    
    feature_freq = (df_feature_selection
                   .groupby('feature_name')
                   .agg({
                       'selection_frequency': ['mean', 'std'],
                       'selected': ['sum', 'count']
                   })
                   .round(4))
    
    feature_freq.columns = ['mean_selection_freq', 'std_selection_freq', 'times_selected', 'total_folds']
    feature_freq['selection_rate'] = feature_freq['times_selected'] / feature_freq['total_folds']
    feature_freq = feature_freq.sort_values('mean_selection_freq', ascending=False)
    
    results['feature_selection'] = {
        'summary': feature_freq,
        'raw_data': df_feature_selection
    }
    
    # Fold-level Analysis
    valid_folds = df_fold_summaries[df_fold_summaries['training_completed'] == True]
    
    results['fold_analysis'] = {
        'summary_stats': {
            'n_total_folds': len(df_fold_summaries),
            'n_valid_folds': len(valid_folds),
            'mean_features_selected': valid_folds['n_features_selected'].mean() if len(valid_folds) > 0 else 0,
            'std_features_selected': valid_folds['n_features_selected'].std() if len(valid_folds) > 0 else 0
        },
        'fold_performance': valid_folds[['cv_id', 'n_features_selected']] if len(valid_folds) > 0 else pd.DataFrame()
    }
    
    return results