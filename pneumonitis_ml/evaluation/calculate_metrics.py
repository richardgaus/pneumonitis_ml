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

def calculate_comprehensive_metrics(df_predictions,
                                    df_coefficients,
                                    df_feature_selection,
                                    df_fold_summaries,
                                    optimal_threshold=None,
                                    n_bins_calib=4,
                                    n_bootstrap=1000,
                                    ci_level=0.95):
    """
    Calculate discrimination, calibration, threshold-based and feature-related
    metrics from nested-CV output, including 95 % bootstrap CIs.
    """
    results = {}
    # ──────────────────────────────────────────────────────────────────────────
    # 1. Aggregate to patient level (test folds only)
    # ──────────────────────────────────────────────────────────────────────────
    test_preds = (df_predictions[df_predictions['dataset'] == 'test']
                  .groupby('patient_id')
                  .agg(y_true=('y_true', 'max'),
                       y_pred_proba=('y_pred_proba', 'mean'))
                  .reset_index())

    if test_preds.empty:
        print("No test predictions found!")
        return results

    print(f"\nCalculating metrics from {len(test_preds)} test patients")
    results['test_predictions'] = test_preds

    # ──────────────────────────────────────────────────────────────────────────
    # 2. Discrimination metrics + CIs
    # ──────────────────────────────────────────────────────────────────────────
    print("Computing AUROC and AUPRC with bootstrap CIs...")

    auroc_boot, auroc_lo, auroc_hi = bootstrap_ci(
        test_preds, lambda d: roc_auc_score(d['y_true'], d['y_pred_proba']),
        n_bootstrap, ci_level)
    auprc_boot, auprc_lo, auprc_hi = bootstrap_ci(
        test_preds, lambda d: average_precision_score(d['y_true'], d['y_pred_proba']),
        n_bootstrap, ci_level)

    results['performance_metrics'] = {
        'auroc': {'mean': auroc_boot.mean(), 'ci_lower': auroc_lo, 'ci_upper': auroc_hi},
        'auprc': {'mean': auprc_boot.mean(), 'ci_lower': auprc_lo, 'ci_upper': auprc_hi}
    }
    print(f"  AUROC  = {auroc_boot.mean():.3f} ({auroc_lo:.3f}-{auroc_hi:.3f})")
    print(f"  AUPRC  = {auprc_boot.mean():.3f} ({auprc_lo:.3f}-{auprc_hi:.3f})")

    # ──────────────────────────────────────────────────────────────────────────
    # 3. Choose / compute threshold (Youden by default)
    # ──────────────────────────────────────────────────────────────────────────
    if optimal_threshold is None:
        optimal_threshold, youden_score = find_optimal_threshold(
            test_preds['y_true'], test_preds['y_pred_proba'])
        threshold_method = 'youden'
    else:
        # recompute Youden score for the given threshold
        thr_preds = (test_preds['y_pred_proba'] >= optimal_threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(test_preds['y_true'], thr_preds).ravel()
        youden_score = tp / (tp + fn) + tn / (tn + fp) - 1
        threshold_method = 'manual'

    print(f"Optimal threshold: {optimal_threshold:.3f} ({threshold_method})")

    # helper to compute threshold-based stats inside a bootstrap draw
    def thr_stats(data, thr):
        preds_bin = (data['y_pred_proba'] >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(data['y_true'], preds_bin).ravel()
        sens = tp / (tp + fn) if (tp + fn) else 0
        spec = tn / (tn + fp) if (tn + fp) else 0
        ppv  = tp / (tp + fp) if (tp + fp) else 0
        npv  = tn / (tn + fn) if (tn + fn) else 0
        return sens, spec, ppv, npv

    # Bootstrap CIs for Se/Sp/PPV/NPV (threshold fixed)
    sns_boot, sns_lo, sns_hi = bootstrap_ci(
        test_preds, lambda d: thr_stats(d, optimal_threshold)[0], n_bootstrap, ci_level)
    spc_boot, spc_lo, spc_hi = bootstrap_ci(
        test_preds, lambda d: thr_stats(d, optimal_threshold)[1], n_bootstrap, ci_level)
    ppv_boot, ppv_lo, ppv_hi = bootstrap_ci(
        test_preds, lambda d: thr_stats(d, optimal_threshold)[2], n_bootstrap, ci_level)
    npv_boot, npv_lo, npv_hi = bootstrap_ci(
        test_preds, lambda d: thr_stats(d, optimal_threshold)[3], n_bootstrap, ci_level)

    # ──────────────────────────────────────────────────────────────────────────
    # 4. Calibration metrics + CIs
    # ──────────────────────────────────────────────────────────────────────────
    print("Computing calibration metrics…")

    # 4-bin quantile reliability diagram
    frac_pos, mean_pred = calibration_curve(test_preds['y_true'],
                                            test_preds['y_pred_proba'],
                                            n_bins=n_bins_calib,
                                            strategy="quantile")

    # slope / intercept via logistic recalibration
    from sklearn.linear_model import LogisticRegression
    log_odds = np.log(test_preds['y_pred_proba'] /
                      (1 - test_preds['y_pred_proba'] + 1e-15))
    calib_clf = LogisticRegression(fit_intercept=True).fit(
        log_odds.values.reshape(-1, 1), test_preds['y_true'])
    slope = calib_clf.coef_[0][0]
    intercept = calib_clf.intercept_[0]
    brier = brier_score_loss(test_preds['y_true'], test_preds['y_pred_proba'])

    # bootstrap CIs
    brier_boot, brier_lo, brier_hi = bootstrap_ci(
        test_preds, lambda d: brier_score_loss(d['y_true'], d['y_pred_proba']),
        n_bootstrap, ci_level)

    def slope_func(data):
        lo = np.log(data['y_pred_proba'] /
                    (1 - data['y_pred_proba'] + 1e-15))
        m = LogisticRegression(fit_intercept=True).fit(
            lo.values.reshape(-1, 1), data['y_true'])
        return m.coef_[0][0]

    def intc_func(data):
        lo = np.log(data['y_pred_proba'] /
                    (1 - data['y_pred_proba'] + 1e-15))
        m = LogisticRegression(fit_intercept=True).fit(
            lo.values.reshape(-1, 1), data['y_true'])
        return m.intercept_[0]

    slope_boot, slope_lo, slope_hi = bootstrap_ci(
        test_preds, slope_func, n_bootstrap, ci_level)
    intc_boot, intc_lo, intc_hi = bootstrap_ci(
        test_preds, intc_func, n_bootstrap, ci_level)

    results['calibration'] = {
        'brier_score': brier,
        'brier_ci_lower': brier_lo,
        'brier_ci_upper': brier_hi,
        'calibration_intercept': intercept,
        'intercept_ci_lower': intc_lo,
        'intercept_ci_upper': intc_hi,
        'calibration_slope': slope,
        'slope_ci_lower': slope_lo,
        'slope_ci_upper': slope_hi,
        'calibration_curve': {
            'fraction_of_positives': frac_pos,
            'mean_predicted_value': mean_pred
        }
    }

    # ──────────────────────────────────────────────────────────────────────────
    # 5. Decision-curve & misc. (unchanged)
    # ──────────────────────────────────────────────────────────────────────────
    print("Computing decision curve analysis…")

    def net_benefit(y_true, y_prob, thr):
        y_bin = (y_prob >= thr).astype(int)
        tp = ((y_true == 1) & (y_bin == 1)).sum()
        fp = ((y_true == 0) & (y_bin == 1)).sum()
        n = len(y_true)
        return (tp / n) - (fp / n) * (thr / (1 - thr))

    thresholds = np.linspace(0.01, 0.99, 99)
    net_benefits = [net_benefit(test_preds['y_true'],
                                test_preds['y_pred_proba'], t)
                    for t in thresholds]
    prevalence = test_preds['y_true'].mean()
    treat_all_nb = [prevalence - (1 - prevalence) * (t / (1 - t))
                    for t in thresholds]

    results['decision_curve'] = {
        'thresholds': thresholds,
        'net_benefits': net_benefits,
        'treat_all_nb': treat_all_nb,
        'treat_none_nb': [0] * len(thresholds),
        'prevalence': prevalence
    }

    # ──────────────────────────────────────────────────────────────────────────
    # 6. Classification metrics summary (point estimates + CIs)
    # ──────────────────────────────────────────────────────────────────────────
    tn, fp, fn, tp = confusion_matrix(
        test_preds['y_true'],
        (test_preds['y_pred_proba'] >= optimal_threshold).astype(int)
    ).ravel()

    results['classification_metrics'] = {
        'threshold': optimal_threshold,
        'threshold_method': threshold_method,
        'youden_score': youden_score,
        'sensitivity': {'mean': sns_boot.mean(), 'ci_lower': sns_lo, 'ci_upper': sns_hi},
        'specificity': {'mean': spc_boot.mean(), 'ci_lower': spc_lo, 'ci_upper': spc_hi},
        'ppv':         {'mean': ppv_boot.mean(), 'ci_lower': ppv_lo, 'ci_upper': ppv_hi},
        'npv':         {'mean': npv_boot.mean(), 'ci_lower': npv_lo, 'ci_upper': npv_hi},
        'confusion_matrix': (tn, fp, fn, tp)
    }

    # ──────────────────────────────────────────────────────────────────────────
    # 7. Coefficient & feature-selection summaries (unchanged)
    # ──────────────────────────────────────────────────────────────────────────
    coef_summary = (df_coefficients.groupby('feature_name')
                    .agg({'coefficient': ['mean', 'std', 'median', 'min', 'max', 'count']})
                    .round(4))
    coef_summary.columns = ['mean_coef', 'std_coef', 'median_coef',
                            'min_coef', 'max_coef', 'n_models']
    coef_summary = coef_summary.reindex(coef_summary['mean_coef']
                                        .abs().sort_values(ascending=False).index)

    results['coefficients'] = {
        'summary': coef_summary,
        'raw_data': df_coefficients
    }

    feature_freq = (df_feature_selection.groupby('feature_name')
                    .agg({'selection_frequency': ['mean', 'std'],
                          'selected': ['sum', 'count']})
                    .round(4))
    feature_freq.columns = ['mean_selection_freq', 'std_selection_freq',
                            'times_selected', 'total_folds']
    feature_freq['selection_rate'] = (feature_freq['times_selected'] /
                                      feature_freq['total_folds'])
    feature_freq = feature_freq.sort_values('mean_selection_freq', ascending=False)

    results['feature_selection'] = {
        'summary': feature_freq,
        'raw_data': df_feature_selection
    }

    valid_folds = df_fold_summaries[df_fold_summaries['training_completed']]
    results['fold_analysis'] = {
        'summary_stats': {
            'n_total_folds': len(df_fold_summaries),
            'n_valid_folds': len(valid_folds),
            'mean_features_selected': valid_folds['n_features_selected'].mean()
            if not valid_folds.empty else 0,
            'std_features_selected': valid_folds['n_features_selected'].std()
            if not valid_folds.empty else 0
        },
        'fold_performance': valid_folds[['cv_id', 'n_features_selected']]
    }

    return results
