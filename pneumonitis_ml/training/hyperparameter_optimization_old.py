from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import numpy as np

def hyperparameter_optimization(X_train, y_train, groups_train, selected_features, 
                               lambda_values=None, n_splits=3, n_repeats=5, random_state=42, verbose=True):
    """
    Perform hyperparameter optimization using repeated inner cross-validation with patient-level splits.
    
    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training feature matrix (preprocessed)
    y_train : pandas.Series
        Training target variable
    groups_train : pandas.Series
        Patient IDs for ensuring patient-level splits
    selected_features : array-like
        List of selected feature names to use
    lambda_values : list, default None
        List of regularization parameters to test. If None, uses [0.001, 0.01, 0.1, 1, 10]
    n_splits : int, default 3
        Number of inner CV folds
    n_repeats : int, default 5
        Number of times to repeat the inner CV
    random_state : int, default 42
        Random seed for reproducibility
    verbose : bool, default True
        Whether to print progress information
    
    Returns:
    --------
    dict
        Dictionary containing:
        - 'best_lambda': best regularization parameter
        - 'best_score': best mean AUROC score
        - 'all_scores': dict with lambda -> mean_auroc mapping
        - 'cv_details': detailed results for each lambda, repeat, and fold
    """
    
    if lambda_values is None:
        lambda_values = [0.001, 0.01, 0.1, 1, 10]
    
    # Filter to selected features only
    if len(selected_features) == 0:
        raise ValueError("No features selected - cannot perform hyperparameter optimization")
    
    X_train_selected = X_train[selected_features]
    
    if verbose:
        print(f"    Hyperparameter optimization:")
        print(f"      Features: {len(selected_features)}")
        print(f"      Lambda values: {lambda_values}")
        print(f"      Inner CV: {n_repeats}x{n_splits}-fold")
    
    # Initialize results storage
    lambda_scores = {}
    cv_details = {}
    
    for lambda_val in lambda_values:
        all_fold_scores = []
        cv_details[lambda_val] = []
        
        if verbose:
            print(f"      Testing λ = {lambda_val}...")
        
        # Repeated inner cross-validation
        for repeat in range(n_repeats):
            # Inner cross-validation with patient-level splits
            inner_cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, 
                                          random_state=random_state + repeat)
            
            for inner_fold, (inner_train_idx, inner_valid_idx) in enumerate(inner_cv.split(X_train_selected, y_train, groups_train)):
                # Split inner training data
                X_inner_train = X_train_selected.iloc[inner_train_idx]
                X_inner_valid = X_train_selected.iloc[inner_valid_idx]
                y_inner_train = y_train.iloc[inner_train_idx]
                y_inner_valid = y_train.iloc[inner_valid_idx]
                groups_inner_train = groups_train.iloc[inner_train_idx]
                groups_inner_valid = groups_train.iloc[inner_valid_idx]
                
                # Verify patient-level split
                train_patients = set(groups_inner_train.unique())
                valid_patients = set(groups_inner_valid.unique())
                assert len(train_patients.intersection(valid_patients)) == 0, "Patient overlap in inner CV!"
                
                # Check if both classes are present in training set
                if y_inner_train.nunique() < 2:
                    if verbose:
                        print(f"        Repeat {repeat + 1}, Fold {inner_fold + 1}: Skipping - only one class in training")
                    continue
                
                # Fit Ridge Logistic Regression (L2 penalty)
                # Note: C = 1/lambda in sklearn
                safe_lambda = max(lambda_val, 1e-6)
                model = LogisticRegression(
                    penalty='l2',
                    C=1/safe_lambda,
                    solver='lbfgs',
                    max_iter=1000,
                    random_state=random_state
                )
                
                try:
                    model.fit(X_inner_train, y_inner_train)
                    
                    # Predict probabilities for AUROC calculation
                    y_pred_proba = model.predict_proba(X_inner_valid)[:, 1]
                    
                    # Calculate AUROC
                    auroc = roc_auc_score(y_inner_valid, y_pred_proba)
                    all_fold_scores.append(auroc)
                    
                    cv_details[lambda_val].append({
                        'repeat': repeat + 1,
                        'fold': inner_fold + 1,
                        'auroc': auroc,
                        'train_size': len(X_inner_train),
                        'valid_size': len(X_inner_valid),
                        'train_patients': len(train_patients),
                        'valid_patients': len(valid_patients)
                    })
                    
                    if verbose:
                        print(f"        Repeat {repeat + 1}, Fold {inner_fold + 1}: AUROC = {auroc:.4f} "
                              f"({len(train_patients)} train patients, {len(valid_patients)} valid patients)")
                        
                except Exception as e:
                    if verbose:
                        print(f"        Repeat {repeat + 1}, Fold {inner_fold + 1}: Failed - {e}")
                    continue
        
        # Calculate mean AUROC for this lambda across all repeats and folds
        if len(all_fold_scores) > 0:
            mean_auroc = np.mean(all_fold_scores)
            lambda_scores[lambda_val] = mean_auroc
            
            if verbose:
                print(f"        Mean AUROC: {mean_auroc:.4f} ± {np.std(all_fold_scores):.4f} "
                      f"({len(all_fold_scores)} folds)")
        else:
            lambda_scores[lambda_val] = 0.0
            if verbose:
                print(f"        Mean AUROC: No valid folds")
    
    # Find best lambda
    if len(lambda_scores) == 0:
        raise RuntimeError("No valid hyperparameter evaluations completed")
    
    best_lambda = max(lambda_scores.keys(), key=lambda k: lambda_scores[k])
    best_score = lambda_scores[best_lambda]
    
    if verbose:
        print(f"      Best λ = {best_lambda} (AUROC = {best_score:.4f})")
    
    return {
        'best_lambda': best_lambda,
        'best_score': best_score,
        'all_scores': lambda_scores,
        'cv_details': cv_details
    }
