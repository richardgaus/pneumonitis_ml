import numpy as np

def stability_selection(X_train, y_train, groups_train, feature_names, 
                        model=None, B=1000, q=0.8, pi_thr=0.70, 
                        min_samples=10, random_state=42, verbose=True):
    """
    Perform stability selection using patient-level subsampling.
    
    Parameters:
    -----------
    X_train : pandas.DataFrame or numpy.ndarray
        Training feature matrix
    y_train : pandas.Series or numpy.ndarray
        Training target variable
    groups_train : pandas.Series or numpy.ndarray
        Patient IDs for grouping (ensures patient-level subsampling)
    feature_names : list or numpy.ndarray
        Names of features in X_train
    model : sklearn estimator, default None
        Model to use for feature selection. If None, uses LogisticRegression with L1 penalty
    B : int, default 1000
        Number of bootstrap iterations
    q : float, default 0.8
        Subsample ratio (proportion of patients to sample)
    pi_thr : float, default 0.70
        Selection frequency threshold for keeping features
    min_samples : int, default 10
        Minimum number of samples required for a valid subsample
    random_state : int, default 42
        Random seed for reproducibility
    verbose : bool, default True
        Whether to print progress information
    
    Returns:
    --------
    dict
        Dictionary containing:
        - 'selected_features': array of selected feature names
        - 'selection_frequencies': array of selection frequencies for all features
        - 'n_selected': number of selected features
        - 'n_total': total number of features
    """
    
    # Default model if none provided
    if model is None:
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(penalty="l1", solver="saga", C=1.0, 
                                 warm_start=True, max_iter=5000)
    
    # Initialize variables
    n_features = len(feature_names)
    sel_counts = np.zeros(n_features)
    unique_patients = groups_train.unique()
    rng = np.random.RandomState(random_state)
    
    if verbose:
        print(f"    Running stability selection...")
        print(f"      B={B}, q={q}, pi_thr={pi_thr}")
        print(f"      {len(unique_patients)} patients, {n_features} features")
    
    successful_iterations = 0
    
    for b in range(B):
        try:
            # 1. Draw random subset of patients
            n_patients_subsample = int(np.ceil(q * len(unique_patients)))
            pat_sub = rng.choice(unique_patients, size=n_patients_subsample, replace=False)
            
            # 2. Get indices for selected patients
            if hasattr(groups_train, 'isin'):
                idx = groups_train.isin(pat_sub)
            else:
                idx = np.isin(groups_train, pat_sub)
            
            # Skip if subsample is too small
            if idx.sum() < min_samples:
                continue
            
            # 3. Extract subsample
            if hasattr(X_train, 'loc'):
                X_sub = X_train.loc[idx]
                y_sub = y_train.loc[idx]
            else:
                X_sub = X_train[idx]
                y_sub = y_train[idx]
            
            # Skip if target has only one class
            if hasattr(y_sub, 'nunique'):
                if y_sub.nunique() < 2:
                    continue
            else:
                if len(np.unique(y_sub)) < 2:
                    continue
            
            # 4. Fit model and extract feature selection
            model_copy = model.__class__(**model.get_params())
            model_copy.set_params(random_state=random_state + b)  # Set unique random state for each iteration
            model_copy.fit(X_sub, y_sub)
            
            # Extract coefficients (assumes model has coef_ attribute)
            if hasattr(model_copy, 'coef_'):
                coefs = model_copy.coef_.ravel()
            elif hasattr(model_copy, 'feature_importances_'):
                # For tree-based models that don't have coef_
                coefs = model_copy.feature_importances_
            else:
                raise ValueError(f"Model {type(model)} doesn't have coef_ or feature_importances_ attribute")
            
            # Count selected features (non-zero coefficients/importances)
            sel_counts += (coefs != 0)
            successful_iterations += 1
            
        except Exception as e:
            if verbose and b < 10:  # Only print first few errors to avoid spam
                print(f"      Warning: Iteration {b} failed: {e}")
            continue
    
    # 5. Compute selection frequencies and apply threshold
    if successful_iterations == 0:
        raise RuntimeError("No successful stability selection iterations completed")

    freqs = sel_counts / successful_iterations
    keep_ix = freqs >= pi_thr
    selected_features = np.array(feature_names)[keep_ix]
    
    if verbose:
        print(f"    Stability selection completed:")
        print(f"      Successful iterations: {successful_iterations}/{B}")
        print(f"      Features selected: {keep_ix.sum()}/{n_features}")
        if len(selected_features) > 0:
            print(f"      Selected features: {list(selected_features)}")
        else:
            print(f"      No features met the threshold pi_thr={pi_thr}")
    
    return {
        'selected_features': selected_features,
        'selection_frequencies': freqs,
        'n_selected': keep_ix.sum(),
        'n_total': n_features,
        'successful_iterations': successful_iterations
    }