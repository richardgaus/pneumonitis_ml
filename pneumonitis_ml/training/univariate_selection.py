import pandas as pd
from scipy.stats import mannwhitneyu, fisher_exact


def perform_univariate_selection(X_train_df, y_train, predictors_metadata, max_features=4, p_threshold=0.20):
    """
    Perform univariate feature selection using Mann-Whitney U and Fisher's exact tests.
    
    Parameters:
    -----------
    X_train_df : pd.DataFrame
        Training features (preprocessed)
    y_train : pd.Series
        Training target
    predictors_metadata : list
        Predictor metadata from config
    max_features : int
        Maximum number of features to select
    p_threshold : float
        P-value threshold for feature selection
        
    Returns:
    --------
    tuple : (selected_features, selection_results)
        - selected_features: List of selected feature names
        - selection_results: Dict with test results for all features
    """
    
    # Create mapping from column names to types
    col_to_type = {}
    for predictor_dict in predictors_metadata:
        for key, metadata in predictor_dict.items():
            col_to_type[metadata['col_name']] = metadata['type']
    
    # Map processed feature names back to original names and types
    feature_types = {}
    for processed_name in X_train_df.columns:
        # Handle sklearn ColumnTransformer naming: "num__feature" or "bin__feature"
        if processed_name.startswith('num__'):
            original_name = processed_name[5:]  # Remove 'num__'
            if original_name in col_to_type:
                feature_types[processed_name] = col_to_type[original_name]
        elif processed_name.startswith('bin__'):
            original_name = processed_name[5:]  # Remove 'bin__'
            if original_name in col_to_type:
                feature_types[processed_name] = col_to_type[original_name]
        else:
            # Direct match
            if processed_name in col_to_type:
                feature_types[processed_name] = col_to_type[processed_name]
    
    # If no matches found, try to infer types
    if not feature_types:
        print("    Warning: Could not match processed feature names to config. Inferring types...")
        for col in X_train_df.columns:
            unique_vals = X_train_df[col].dropna().unique()
            if len(unique_vals) <= 2 and all(val in [0, 1, True, False] for val in unique_vals):
                feature_types[col] = 'binary'
            else:
                feature_types[col] = 'continuous'
    
    results = []
    
    for feature in X_train_df.columns:
        try:
            feature_data = X_train_df[feature].dropna()
            target_data = y_train.loc[feature_data.index]
            
            # Skip if no variance or insufficient data
            if len(feature_data.unique()) <= 1 or len(feature_data) < 3:
                continue
                
            feature_type = feature_types.get(feature, 'continuous')
            
            if feature_type == 'binary':
                # Fisher's exact test for binary features
                # Convert to 0/1 if True/False
                if feature_data.dtype == bool:
                    feature_data = feature_data.astype(int)
                
                # Create contingency table
                contingency = pd.crosstab(feature_data, target_data)
                
                # Skip if not 2x2 table or any cell is 0
                if contingency.shape != (2, 2) or (contingency == 0).any().any():
                    continue
                    
                # Convert to numpy array for scipy
                table = contingency.values
                _, p_value = fisher_exact(table, alternative='two-sided')
                test_type = 'fisher_exact'
                
            else:
                # Mann-Whitney U test for continuous features
                group_0 = feature_data[target_data == 0]
                group_1 = feature_data[target_data == 1]
                
                # Skip if either group is empty
                if len(group_0) == 0 or len(group_1) == 0:
                    continue
                    
                _, p_value = mannwhitneyu(group_0, group_1, alternative='two-sided')
                test_type = 'mann_whitney'
            
            results.append({
                'feature': feature,
                'p_value': p_value,
                'test_type': test_type,
                'feature_type': feature_type
            })
            
        except Exception as e:
            print(f"    Warning: Failed to test feature {feature}: {e}")
            continue
    
    # Sort by p-value
    results.sort(key=lambda x: x['p_value'])
    
    # Select features: p < threshold and top max_features
    selected_features = []
    for result in results:
        if result['p_value'] < p_threshold and len(selected_features) < max_features:
            selected_features.append(result['feature'])
    
    selection_summary = {
        'all_results': results,
        'selected_features': selected_features,
        'n_tested': len(results),
        'n_selected': len(selected_features),
        'max_features': max_features,
        'p_threshold': p_threshold
    }
    
    return selected_features, selection_summary