#!/usr/bin/env python3
"""
03_run_nested_cv.py

Script to run nested cross-validation for pneumonitis prediction.
Handles stability selection, hyperparameter optimization, and model training
with comprehensive data collection. Supports both Logistic Regression and Random Forest.

Usage:
    python 03_run_nested_cv.py <dataset_file> --analysis {1,2} [options]

Examples:
    python 03_run_nested_cv.py data/processed/analysis_1_single.xlsx --analysis 1
    python 03_run_nested_cv.py data/processed/analysis_1_single.xlsx --analysis 1 --model random-forest
    python 03_run_nested_cv.py data/simulation/synthetic_easy.xlsx --analysis 1 --experiment pneumonitis_synthetic
    python 03_run_nested_cv.py analysis_1_data.xlsx --analysis 1 --n-splits 5 --n-repeats 5
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Import required functions and config
from pneumonitis_ml.config import (
    REPO_ROOT, load_predictor_set, load_endpoint
)
from pneumonitis_ml.evaluation.data_collector import CVDataCollector
from pneumonitis_ml.training.stability_selection import stability_selection
from pneumonitis_ml.training.hyperparameter_optimization import hyperparameter_optimization
from pneumonitis_ml.training.univariate_selection import perform_univariate_selection

warnings.filterwarnings('ignore')

# Set default output directory
DEFAULT_OUTPUT_DIR = REPO_ROOT / 'results' / 'raw_cv_results'


def load_and_validate_data(dataset_file, analysis_number):
    """
    Load dataset and validate it has required columns.
    
    Parameters:
    -----------
    dataset_file : Path
        Path to dataset file
    analysis_number : int
        Analysis number (1 or 2)
        
    Returns:
    --------
    tuple : (df_data, predictors, target_col, numeric_cols, binary_cols)
    """
    
    print(f"Loading dataset: {dataset_file}")
    
    # Load dataset
    if dataset_file.suffix.lower() == '.xlsx':
        df_data = pd.read_excel(dataset_file)
    elif dataset_file.suffix.lower() == '.csv':
        df_data = pd.read_csv(dataset_file)
    else:
        raise ValueError(f"Unsupported file format: {dataset_file.suffix}")
    
    print(f"Dataset shape: {df_data.shape}")
    
    # Load analysis configuration
    try:
        predictors = load_predictor_set(which=f"analysis_{analysis_number}")
        target_col = load_endpoint(which=f"analysis_{analysis_number}")
        predictors_metadata = load_predictor_set(which=f"analysis_{analysis_number}", return_type="metadata")
        
        print(f"Analysis {analysis_number} configuration:")
        print(f"  - Predictors: {len(predictors)} features")
        print(f"  - Target: {target_col}")
        
    except Exception as e:
        print(f"Error loading analysis {analysis_number} configuration: {e}")
        sys.exit(1)
    
    # Extract column types
    numeric_cols = [list(d.values())[0]['col_name'] for d in predictors_metadata 
                    if list(d.values())[0]['type'] == 'continuous']
    
    binary_cols = [list(d.values())[0]['col_name'] for d in predictors_metadata 
                   if list(d.values())[0]['type'] == 'binary']
    
    print(f"  - Numeric features: {len(numeric_cols)}")
    print(f"  - Binary features: {len(binary_cols)}")
    
    # Validate required columns exist
    required_cols = ['patient_id'] + predictors + [target_col]
    missing_cols = [col for col in required_cols if col not in df_data.columns]
    
    if missing_cols:
        print(f"Error: Missing required columns: {missing_cols}")
        print(f"Available columns: {list(df_data.columns)}")
        sys.exit(1)
    
    # Check for missing values in key columns
    if df_data['patient_id'].isna().any():
        print("Error: Missing values in patient_id column")
        sys.exit(1)
    
    if df_data[target_col].isna().any():
        print(f"Warning: Missing values in target column {target_col}")
        print(f"Dropping {df_data[target_col].isna().sum()} rows with missing target")
        df_data = df_data.dropna(subset=[target_col])
    
    # Print dataset summary
    print(f"\nDataset summary:")
    print(f"  - Total samples: {len(df_data)}")
    print(f"  - Unique patients: {df_data['patient_id'].nunique()}")
    print(f"  - Target prevalence: {df_data[target_col].mean():.3f}")
    print(f"  - Target distribution: {df_data[target_col].value_counts().to_dict()}")
    
    return df_data, predictors, target_col, numeric_cols, binary_cols


def get_hyperparameter_config(model_type):
    """
    Get hyperparameter configuration for different models.
    
    Parameters:
    -----------
    model_type : str
        Type of model ('logistic-regression' or 'random-forest')
    
    Returns:
    --------
    dict : Configuration with model_factory and param_grid
    """
    
    # Import config function here to avoid circular imports
    from pneumonitis_ml.config import load_hyperparameter_grid
    
    if model_type == 'logistic-regression':
        model_factory = lambda **params: LogisticRegression(
            penalty='l2',
            solver='lbfgs', 
            max_iter=1000,
            random_state=42,
            **params
        )
        
    elif model_type == 'random-forest':
        model_factory = lambda **params: RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            **params
        )
        
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load parameter grid from config
    param_grid = load_hyperparameter_grid(model_type)
    
    return {
        'model_factory': model_factory,
        'param_grid': param_grid
    }


def main():
    """Main function to run nested cross-validation."""
    
    parser = argparse.ArgumentParser(
        description="Run nested cross-validation for pneumonitis prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with logistic regression (default)
  python 03_run_nested_cv.py data/processed/analysis_1_single.xlsx --analysis 1

  # Run with random forest
  python 03_run_nested_cv.py data/processed/analysis_1_single.xlsx --analysis 1 --model random-forest

  # Run with stability selection
  python 03_run_nested_cv.py data/processed/analysis_1_single.xlsx --analysis 1 --use-stability-selection

  # Run with univariate selection
  python 03_run_nested_cv.py data/processed/analysis_1_single.xlsx --analysis 1 --use-univariate-selection

  # Test run with minimal iterations
  python 03_run_nested_cv.py data.xlsx --analysis 1 --n-splits 2 --n-repeats 2 --experiment quick_test
        """
    )
    
    # Required arguments
    parser.add_argument(
        'dataset_file',
        type=str,
        help='Path to dataset file (Excel or CSV)'
    )
    
    parser.add_argument(
        '--analysis',
        type=int,
        choices=[1, 2],
        required=True,
        help='Analysis number (1 or 2)'
    )
    
    # Model selection
    parser.add_argument(
        '--model',
        type=str,
        choices=['logistic-regression', 'random-forest'],
        default='logistic-regression',
        help='Model type to use (default: logistic-regression)'
    )
    
    # Cross-validation parameters
    parser.add_argument(
        '--n-splits',
        type=int,
        default=3,
        help='Number of CV splits (default: 3)'
    )
    
    parser.add_argument(
        '--n-repeats',
        type=int,
        default=3,
        help='Number of CV repeats (default: 3)'
    )
    
    # Stability selection parameters
    parser.add_argument(
        '--use-stability-selection',
        action='store_true',
        help='Use stability selection for feature selection (default: use all features)'
    )
    
    parser.add_argument(
        '--stability-iterations',
        type=int,
        default=1000,
        help='Number of stability selection iterations (default: 1000, only used with --use-stability-selection)'
    )
    
    parser.add_argument(
        '--stability-q',
        type=float,
        default=0.8,
        help='Stability selection subsampling rate (default: 0.8, only used with --use-stability-selection)'
    )
    
    parser.add_argument(
        '--stability-threshold',
        type=float,
        default=0.7,
        help='Stability selection threshold (default: 0.7, only used with --use-stability-selection)'
    )
    
    # Univariate selection parameters
    parser.add_argument(
        '--use-univariate-selection',
        action='store_true',
        help='Use univariate statistical tests for feature selection (Mann-Whitney U for continuous, Fisher exact for binary)'
    )
    
    parser.add_argument(
        '--univariate-max-features',
        type=int,
        default=4,
        help='Maximum number of features to select with univariate testing (default: 4)'
    )
    
    parser.add_argument(
        '--univariate-p-threshold',
        type=float,
        default=0.20,
        help='P-value threshold for univariate feature selection (default: 0.20)'
    )
    
    # Inner CV parameters
    parser.add_argument(
        '--inner-cv-splits',
        type=int,
        default=3,
        help='Number of inner CV splits for hyperparameter optimization (default: 3)'
    )
    
    parser.add_argument(
        '--inner-cv-repeats',
        type=int,
        default=5,
        help='Number of inner CV repeats for hyperparameter optimization (default: 5)'
    )
    
    # Experiment parameters
    parser.add_argument(
        '--experiment',
        type=str,
        default=None,
        help='Experiment name (default: auto-generated from dataset and analysis)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f'Output directory for results (default: {DEFAULT_OUTPUT_DIR})'
    )
    
    # Control parameters
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show configuration without running CV'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Validate feature selection options
    if args.use_stability_selection and args.use_univariate_selection:
        print("Error: Cannot use both stability selection and univariate selection. Choose one.")
        sys.exit(1)
    
    # Validate and resolve dataset file path
    dataset_file = Path(args.dataset_file)
    if not dataset_file.is_absolute():
        dataset_file = REPO_ROOT / dataset_file
    
    if not dataset_file.exists():
        print(f"Error: Dataset file not found: {dataset_file}")
        sys.exit(1)
    
    # Set output directory
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    
    # Generate experiment name if not provided
    if args.experiment is None:
        dataset_name = dataset_file.stem
        model_suffix = 'lr' if args.model == 'logistic-regression' else 'rf'
        args.experiment = f"{dataset_name}_analysis_{args.analysis}_{model_suffix}"
    
    # Get model configuration
    model_config = get_hyperparameter_config(args.model)
    param_grid = model_config['param_grid']
    
    # Print configuration
    print("="*60)
    print("NESTED CROSS-VALIDATION CONFIGURATION")
    print("="*60)
    print(f"Dataset: {dataset_file}")
    print(f"Analysis: {args.analysis}")
    print(f"Model: {args.model}")
    print(f"Experiment: {args.experiment}")
    print(f"CV: {args.n_repeats} repeats × {args.n_splits} splits")
    print(f"Use stability selection: {args.use_stability_selection}")
    if args.use_stability_selection:
        print(f"Stability selection: {args.stability_iterations} iterations, q={args.stability_q}")
    print(f"Use univariate selection: {args.use_univariate_selection}")
    if args.use_univariate_selection:
        print(f"Univariate selection: max {args.univariate_max_features} features, p < {args.univariate_p_threshold}")
    print(f"Hyperparameters: {param_grid}")
    print(f"Inner CV: {args.inner_cv_repeats} repeats × {args.inner_cv_splits} splits")
    print(f"Output directory: {output_dir}")
    print(f"Random seed: {args.seed}")
    print(f"Dry run: {args.dry_run}")
    print("="*60)
    
    if args.dry_run:
        print("DRY RUN MODE - Configuration validated, no CV will be run")
        return
    
    try:
        # Load and validate data
        df_data, predictors, target_col, numeric_cols, binary_cols = load_and_validate_data(
            dataset_file, args.analysis
        )
        
        # Load predictors metadata for univariate selection
        predictors_metadata = load_predictor_set(which=f"analysis_{args.analysis}", return_type="metadata")
        
        # Initialize data collector
        collector = CVDataCollector(
            experiment_name=args.experiment, 
            save_directory=str(output_dir)
        )

        # Convert Path objects to strings for metadata
        command_line_args = {}
        for key, value in vars(args).items():
            if isinstance(value, Path):
                command_line_args[key] = str(value)
            else:
                command_line_args[key] = value
        
        # Set experiment metadata
        collector.set_experiment_metadata(
            dataset_file=str(dataset_file),
            analysis_number=args.analysis,
            model_type=args.model,
            n_splits=args.n_splits,
            n_repeats=args.n_repeats,
            total_samples=len(df_data),
            total_patients=df_data['patient_id'].nunique(),
            target_prevalence=df_data[target_col].mean(),
            target_column=target_col,
            predictor_columns=predictors,
            numeric_columns=numeric_cols,
            binary_columns=binary_cols,
            group_column='patient_id',
            # Feature selection settings
            use_stability_selection=args.use_stability_selection,
            stability_iterations=args.stability_iterations if args.use_stability_selection else None,
            stability_q=args.stability_q if args.use_stability_selection else None,
            stability_threshold=args.stability_threshold if args.use_stability_selection else None,
            use_univariate_selection=args.use_univariate_selection,
            univariate_max_features=args.univariate_max_features if args.use_univariate_selection else None,
            univariate_p_threshold=args.univariate_p_threshold if args.use_univariate_selection else None,
            # Hyperparameter optimization settings
            hyperparameter_grid=param_grid,
            inner_cv_splits=args.inner_cv_splits,
            inner_cv_repeats=args.inner_cv_repeats,
            # General settings
            random_seed=args.seed,
            # Script version info
            script_name='03_run_nested_cv.py',
            command_line_args=command_line_args  # Save all command line arguments
        )
        
        # Create preprocessing pipeline
        numeric_pipe = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scal", StandardScaler())
        ])

        binary_pipe = Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent"))
        ])
        
        preprocess = ColumnTransformer([
            ("num", numeric_pipe, numeric_cols),
            ("bin", binary_pipe, binary_cols)
        ])
        
        # Prepare data for cross-validation
        X = df_data[predictors]  # Filter to only prespecified predictors
        y = df_data[target_col]
        groups = df_data['patient_id']
        
        print(f"\nStarting {args.n_repeats} x {args.n_splits}-fold nested cross-validation")
        print(f"Total samples: {len(X)}, Total patients: {groups.nunique()}")
        print(f"Target prevalence: {y.mean():.3f}")
        print("-" * 60)
        
        # Nested cross-validation loop
        for repeat in range(args.n_repeats):
            print(f"\nStarting repeat {repeat + 1}/{args.n_repeats}")
            
            # Generate new random seed for each repeat
            sgkf = StratifiedGroupKFold(
                n_splits=args.n_splits, 
                shuffle=True, 
                random_state=args.seed + repeat
            )
            
            # Cross-validation within this repeat
            for fold, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups)):
                print(f"  Processing fold {fold + 1}/{args.n_splits}")
                
                # Create fold identifiers
                repeat_id = repeat + 1
                fold_id = fold + 1
                cv_id = f"R{repeat_id}_F{fold_id}"
                
                # Split data
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                groups_train, groups_test = groups.iloc[train_idx], groups.iloc[test_idx]
                
                # Fit and transform data
                preprocess.fit(X_train)
                X_train_processed = preprocess.transform(X_train)
                X_test_processed = preprocess.transform(X_test)
                
                # Get feature names
                if hasattr(preprocess, 'get_feature_names_out'):
                    feature_names = preprocess.get_feature_names_out()
                else:
                    feature_names = numeric_cols + binary_cols
                
                X_train_df = pd.DataFrame(X_train_processed, columns=feature_names, index=X_train.index)
                X_test_df = pd.DataFrame(X_test_processed, columns=feature_names, index=X_test.index)
                
                # Record fold start
                fold_info = collector.record_fold_start(
                    cv_id=cv_id,
                    repeat_id=repeat_id,
                    fold_id=fold_id,
                    train_data=(X_train, y_train),
                    test_data=(X_test, y_test),
                    groups_train=groups_train,
                    groups_test=groups_test,
                    feature_names=feature_names
                )
                
                print(f"    Train: {fold_info['train_size']} samples, {fold_info['train_patients']} patients")
                print(f"    Test:  {fold_info['test_size']} samples, {fold_info['test_patients']} patients")
                print(f"    Train events: {fold_info['train_events']}/{fold_info['train_size']} ({fold_info['train_prevalence']:.3f})")
                print(f"    Test events:  {fold_info['test_events']}/{fold_info['test_size']} ({fold_info['test_prevalence']:.3f})")
                
                # FEATURE SELECTION
                if args.use_univariate_selection:
                    print(f"    Running univariate feature selection...")
                    try:
                        selected_features, selection_summary = perform_univariate_selection(
                            X_train_df, y_train, predictors_metadata,
                            max_features=args.univariate_max_features,
                            p_threshold=args.univariate_p_threshold
                        )
                        
                        # Create selection frequencies for compatibility (all selected = 1.0, others = 0.0)
                        selection_frequencies = np.zeros(len(feature_names))
                        for i, feature in enumerate(feature_names):
                            if feature in selected_features:
                                selection_frequencies[i] = 1.0
                        
                        print(f"    Tested {selection_summary['n_tested']} features")
                        print(f"    Selected {len(selected_features)} features with p < {args.univariate_p_threshold}")
                        
                    except Exception as e:
                        print(f"    Error in univariate selection: {e}")
                        if args.verbose:
                            import traceback
                            traceback.print_exc()
                        collector.fail_fold(cv_id=cv_id, reason=f"Univariate selection failed: {str(e)}")
                        continue
                        
                elif args.use_stability_selection:
                    print(f"    Running stability selection...")
                    stability_model = LogisticRegression(
                        penalty="l1", 
                        solver="saga", 
                        C=1.0, 
                        warm_start=True, 
                        max_iter=5000
                    )
                    
                    try:
                        stability_results = stability_selection(
                            X_train=X_train_df,
                            y_train=y_train,
                            groups_train=groups_train,
                            feature_names=feature_names,
                            model=stability_model,
                            B=args.stability_iterations,
                            q=args.stability_q,
                            pi_thr=args.stability_threshold,
                            random_state=args.seed + repeat * 100 + fold
                        )
                        
                        selected_features = stability_results['selected_features']
                        selection_frequencies = stability_results['selection_frequencies']
                        
                    except Exception as e:
                        print(f"    Error in stability selection: {e}")
                        if args.verbose:
                            import traceback
                            traceback.print_exc()
                        collector.fail_fold(cv_id=cv_id, reason=f"Stability selection failed: {str(e)}")
                        continue
                else:
                    print(f"    Using all features (no feature selection)")
                    selected_features = list(feature_names)
                    # Create dummy selection frequencies (all features selected with frequency 1.0)
                    selection_frequencies = np.ones(len(feature_names))
                
                # Record feature selection results
                collector.record_feature_selection(
                    cv_id=cv_id,
                    repeat_id=repeat_id,
                    fold_id=fold_id,
                    feature_names=feature_names,
                    selection_frequencies=selection_frequencies,
                    selected_features=selected_features
                )
                
                print(f"    Selected {len(selected_features)} features")
                if args.verbose and len(selected_features) > 0:
                    print(f"    Selected features: {list(selected_features)}")
                
                # Check if any features were selected
                if len(selected_features) == 0:
                    print(f"    No features selected - marking fold as failed")
                    collector.fail_fold(cv_id=cv_id, reason="No features selected")
                    continue
                
                # HYPERPARAMETER OPTIMIZATION
                print(f"    Running hyperparameter optimization...")
                try:
                    hyperopt_results = hyperparameter_optimization(
                        X_train=X_train_df,
                        y_train=y_train,
                        groups_train=groups_train,
                        selected_features=selected_features,
                        model_factory=model_config['model_factory'],
                        param_grid=param_grid,
                        n_splits=args.inner_cv_splits,
                        n_repeats=args.inner_cv_repeats,
                        random_state=args.seed + repeat * 1000 + fold * 10,
                        verbose=args.verbose
                    )
                    
                    best_params = hyperopt_results['best_params']
                    best_inner_score = hyperopt_results['best_score']
                    
                except Exception as e:
                    print(f"    Error in hyperparameter optimization: {e}")
                    if args.verbose:
                        import traceback
                        traceback.print_exc()
                    collector.fail_fold(cv_id=cv_id, reason=f"Hyperparameter optimization failed: {str(e)}")
                    continue
                
                # Record hyperparameter results
                collector.record_hyperparameter_results(
                    cv_id=cv_id,
                    best_score=best_inner_score,
                    all_scores=hyperopt_results['all_scores'],
                    best_params=best_params
                )
                
                print(f"    Best params: {best_params}, Inner CV score: {best_inner_score:.4f}")
                
                # FINAL MODEL TRAINING
                print(f"    Training final model...")
                
                try:
                    # Create final model with best hyperparameters using the same factory
                    final_model = model_config['model_factory'](**best_params)
                    
                    # Use selected features
                    X_train_selected = X_train_df[selected_features]
                    X_test_selected = X_test_df[selected_features]
                    
                    # Fit final model
                    final_model.fit(X_train_selected, y_train)
                    
                    # Record model coefficients (for logistic regression) or feature importances (for random forest)
                    collector.record_model_coefficients(
                        cv_id=cv_id,
                        repeat_id=repeat_id,
                        fold_id=fold_id,
                        model=final_model,
                        selected_features=selected_features,
                        numeric_cols=numeric_cols,
                        binary_cols=binary_cols
                    )
                    
                    # Get predictions
                    y_pred_proba_train = final_model.predict_proba(X_train_selected)[:, 1]
                    y_pred_proba_test = final_model.predict_proba(X_test_selected)[:, 1]
                    y_pred_binary_train = final_model.predict(X_train_selected)
                    y_pred_binary_test = final_model.predict(X_test_selected)
                    
                    # Record predictions (convert numpy indices to regular Python lists)
                    collector.record_predictions(
                        cv_id=cv_id,
                        repeat_id=repeat_id,
                        fold_id=fold_id,
                        train_data=(X_train, y_train, train_idx.tolist()),
                        test_data=(X_test, y_test, test_idx.tolist()),
                        train_predictions=(y_pred_proba_train, y_pred_binary_train),
                        test_predictions=(y_pred_proba_test, y_pred_binary_test),
                        groups=groups
                    )
                    
                    # Mark fold as completed
                    collector.complete_fold(cv_id=cv_id, selected_features=selected_features)
                    
                    print(f"    Fold {fold_id} completed successfully")
                    
                except Exception as e:
                    print(f"    Error in final model training: {e}")
                    if args.verbose:
                        import traceback
                        traceback.print_exc()
                    collector.fail_fold(cv_id=cv_id, reason=f"Final model training failed: {str(e)}")
                    continue
        
        # Print summary and save data
        print(f"\n{'='*60}")
        print("CROSS-VALIDATION COMPLETED")
        print(f"{'='*60}")
        
        collector.print_summary()
        base_filename = collector.save_data()
        
        print(f"\nResults saved with base filename:")
        print(f"{base_filename}")
        print(f"\nTo analyze results, run:")
        print(f"python 04_analyze_results.py {base_filename}")
        
    except KeyboardInterrupt:
        print(f"\nTraining interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error during training: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()