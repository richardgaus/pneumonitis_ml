import pandas as pd
import numpy as np
import json
from datetime import datetime
from pathlib import Path

class CVDataCollector:
    """
    Handles collection and storage of all cross-validation training data.
    """
    
    def __init__(self, experiment_name=None, save_directory=None):
        """
        Initialize the data collector.
        
        Parameters:
        -----------
        experiment_name : str, optional
            Name for this experiment. If None, uses timestamp
        save_directory : str
            Directory to save results
        """
        if save_directory:
            self.save_directory = Path(save_directory)
            self.save_directory.mkdir(parents=True, exist_ok=True)
        
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = experiment_name or f"cv_experiment_{self.timestamp}"
        
        # Initialize storage lists
        self.raw_predictions = []
        self.model_coefficients = []
        self.feature_selection_results = []
        self.fold_summaries = []
        
        # Metadata
        self.metadata = {
            'experiment_info': {
                'name': self.experiment_name,
                'timestamp': self.timestamp,
                'status': 'initialized'
            }
        }
        
        print(f"CVDataCollector initialized: {self.experiment_name}")
    
    def set_experiment_metadata(self, **kwargs):
        """
        Set experiment-level metadata.
        
        Parameters:
        -----------
        **kwargs : dict
            Metadata key-value pairs
        """
        # Convert numpy types to Python native types for JSON serialization
        def convert_for_json(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (list, tuple)):
                return [convert_for_json(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: convert_for_json(value) for key, value in obj.items()}
            else:
                return obj
        
        # Convert all kwargs to JSON-serializable types
        json_kwargs = {key: convert_for_json(value) for key, value in kwargs.items()}
        
        self.metadata['experiment_info'].update(json_kwargs)
    
    def record_fold_start(self, cv_id, repeat_id, fold_id, train_data, test_data, 
                         groups_train, groups_test, feature_names):
        """
        Record the start of a fold with basic information.
        
        Parameters:
        -----------
        cv_id : str
            Unique identifier for this CV fold
        repeat_id : int
            Repeat number
        fold_id : int  
            Fold number within repeat
        train_data : tuple
            (X_train, y_train) training data
        test_data : tuple
            (X_test, y_test) test data
        groups_train : pd.Series
            Training patient groups
        groups_test : pd.Series
            Test patient groups
        feature_names : list
            List of feature names
        """
        X_train, y_train = train_data
        X_test, y_test = test_data
        
        train_patients = set(groups_train.unique())
        test_patients = set(groups_test.unique())
        
        # Verify no patient overlap
        assert len(train_patients.intersection(test_patients)) == 0, "Patient overlap detected!"
        
        fold_info = {
            'cv_id': cv_id,
            'repeat': repeat_id,
            'fold': fold_id,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'train_patients': len(train_patients),
            'test_patients': len(test_patients),
            'train_events': int(y_train.sum()),  # Convert to int for JSON
            'test_events': int(y_test.sum()),    # Convert to int for JSON
            'train_prevalence': float(y_train.mean()),  # Convert to float for JSON
            'test_prevalence': float(y_test.mean()),    # Convert to float for JSON
            'n_features_total': len(feature_names),
            'training_completed': False
        }
        
        # Store temporarily until fold completion
        self._current_fold_info = fold_info
        
        return fold_info
    
    def record_feature_selection(self, cv_id, repeat_id, fold_id, feature_names, 
                                selection_frequencies, selected_features):
        """
        Record feature selection results.
        
        Parameters:
        -----------
        cv_id : str
            CV fold identifier
        repeat_id : int
            Repeat number
        fold_id : int
            Fold number
        feature_names : list
            All feature names
        selection_frequencies : array-like
            Selection frequency for each feature
        selected_features : list
            Names of selected features
        """
        for i, feature in enumerate(feature_names):
            self.feature_selection_results.append({
                'cv_id': cv_id,
                'repeat': repeat_id,
                'fold': fold_id,
                'feature_name': feature,
                'selection_frequency': float(selection_frequencies[i]),  # Convert for JSON
                'selected': feature in selected_features,
                'feature_index': i
            })
    
    def record_hyperparameter_results(self, cv_id, best_lambda, best_score, all_scores):
        """
        Record hyperparameter optimization results.
        
        Parameters:
        -----------
        cv_id : str
            CV fold identifier
        best_lambda : float
            Best lambda value found
        best_score : float
            Best inner CV score
        all_scores : dict
            All lambda values and their scores
        """
        # Convert all_scores to JSON-serializable format
        json_scores = {}
        for key, value in all_scores.items():
            if isinstance(key, (int, float)):
                json_key = str(key)
            else:
                json_key = key
            
            if isinstance(value, (list, tuple)):
                json_scores[json_key] = [float(x) for x in value]
            else:
                json_scores[json_key] = float(value)
        
        self._current_fold_info.update({
            'best_lambda': float(best_lambda),
            'best_inner_score': float(best_score),
            'lambda_scores': json_scores
        })
    
    def record_model_coefficients(self, cv_id, repeat_id, fold_id, model, 
                                 selected_features, numeric_cols, binary_cols):
        """
        Record trained model coefficients.
        
        Parameters:
        -----------
        cv_id : str
            CV fold identifier
        repeat_id : int
            Repeat number
        fold_id : int
            Fold number
        model : sklearn model
            Trained logistic regression model
        selected_features : list
            Names of features used in model
        numeric_cols : list
            Names of numeric columns (for feature type annotation)
        binary_cols : list
            Names of binary columns (for feature type annotation)
        """
        # Store intercept
        self.model_coefficients.append({
            'cv_id': cv_id,
            'repeat': repeat_id,
            'fold': fold_id,
            'feature_name': 'intercept',
            'coefficient': float(model.intercept_[0]),  # Convert for JSON
            'feature_type': 'intercept'
        })
        
        # Store feature coefficients
        coefficients = model.coef_[0]
        for feature, coef in zip(selected_features, coefficients):
            feature_type = 'continuous' if feature in numeric_cols else 'binary'
            
            self.model_coefficients.append({
                'cv_id': cv_id,
                'repeat': repeat_id,
                'fold': fold_id,
                'feature_name': feature,
                'coefficient': float(coef),  # Convert for JSON
                'feature_type': feature_type
            })
    
    def record_predictions(self, cv_id, repeat_id, fold_id, train_data, test_data, 
                          train_predictions, test_predictions, groups):
        """
        Record model predictions for both training and test sets.
        
        Parameters:
        -----------
        cv_id : str
            CV fold identifier
        repeat_id : int
            Repeat number
        fold_id : int
            Fold number
        train_data : tuple
            (X_train, y_train, train_indices)
        test_data : tuple
            (X_test, y_test, test_indices)
        train_predictions : tuple
            (y_pred_proba_train, y_pred_binary_train)
        test_predictions : tuple
            (y_pred_proba_test, y_pred_binary_test)
        groups : pd.Series
            Patient group information (full series)
        """
        X_train, y_train, train_idx = train_data
        X_test, y_test, test_idx = test_data
        y_pred_proba_train, y_pred_binary_train = train_predictions
        y_pred_proba_test, y_pred_binary_test = test_predictions
        
        # Store training predictions
        for idx, (orig_idx, true_label, pred_proba, pred_binary) in enumerate(zip(
            train_idx, y_train, y_pred_proba_train, y_pred_binary_train)):
            self.raw_predictions.append({
                'cv_id': cv_id,
                'repeat': repeat_id,
                'fold': fold_id,
                'dataset': 'train',
                'original_index': int(orig_idx),
                'patient_id': str(groups.iloc[int(orig_idx)]),  # Use iloc with int conversion
                'y_true': int(true_label),
                'y_pred_proba': float(pred_proba),
                'y_pred_binary': int(pred_binary),
                'sample_index_in_fold': idx
            })
        
        # Store test predictions
        for idx, (orig_idx, true_label, pred_proba, pred_binary) in enumerate(zip(
            test_idx, y_test, y_pred_proba_test, y_pred_binary_test)):
            self.raw_predictions.append({
                'cv_id': cv_id,
                'repeat': repeat_id,
                'fold': fold_id,
                'dataset': 'test',
                'original_index': int(orig_idx),
                'patient_id': str(groups.iloc[int(orig_idx)]),  # Use iloc with int conversion
                'y_true': int(true_label),
                'y_pred_proba': float(pred_proba),
                'y_pred_binary': int(pred_binary),
                'sample_index_in_fold': idx
            })
    
    def complete_fold(self, cv_id, selected_features):
        """
        Mark a fold as completed and store final summary.
        
        Parameters:
        -----------
        cv_id : str
            CV fold identifier
        selected_features : list
            Final list of selected features
        """
        self._current_fold_info.update({
            'n_features_selected': len(selected_features),
            'selected_features': list(selected_features),
            'training_completed': True
        })
        
        self.fold_summaries.append(self._current_fold_info.copy())
        self._current_fold_info = None
    
    def fail_fold(self, cv_id, reason="No features selected"):
        """
        Mark a fold as failed and store summary.
        
        Parameters:
        -----------
        cv_id : str
            CV fold identifier
        reason : str
            Reason for failure
        """
        self._current_fold_info.update({
            'n_features_selected': 0,
            'selected_features': [],
            'training_completed': False,
            'failure_reason': reason,
            'best_lambda': None,
            'best_inner_score': None,
            'lambda_scores': {}
        })
        
        self.fold_summaries.append(self._current_fold_info.copy())
        self._current_fold_info = None
    
    def get_dataframes(self):
        """
        Convert collected data to pandas DataFrames.
        
        Returns:
        --------
        dict : Dictionary of DataFrames
        """
        dfs = {
            'predictions': pd.DataFrame(self.raw_predictions),
            'coefficients': pd.DataFrame(self.model_coefficients),
            'feature_selection': pd.DataFrame(self.feature_selection_results),
            'fold_summaries': pd.DataFrame(self.fold_summaries)
        }
        
        return dfs
    
    def print_summary(self):
        """Print a summary of collected data."""
        dfs = self.get_dataframes()
        
        print(f"\n{'='*60}")
        print(f"CV DATA COLLECTION SUMMARY: {self.experiment_name}")
        print(f"{'='*60}")
        print(f"Timestamp: {self.timestamp}")
        print(f"DataFrames created:")
        for name, df in dfs.items():
            print(f"- {name}: {df.shape}")
        
        # Summary statistics
        if len(dfs['fold_summaries']) > 0:
            completed = dfs['fold_summaries']['training_completed'].sum()
            total = len(dfs['fold_summaries'])
            print(f"\nTraining Summary:")
            print(f"- Completed folds: {completed}/{total}")
            
            if len(dfs['predictions']) > 0:
                test_preds = dfs['predictions'][dfs['predictions']['dataset'] == 'test']
                print(f"- Total predictions: {len(dfs['predictions'])}")
                print(f"- Test predictions: {len(test_preds)}")
                if len(test_preds) > 0:
                    print(f"- Unique test patients: {test_preds['patient_id'].nunique()}")
            
            completed_folds = dfs['fold_summaries'][dfs['fold_summaries']['training_completed']]
            if len(completed_folds) > 0:
                print(f"- Mean features selected: {completed_folds['n_features_selected'].mean():.1f}")
                print(f"- Feature selection range: {completed_folds['n_features_selected'].min()}-{completed_folds['n_features_selected'].max()}")
    
    def save_data(self):
        """
        Save all collected data to files in a directory structure.
        
        Returns:
        --------
        str : Base filename used for saving (path to the directory + base name)
        """
        # Create base filename with experiment name and timestamp
        base_filename = f"{self.experiment_name}_{self.timestamp}"
        
        # Create directory for this experiment
        experiment_dir = self.save_directory / base_filename
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Update metadata
        self.metadata['experiment_info']['status'] = 'completed'
        self.metadata['experiment_info']['save_timestamp'] = datetime.now().isoformat()
        self.metadata['experiment_info']['save_directory'] = str(experiment_dir)
        
        # Get DataFrames
        dfs = self.get_dataframes()
        
        print(f"\nSaving data to directory: {experiment_dir}")
        print(f"Base filename: {base_filename}")
        
        saved_files = []
        
        # Save DataFrames as CSV
        for name, df in dfs.items():
            csv_file = experiment_dir / f"{base_filename}_{name}.csv"
            df.to_csv(csv_file, index=False)
            saved_files.append(csv_file)
        
        # Save metadata as JSON
        metadata_file = experiment_dir / f"{base_filename}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2, sort_keys=True)
        saved_files.append(metadata_file)
        
        print(f"Files saved:")
        for file in saved_files:
            print(f"- {file.name}")
        
        # Return the full path to the base filename (directory + base name)
        return str(experiment_dir / base_filename)
    
    def load_data(self, base_filename):
        """
        Load previously saved data from directory structure.
        
        Parameters:
        -----------
        base_filename : str
            Base filename to load from. Can be:
            - Full path to directory + base name
            - Path to directory only
        
        Returns:
        --------
        dict : Dictionary of loaded DataFrames
        """
        base_path = Path(base_filename)
        
        print(f"Loading data from: {base_filename}")
        
        # Determine directory and base filename
        if base_path.is_dir():
            # base_filename is a directory
            directory = base_path
            experiment_base = directory.name
        else:
            # base_filename includes directory and base name
            directory = base_path.parent
            experiment_base = base_path.name
        
        print(f"Looking in directory: {directory}")
        print(f"Using base name: {experiment_base}")
        
        dfs = {}
        for name in ['predictions', 'coefficients', 'feature_selection', 'fold_summaries']:
            csv_file = directory / f"{experiment_base}_{name}.csv"
            
            try:
                df = pd.read_csv(csv_file)
                dfs[name] = df
                print(f"Loaded {name} from {csv_file.name}")
                
            except FileNotFoundError:
                print(f"Warning: Could not load {name} from {csv_file}")
                dfs[name] = pd.DataFrame()
        
        # Load metadata from JSON
        metadata_file = directory / f"{experiment_base}_metadata.json"
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                self.metadata = metadata
                print(f"Loaded metadata from {metadata_file.name}")
        except FileNotFoundError:
            print(f"Warning: Could not load metadata from {metadata_file}")
        
        # Update internal storage
        if len(dfs['predictions']) > 0:
            self.raw_predictions = dfs['predictions'].to_dict('records')
        if len(dfs['coefficients']) > 0:
            self.model_coefficients = dfs['coefficients'].to_dict('records')
        if len(dfs['feature_selection']) > 0:
            self.feature_selection_results = dfs['feature_selection'].to_dict('records')
        if len(dfs['fold_summaries']) > 0:
            self.fold_summaries = dfs['fold_summaries'].to_dict('records')
        
        print(f"Data loaded successfully")
        self.print_summary()
        
        return dfs