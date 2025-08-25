from pathlib import Path
import yaml

REPO_ROOT       = Path(__file__).resolve().parents[1]
DATA_RAW        = REPO_ROOT / "data" / "raw"
DATA_SIMULATION   = REPO_ROOT / "data" / "simulation"
DATA_PROCESSED  = REPO_ROOT / "data" / "processed"

FILE_ANALYSIS_1 = "SBRT_Lung_pseudonymisiert_.xlsx"
FILE_ANALYSIS_2 = "SBRT_CIGI_PTV_pseudonymisiert.xlsx"

CONFIG = REPO_ROOT / "config"

def load_predictor_set(which="analysis_1", return_type="col_names"):
    """
    Load predictor configuration from YAML file.
    
    Parameters:
    -----------
    which : str, default "analysis_1"
        Which analysis configuration to load (e.g., "analysis_1", "analysis_2")
    return_type : str, default "col_names"
        What to return:
        - "col_names": List of column names for dataframe indexing
        - "metadata": Full predictor metadata with types
        - "keys": List of predictor keys
    
    Returns:
    --------
    list or dict
        Depending on return_type:
        - col_names: ["Age_SBRT", "Sex (0 = F, 1 = M)", ...]
        - metadata: Full predictor configuration with types
        - keys: ["age_sbrt", "sex", "smoker", ...]
    """
    fname = CONFIG / f"predictors_{which}.yml"
    with open(fname) as f:
        data = yaml.safe_load(f)
    
    predictors = data["candidate_predictors"]
    
    if return_type == "col_names":
        # Extract column names for dataframe indexing (most common use case)
        col_names = []
        for predictor_dict in predictors:
            for key, metadata in predictor_dict.items():
                col_names.append(metadata["col_name"])
        return col_names
    
    elif return_type == "metadata":
        # Return full metadata structure
        return predictors
    
    elif return_type == "keys":
        # Extract predictor keys
        keys = []
        for predictor_dict in predictors:
            for key in predictor_dict.keys():
                keys.append(key)
        return keys
    
    else:
        raise ValueError(f"Invalid return_type: {return_type}. "
                        f"Must be 'col_names', 'metadata', or 'keys'")

def load_endpoint(which="analysis_1"):
    """
    Load endpoint configuration from YAML file.
    
    Parameters:
    -----------
    which : str, default "analysis_1"
        Which analysis configuration to load (e.g., "analysis_1", "analysis_2")
    
    Returns:
    --------
    str : Endpoint column name
    """
    fname = CONFIG / f"predictors_{which}.yml"
    with open(fname) as f:
        data = yaml.safe_load(f)
    return data["endpoint"]

def load_hyperparameter_grid(model_type):
    """
    Load hyperparameter grid for different models from YAML configuration.
    
    Parameters:
    -----------
    model_type : str
        Type of model (e.g., 'logistic-regression', 'random-forest')
    
    Returns:
    --------
    dict : Parameter grid for hyperparameter optimization
        Keys are parameter names, values are lists of values to try
        
    Examples:
    --------
    >>> grid = load_hyperparameter_grid('logistic-regression')
    >>> print(grid)
    {'C': [0.001, 0.01, 0.1, 1.0, 10.0]}
    
    >>> grid = load_hyperparameter_grid('random-forest')
    >>> print(grid)
    {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 10], 'min_samples_split': [2, 5, 10]}
    """
    # Load hyperparameter configuration
    fname = CONFIG / "hyperparameter_grids.yml"
    
    try:
        with open(fname) as f:
            config_data = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Hyperparameter grid configuration not found: {fname}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing hyperparameter grid configuration: {e}")
    
    # Extract models configuration
    models_config = config_data.get('models', {})
    
    if model_type not in models_config:
        available_models = list(models_config.keys())
        raise ValueError(f"Unknown model type: {model_type}. Available models: {available_models}")
    
    # Extract parameter grid for the specified model
    model_config = models_config[model_type]
    parameters = model_config.get('parameters', {})
    
    # Build parameter grid
    param_grid = {}
    for param_name, param_config in parameters.items():
        param_grid[param_name] = param_config['values']
    
    if not param_grid:
        raise ValueError(f"No parameters defined for model type: {model_type}")
    
    return param_grid