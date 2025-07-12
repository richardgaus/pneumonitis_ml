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
    fname = CONFIG / f"predictors_{which}.yml"
    with open(fname) as f:
        data = yaml.safe_load(f)
    return data["endpoint"]