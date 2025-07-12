import pandas as pd
import numpy as np

def trim_dataframe_at_empty_index(df, index_col):
    """
    Trim a dataframe by dropping rows starting from the first empty value in index_col.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe to trim
    index_col : int, str, or None
        Column to check for empty values. Can be column name (str) or column index (int).
        If None, returns the dataframe unchanged.
    
    Returns:
    --------
    pandas.DataFrame
        Trimmed dataframe with rows dropped from first empty value in index_col onwards.
        If index_col was specified, it will be set as the dataframe index.
    
    Examples:
    ---------
    # Trim using column name
    df_trimmed = trim_dataframe_at_empty_index(df, 'ID')
    
    # Trim using column index
    df_trimmed = trim_dataframe_at_empty_index(df, 0)
    
    # No trimming (returns original df)
    df_unchanged = trim_dataframe_at_empty_index(df, None)
    """
    
    # If index_col is None, return dataframe unchanged
    if index_col is None:
        return df.copy()
    
    # Make a copy to avoid modifying original
    df_result = df.copy()
    
    # Handle both column name and column index
    if isinstance(index_col, str):
        if index_col not in df_result.columns:
            raise ValueError(f"Column '{index_col}' not found in dataframe")
        index_column = df_result[index_col]
    else:
        if index_col >= len(df_result.columns):
            raise ValueError(f"Column index {index_col} out of range")
        index_column = df_result.iloc[:, index_col]
    
    # Find first row where index column is empty (NaN, None, or empty string)
    empty_mask = index_column.isna() | (index_column == '') | (index_column == ' ')
    
    if empty_mask.any():
        first_empty_idx = empty_mask.idxmax()
        df_result = df_result.loc[:first_empty_idx-1]  # Keep rows before first empty
    
    # Set the index column
    if isinstance(index_col, str):
        df_result = df_result.set_index(index_col)
    else:
        df_result = df_result.set_index(df_result.columns[index_col])
    
    return df_result


def select_one_treatment_per_patient(df, pneumonitis_col='Pneumonitis G0-1=0, G>/=2=1', 
                                   date_col='Date_SBRT_start'):
    """
    Select one treatment row per patient based on pneumonitis occurrence or latest date.
    
    Uses the patient_id, treatment_id, and lesion_id columns to identify unique patients
    and their treatments/lesions.
    
    For each patient:
    1. If the patient has any treatment with pneumonitis (pneumonitis_col == 1), 
       select that treatment row
    2. If the patient has multiple treatments with pneumonitis, select the first one found
    3. If the patient has no treatments with pneumonitis, select the treatment 
       with the latest date (date_col)
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing treatment data with patient_id, treatment_id, lesion_id columns
    pneumonitis_col : str, default 'Pneumonitis G0-1=0, G>/=2=1'
        Name of the column indicating pneumonitis (1 = pneumonitis, 0 = no pneumonitis)
    date_col : str, default 'Date_SBRT_start'
        Name of the column containing treatment dates
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with one row per patient, selected according to the criteria
    
    Notes:
    ------
    - Requires patient_id, treatment_id, and lesion_id columns in the dataframe
    - Date column should be in a format that pandas can parse/compare
    - If pneumonitis column contains NaN values, they are treated as 0 (no pneumonitis)
    - Selection considers all treatments and lesions for each patient
    
    Examples:
    ---------
    # Basic usage (assumes standard column names)
    df_selected = select_one_treatment_per_patient(df_full)
    
    # With custom column names
    df_selected = select_one_treatment_per_patient(
        df_full, 
        pneumonitis_col='HasPneumonitis',
        date_col='TreatmentDate'
    )
    """
    
    # Validate required columns exist
    required_cols = ['patient_id', 'treatment_id', 'lesion_id']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}. "
                        f"Use extract_id_columns() first to create these columns.")
    
    # Make a copy to avoid modifying original dataframe
    df_work = df.copy()
    
    # Handle missing values in pneumonitis column (treat as 0)
    df_work[pneumonitis_col] = df_work[pneumonitis_col].fillna(0)
    
    # Convert date column to datetime for proper comparison
    df_work[date_col] = pd.to_datetime(df_work[date_col])
    
    selected_rows = []
    
    # Process each unique patient
    for patient_id in df_work['patient_id'].unique():
        patient_data = df_work[df_work['patient_id'] == patient_id].copy()
        
        # Check if patient has any treatments/lesions with pneumonitis
        has_pneumonitis = patient_data[pneumonitis_col] == 1
        
        if has_pneumonitis.any():
            # Select first treatment/lesion with pneumonitis (by dataframe order)
            selected_row = patient_data[has_pneumonitis].iloc[0:1]
        else:
            # Select treatment/lesion with latest date
            latest_date_idx = patient_data[date_col].idxmax()
            selected_row = patient_data.loc[[latest_date_idx]]
        
        selected_rows.append(selected_row)
    
    # Combine all selected rows
    result = pd.concat(selected_rows, ignore_index=False)
    
    # Sort by patient_id for consistent output
    result = result.sort_values('patient_id')
    
    return result


def extract_id_columns(df, pat_col=None):
    """
    Extract separate patient_id, treatment_id, and lesion_id columns from Pat codes.
    
    Takes Pat codes in format XXX.Y and creates:
    - patient_id: XXX (the part before the dot) as integer
    - treatment_id: Y (the part after the dot) as integer
    - lesion_id: Sequential numbering for duplicate patient-treatment combinations as integer
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe containing Pat codes
    pat_col : str or None, default None
        Column name containing Pat codes. If None, uses the dataframe index.
    
    Returns:
    --------
    pandas.DataFrame
        Dataframe with added patient_id, treatment_id, and lesion_id columns
        at the beginning, preserving all original columns and index
    
    Examples:
    ---------
    # Using index as Pat column
    df_with_ids = extract_id_columns(df)
    
    # Using specific column
    df_with_ids = extract_id_columns(df, pat_col='PatientCode')
    
    Notes:
    ------
    - Pat codes should be in format XXX.Y where XXX and Y are typically numeric
    - For duplicate patient-treatment combinations, lesion_id will be 1, 2, 3, etc.
    - Original Pat column/index is preserved unchanged
    - All ID columns are converted to integers
    """
    
    # Make a copy to avoid modifying original dataframe
    df_result = df.copy()
    
    # Get Pat codes from specified column or index
    if pat_col is None:
        pat_codes = df_result.index.astype(str)
    else:
        if pat_col not in df_result.columns:
            raise ValueError(f"Column '{pat_col}' not found in dataframe")
        pat_codes = df_result[pat_col].astype(str)
    
    # Extract patient_id and treatment_id by splitting on '.'
    split_codes = np.array(list(pat_codes.str.split('.', expand=False)))
    
    if split_codes.shape[1] < 2:
        raise ValueError("Pat codes must contain a '.' separator (format: XXX.Y)")
    
    # Convert to integers
    df_result['patient_id'] = pd.to_numeric(split_codes[:, 0], errors='coerce').astype(int)
    df_result['treatment_id'] = pd.to_numeric(split_codes[:, 1], errors='coerce').astype(int)
    
    # Initialize lesion_id as 1 for all rows (as integer)
    df_result['lesion_id'] = 1
    
    # Create lesion numbering for duplicate patient-treatment combinations
    lesion_counter = {}
    
    for idx, row in enumerate(df_result.iterrows()):
        patient_id = row[1]['patient_id']
        treatment_id = row[1]['treatment_id']
        key = (patient_id, treatment_id)
        
        # Increment counter for this patient-treatment combination
        if key not in lesion_counter.keys():
            lesion_counter[key] = 0
        
        lesion_counter[key] += 1
        df_result.iloc[idx, df_result.columns.get_loc('lesion_id')] = lesion_counter[key]
    
    # Reorder columns to place ID columns at the beginning
    id_cols = ['patient_id', 'treatment_id', 'lesion_id']
    other_cols = [col for col in df_result.columns if col not in id_cols]
    df_result = df_result[id_cols + other_cols]
    # Reset index so index values are unique
    df_result = df_result.reset_index()
    
    return df_result