import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal, bernoulli
from scipy.optimize import brentq

def generate_synthetic_pneumonitis(
    n_patients: int = 126,
    n_treatments: int = 150,
    n_rows: int = 167,
    seed: int = 0,
    target_event_rate: float = None,
    easy: bool = False,
) -> pd.DataFrame:
    """
    Fixed version of synthetic lesion-level dataset with proper event rate calibration.
      
    The key fix: Uses numerical optimization to find the correct shift that achieves
    the target event rate, rather than assuming mean(logistic(x + shift)) = logistic(mean(x) + shift).
    
    Now includes Date_SBRT_start generation for compatibility with select_one_treatment_per_patient.
    
    Parameters:
    -----------
    n_patients : int, default=126
        Number of unique patients
    n_treatments : int, default=150
        Total number of treatments across all patients
    n_rows : int, default=167
        Total number of lesions (rows) in final dataset
    seed : int, default=0
        Random seed for reproducibility
    target_event_rate : float, optional
        Target event rate. If None, uses 0.12 for realistic and 0.25 for easy
    easy : bool, default=False
        If True, generates easy dataset with large effect sizes (AUROC > 0.9)
        If False, generates realistic dataset with subtle signals
    
    Returns:
    --------
    pd.DataFrame
        Synthetic dataset with patient, treatment, and lesion level data including Date_SBRT_start
    """

    if not (n_patients <= n_treatments <= n_rows):
        raise ValueError("Need n_patients ≤ n_treatments ≤ n_rows")

    rng = np.random.default_rng(seed)

    # Set default event rates based on difficulty
    if target_event_rate is None:
        target_event_rate = 0.25 if easy else 0.12

    # ───────────────────────── patient-level variables ──────────────────────────
    pid = np.arange(n_patients)

    age = rng.normal(70, 8, n_patients)
    dlco = rng.normal(55, 15, n_patients)
    fev1 = rng.normal(1.9, 0.5, n_patients)
    cci = rng.poisson(2, n_patients)
    sex = bernoulli.rvs(0.4 if easy else 0.35, size=n_patients, random_state=rng)
    smoker = bernoulli.rvs(0.6 if easy else 0.55, size=n_patients, random_state=rng)
    lower = bernoulli.rvs(0.4, size=n_patients, random_state=rng)
    periTx = bernoulli.rvs(0.3, size=n_patients, random_state=rng)

    # Correlated DVH metrics (log-normal base)
    if easy:
        mvn = multivariate_normal(mean=[3.3, 1.6],
                                  cov=[[0.10, 0.05],
                                       [0.05, 0.15]])
    else:
        mvn = multivariate_normal(mean=[3.2, 1.5],
                                  cov=[[0.15, 0.10],
                                       [0.10, 0.25]])
    
    log_dose = mvn.rvs(size=n_patients, random_state=rng)
    v_ptv_cc, mld_eqd2 = np.exp(log_dose.T)

    # Derive V5/V10/V20 % from MLD with noise & monotonicity
    if easy:
        lung_v20 = 20 + 0.9 * (mld_eqd2 - mld_eqd2.mean())
        lung_v10 = lung_v20 + 10
        lung_v5 = lung_v10 + 10
    else:
        lung_v20 = np.clip(rng.normal(25, 7, n_patients) +
                           0.8 * (mld_eqd2 - mld_eqd2.mean()), 0, 70)
        lung_v10 = lung_v20 + rng.normal(10, 4, n_patients)  # > V20
        lung_v5 = lung_v10 + rng.normal(10, 4, n_patients)   # > V10

    # ───────── true logistic link (three real predictors) ──────────
    Z = lambda x: (x - x.mean()) / x.std()
    
    if easy:
        # Large effect sizes for easy separation
        lin = (-1.6 +
               2.20 * Z(v_ptv_cc) +
               2.00 * Z(mld_eqd2) -
               2.50 * Z(dlco))
    else:
        # Subtle effects for realistic challenge
        lin = (-2.0 +
               0.55 * Z(v_ptv_cc) +
               0.45 * Z(mld_eqd2) -
               0.35 * Z(dlco))

    def mean_prob(shift):
        """Return mean logistic probability at given shift."""
        return np.mean(1 / (1 + np.exp(-(lin + shift))))

    # root of  mean_prob(shift) – target_event_rate  on a wide bracket
    f = lambda s: mean_prob(s) - target_event_rate
    shift = brentq(f, -10, 10)

    p_event = 1 / (1 + np.exp(-(lin + shift)))
    outcome = rng.binomial(1, p_event)

    print("Event-rate calibration:")
    print(f"  Target  : {target_event_rate:.4f}")
    print(f"  Achieved: {p_event.mean():.4f}")
    print(f"  Outcomes: {outcome.mean():.4f}")
    print(f"  Shift   : {shift:.4f}")

    pat_df = pd.DataFrame({
        "patient_id": pid.astype("int32"),
        "Age_SBRT": age,
        "Sex (0 = F, 1 = M)": sex,
        "Former_or_current_smoker": smoker,
        "lower lobe =1, others =0": lower,
        "peri-RT Systemic_therapy_prior_RT (3M)_yes_1_no_0": periTx,
        "CCI score": cci,
        "Baseline_FEV1 [L]": fev1,
        "Baseline_DLCOcSB [%] ": dlco,
        "V(PTV1) [cc]": v_ptv_cc,
        "total_lung_V_5Gy_(%)": lung_v5,
        "total_lung_V_10Gy_(%)": lung_v10,
        "total_lung_V_20Gy_(%)": lung_v20,
        "total_lung_Dmean_[Gy]_(EQD2)": mld_eqd2,
        "Pneumonitis G0-1=0, G>/=2=1": outcome,
    })

    # Missing DLCO (less missing for easy dataset)
    missing_rate = 0.05 if easy else 0.10
    pat_df.loc[rng.random(n_patients) < missing_rate, "Baseline_DLCOcSB [%] "] = np.nan

    # ───────────────────── treatments per patient ─────────────────────
    treat_per_pat = np.ones(n_patients, dtype=int)
    for _ in range(n_treatments - n_patients):
        treat_per_pat[rng.integers(0, n_patients)] += 1

    lesions_per_treat = np.ones(n_treatments, dtype=int)
    for _ in range(n_rows - n_treatments):
        lesions_per_treat[rng.integers(0, n_treatments)] += 1

    # ───────────────────── treatment dates generation ─────────────────────
    # Generate realistic treatment dates for each patient/treatment combination
    
    date_rng = np.random.default_rng(seed + 999)  # Different seed for dates
    start_date = pd.Timestamp('2018-01-01')
    end_date = pd.Timestamp('2024-12-31')
    date_range_days = (end_date - start_date).days
    
    # Create treatment dates mapping
    treatment_dates = {}
    
    for patient_idx in range(n_patients):
        patient_treatments = list(range(1, treat_per_pat[patient_idx] + 1))
        n_patient_treatments = len(patient_treatments)
        
        if n_patient_treatments == 1:
            # Single treatment - random date in range
            base_days = date_rng.integers(0, date_range_days)
            base_date = start_date + pd.Timedelta(days=int(base_days))
            treatment_dates[(patient_idx, 1)] = base_date
        else:
            # Multiple treatments - space them out realistically
            # Leave room for spacing treatments
            max_base_days = date_range_days - (n_patient_treatments * 90)
            base_days = date_rng.integers(0, max(max_base_days, 365))
            base_date = start_date + pd.Timedelta(days=int(base_days))
            
            for i, treatment_id in enumerate(patient_treatments):
                # Space treatments 30-180 days apart
                days_offset = i * date_rng.integers(30, 180)
                treatment_date = base_date + pd.Timedelta(days=int(days_offset))
                treatment_dates[(patient_idx, treatment_id)] = treatment_date

    # ───────────────────── lesion-level expansion ─────────────────────
    rows = []
    treat_global = 0
    for pid_i, row in pat_df.iterrows():
        for t_local in range(1, treat_per_pat[pid_i] + 1):
            n_les = lesions_per_treat[treat_global]
            treat_global += 1
            for l_local in range(1, n_les + 1):
                r = row.copy()
                r["treatment_id"] = np.int32(t_local)   # starts at 1 per patient
                r["lesion_id"] = np.int32(l_local)      # starts at 1 per treatment
                
                # Add treatment date
                r["Date_SBRT_start"] = treatment_dates[(pid_i, t_local)]
                
                # Lesion-level jitter (less jitter for easy dataset)
                jitter_std = 0.02 if easy else 0.05
                r["V(PTV1) [cc]"] *= rng.normal(1, jitter_std)
                r["total_lung_Dmean_[Gy]_(EQD2)"] *= rng.normal(1, jitter_std)
                rows.append(r)

    df = pd.DataFrame(rows).astype({
        "patient_id": "int32",
        "treatment_id": "int32",
        "lesion_id": "int32",
    })

    df = df.reset_index(drop=True)

    # Sanity checks
    assert df.patient_id.nunique() == n_patients
    assert df.shape[0] == n_rows
    assert df.groupby("patient_id").treatment_id.min().eq(1).all()
    
    # Verify date generation worked
    assert "Date_SBRT_start" in df.columns
    assert df["Date_SBRT_start"].notna().all()
    
    # For patients with multiple treatments, dates should be in order
    for patient_id in df.patient_id.unique():
        patient_data = df[df.patient_id == patient_id]
        if patient_data.treatment_id.nunique() > 1:
            treatment_dates_check = patient_data.groupby('treatment_id')['Date_SBRT_start'].first().sort_index()
            assert treatment_dates_check.is_monotonic_increasing, f"Dates not ordered for patient {patient_id}"

    print(f"Generated dataset with Date_SBRT_start column")
    print(f"Date range: {df['Date_SBRT_start'].min()} to {df['Date_SBRT_start'].max()}")

    return df

def validate_dataset(df, predictors, endpoint, dataset_type):
    """Validate dataset against expected configuration."""
    expected_cols = set(["patient_id", "treatment_id", "lesion_id"] + predictors + [endpoint])
    expected_cols |= {"Date_SBRT_start"}          # allow date column
    actual_cols = set(df.columns)
    
    missing_cols = expected_cols - actual_cols
    extra_cols = actual_cols - expected_cols
    
    if missing_cols:
        print(f"Warning: Missing expected columns in {dataset_type}: {missing_cols}")
    if extra_cols:
        print(f"Info: Extra columns in {dataset_type}: {extra_cols}")
    
    # Check endpoint distribution
    if endpoint in df.columns:
        endpoint_stats = df[endpoint].value_counts()
        event_rate = endpoint_stats.get(1, 0) / len(df)
        print(f"{dataset_type.title()} event rate: {event_rate:.3f}")
        print(f"{dataset_type.title()} endpoint distribution:")
        for value, count in endpoint_stats.items():
            percentage = (count / len(df)) * 100
            print(f"  {value}: {count} ({percentage:.1f}%)")