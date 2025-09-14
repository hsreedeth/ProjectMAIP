# import argparse
# import pandas as pd
# import numpy as np
# import time
# from sklearn.experimental import enable_iterative_imputer
# from sklearn.impute import IterativeImputer
# from sklearn.linear_model import BayesianRidge
# from sklearn.preprocessing import StandardScaler
# from pathlib import Path

# # --- Configuration ---
# ROOT_DIR = Path(__file__).resolve().parents[1]
# RAW_DATA_PATH = ROOT_DIR / 'data' / '00_raw' / 'support2.csv'
# PROCESSED_PATH = ROOT_DIR / 'data' / '01_processed'

# # --- Helper Functions ---

# def generate_report(df, stage_name):
#     """Generates and prints a summary report for a DataFrame at a specific stage."""
#     print("\n" + "="*80)
#     print(f"REPORT: {stage_name}")
#     print("="*80)
#     print(f"\nShape of the dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    
#     missing_df = (
#         df.isnull().sum()
#         .to_frame('Missing Count')
#         .assign(Missing_Percent=lambda x: 100 * x['Missing Count'] / len(df))
#         .query("`Missing Count` > 0")
#         .sort_values('Missing Count', ascending=False)
#     )
    
#     print("\nMissing Data Summary:")
#     if not missing_df.empty:
#         print(missing_df.to_string())
#     else:
#         print("No missing values found.")
#     print("\n" + "="*80)

# def run_preprocessing(save_output=True):
#     """
#     Executes the full preprocessing, imputation, and view creation pipeline.
    
#     Args:
#         save_output (bool): If True, saves the final view files to disk.
#     """
#     # Load raw data
#     try:
#         data = pd.read_csv(RAW_DATA_PATH)
#     except FileNotFoundError:
#         print(f"Error: Raw data file not found at {RAW_DATA_PATH}")
#         return

#     # --- Initial Report ---
#     generate_report(data, "BEFORE PREPROCESSING")


#     # --- Step 0: Add a stable row ID if not present ---
#     if 'eid' not in data.columns:
#         data.insert(0, 'eid', range(1, len(data)+1))
    
#     # --- Step 1: Impute Normal Values & Drop Rows with Low Missingness ---
#     print("\n[Step 1] Applying initial imputations and row deletions...")
#     normal_values = {'alb': 3.5, 'pafi': 333.3, 'bili': 1.01, 'crea': 1.01, 
#                      'bun': 6.51, 'wblc': 9, 'urine': 2502}
#     data.fillna(value=normal_values, inplace=True)
    
#     missing_counts = data.isnull().sum()
#     low_missing_cols = missing_counts[(missing_counts <= 82) & (missing_counts > 0)]
#     if not low_missing_cols.empty:
#         print(f"Dropping rows with NAs in columns with <= 82 missing values: {list(low_missing_cols.index)}")
#         initial_rows = len(data)
#         data.dropna(subset=low_missing_cols.index, inplace=True)
#         print(f"Removed {initial_rows - len(data)} rows.")

#     # --- Step 2a: Create a Combined ADL Feature ---
#     print("\n[Step 2a] Creating combined ADL feature 'adlp_s'...")
#     data['adlp_s'] = data['adlp'].fillna(data['adls'])
#     print("Created 'adlp_s' by filling patient ADL with surrogate ADL.")

#     # --- Step 2b: Drop Highly Correlated / Redundant Columns ---
#     print("\n[Step 2b] Dropping redundant or pre-imputed columns...")
#     # Now we drop the original adlp and adls, keeping our new adlp_s
#     cols_to_drop = ['totcst', 'charges', 'surv2m', 'prg2m', 'adls', 'adlp', 'adlsc']
#     data.drop(columns=cols_to_drop, inplace=True, errors='ignore')
#     print(f"Dropped columns: {cols_to_drop}")

#     # --- Step 3: Ordinal and Categorical Encoding ---
#     print("\n[Step 3] Encoding categorical and ordinal features...")
#     sfdm2_map = {"<2 mo. follow-up": 5, "no(M2 and SIP pres)": 1, "adl>=4 (>=5 if sur)": 2, 
#                    "SIP>=30": 3, "Coma or Intub": 4}
#     data["sfdm2"] = data["sfdm2"].map(sfdm2_map)

#     income_map = {'under $11k': 1, '$11-$25k': 2, '$25-$50k': 3, '>$50k': 4}
#     data['income'] = data['income'].map(income_map)

#     data['edu'].fillna(data['edu'].mode()[0], inplace=True)
#     data['sex'] = data['sex'].map({'male': 1, 'female': 0})
#     data['ca'] = data['ca'].map({'yes': 1, 'no': 0, 'metastatic': 2})

#     # Manual One-Hot Encoding
#     data['dzgroup_arf_mosf'] = (data['dzgroup'] == 'ARF/MOSF w/Sepsis').astype(int)
#     data['dzgroup_chf'] = (data['dzgroup'] == 'CHF').astype(int)
#     data['dzgroup_copd'] = (data['dzgroup'] == 'COPD').astype(int)
#     data['dzgroup_lung_cancer'] = (data['dzgroup'] == 'Lung Cancer').astype(int)
#     data['dzgroup_mosf_malig'] = (data['dzgroup'] == 'MOSF w/Malig').astype(int)
#     data['dzgroup_coma'] = (data['dzgroup'] == 'Coma').astype(int)
#     data['dzgroup_cirrhosis'] = (data['dzgroup'] == 'Cirrhosis').astype(int)
#     data['dzgroup_colon_cancer'] = (data['dzgroup'] == 'Colon Cancer').astype(int)
#     data.drop('dzgroup', axis=1, inplace=True)

#     data['dzclass_arf_mosf'] = (data['dzclass'] == 'ARF/MOSF').astype(int)
#     data['dzclass_copd_chf_cirrhosis'] = (data['dzclass'] == 'COPD/CHF/Cirrhosis').astype(int)
#     data['dzclass_cancer'] = (data['dzclass'] == 'Cancer').astype(int)
#     data['dzclass_coma'] = (data['dzclass'] == 'Coma').astype(int)
#     data.drop('dzclass', axis=1, inplace=True)

#     data['race_white'] = (data['race'] == 'white').astype(int)
#     data['race_black'] = (data['race'] == 'black').astype(int)
#     data['race_hispanic'] = (data['race'] == 'hispanic').astype(int)
#     data['race_other'] = (data['race'] == 'other').astype(int)
#     data['race_asian'] = (data['race'] == 'asian').astype(int)
#     data.drop('race', axis=1, inplace=True)
    
#     data['dnr_no_dnr'] = (data['dnr'] == 'no dnr').astype(int)
#     data['dnr_after_sadm'] = (data['dnr'] == 'dnr after sadm').astype(int)
#     data['dnr_before_sadm'] = (data['dnr'] == 'dnr before sadm').astype(int)
#     data.drop('dnr', axis=1, inplace=True)

#     print("Encoded ordinal features and manually one-hot encoded multi-class features.")

#     # --- Step 4: MICE Imputation for Remaining Missing Values ---
#     print("\n[Step 4] Performing MICE imputation...")
#     imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=100, random_state=42, verbose=1)
    
#     start_time = time.time()
#     imputed_data_array = imputer.fit_transform(data)
#     end_time = time.time()

#     processed_df = pd.DataFrame(imputed_data_array, columns=data.columns)
#     print(f"MICE imputation completed in {end_time - start_time:.2f} seconds.")

#     # --- Step 4a: Round Imputed Ordinal/Discrete Features ---
#     print("\n[Step 4a] Rounding and clipping discrete features post-imputation...")
#     processed_df['income'] = processed_df['income'].round().clip(1, 4)
#     processed_df['sfdm2'] = processed_df['sfdm2'].round().clip(1, 5)
#     processed_df['adlp_s'] = processed_df['adlp_s'].round().clip(0, data['adlp_s'].max())
#     print("Rounded and clipped 'income', 'sfdm2', and 'adlp_s'.")

#     # --- Step 4b: Correct Negative Cost Artifacts ---
#     print("\n[Step 4b] Correcting for negative cost values from imputation...")
#     if 'totmcst' in processed_df.columns:
#         neg_costs_count = (processed_df['totmcst'] < 0).sum()
#         if neg_costs_count > 0:
#             processed_df['totmcst'] = processed_df['totmcst'].clip(lower=0)
#             print(f"Corrected {neg_costs_count} instances of negative 'totmcst' by clipping at 0.")
#         else:
#             print("No negative costs found.")

#     # --- Final Report ---
#     generate_report(processed_df, "AFTER PREPROCESSING AND IMPUTATION")

#     # --- Step 5: Create and Save Final Feature Views ---
#     print("\n[Step 5] Creating and scaling final C/P/S views...")
#     schema = {
#         'C_view_cols': ['num.co', 'diabetes', 'dementia'] + [c for c in processed_df.columns if 'dzgroup_' in c or 'dzclass_' in c or 'ca' == c],
#         'P_view_cols': ['age', 'scoma', 'avtisst', 'sps', 'aps', 'meanbp', 'wblc', 'hrt', 'resp', 'temp', 'pafi', 'alb', 'bili', 'crea', 'sod', 'ph', 'glucose', 'bun', 'urine'],
#         'S_view_cols': ['sex', 'income', 'edu', 'adlp_s'] + [c for c in processed_df.columns if 'race_' in c or 'dnr_' in c],
#         'Y_validation_cols': ['death', 'hospdead', 'd.time', 'slos', 'hday', 'sfdm2', 'surv6m', 'prg6m', 'dnrday', 'totmcst']
#     }
    
#     # Create View DataFrames
#     C_view = processed_df[schema['C_view_cols']]
#     P_view = processed_df[schema['P_view_cols']]
#     S_view = processed_df[schema['S_view_cols']]
    
#     # Scale P-view
#     scaler = StandardScaler()
#     P_view_scaled = pd.DataFrame(scaler.fit_transform(P_view), columns=schema['P_view_cols'])
    
#     # Isolate Validation set
#     Y_validation = processed_df[[c for c in schema['Y_validation_cols'] if c in processed_df.columns]]

#     if save_output:
#         print(f"\nSaving processed views to {PROCESSED_PATH}...")
#         PROCESSED_PATH.mkdir(exist_ok=True)
#         C_view.to_csv(PROCESSED_PATH / 'C_view.csv', index=False)
#         P_view_scaled.to_csv(PROCESSED_PATH / 'P_view_scaled.csv', index=False)
#         S_view.to_csv(PROCESSED_PATH / 'S_view.csv', index=False)
#         Y_validation.to_csv(PROCESSED_PATH / 'Y_validation.csv', index=False)
#         processed_df.to_csv(PROCESSED_PATH / 'support_preprocessed.csv', index=False)
#         print("Files saved successfully.")

#     return processed_df

# def main():
#     parser = argparse.ArgumentParser(description="Run the SUPPORT-II preprocessing pipeline.")
#     parser.add_argument(
#         '--no-save', 
#         action='store_true',
#         help="Run the script without saving the output files (dry run)."
#     )
#     args = parser.parse_args()
    
#     run_preprocessing(save_output=not args.no_save)

# if __name__ == "__main__":
#     main()

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler

# --- Configuration ---
ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = ROOT_DIR / 'data' / '00_raw' / 'support2.csv'
PROCESSED_PATH = ROOT_DIR / 'data' / '01_processed'
PROCESSED_PATH.mkdir(parents=True, exist_ok=True)

# Outcomes / validation columns (exclude from X-imputation)
Y_COLS = [
    'death', 'hospdead', 'd.time', 'slos', 'hday', 'sfdm2',
    'surv6m', 'prg6m', 'dnrday', 'totmcst'
]

# Core P-view (for optional strict row-drop)
P_VIEW_BASICS = [
    'age', 'scoma', 'avtisst', 'sps', 'aps', 'meanbp', 'wblc', 'hrt',
    'resp', 'temp', 'pafi', 'alb', 'bili', 'crea', 'sod', 'ph',
    'glucose', 'bun', 'urine'
]

def generate_report(df: pd.DataFrame, stage_name: str) -> None:
    print("\n" + "="*80)
    print(f"REPORT: {stage_name}")
    print("="*80)
    print(f"\nShape: {df.shape[0]} rows × {df.shape[1]} cols")
    miss = (df.isnull().sum()
            .to_frame('Missing Count')
            .assign(Missing_Percent=lambda x: 100 * x['Missing Count'] / len(df))
            .query("`Missing Count` > 0")
            .sort_values('Missing Count', ascending=False))
    print("\nMissing Data Summary:")
    print(miss.to_string() if not miss.empty else "No missing values found.")
    print("\n" + "="*80)

def run_preprocessing(save_output: bool = True, strict_row_drop: bool = False) -> pd.DataFrame:
    # Load
    try:
        data = pd.read_csv(RAW_DATA_PATH)
    except FileNotFoundError:
        print(f"[ERROR] Raw data not found at {RAW_DATA_PATH}")
        return

    generate_report(data, "BEFORE PREPROCESSING")

    # --- Step 0: Ensure a stable ID ---
    if 'eid' not in data.columns:
        data.insert(0, 'eid', range(1, len(data) + 1))  # FIX: len(data), not len(df)

    # --- Step 1: Gentle initial fills for a few physiologic vars (optional domain priors) ---
    normal_values = {
        'alb': 3.5, 'pafi': 333.3, 'bili': 1.01, 'crea': 1.01,
        'bun': 6.51, 'wblc': 9, 'urine': 2502
    }
    data = data.copy()
    for k, v in normal_values.items():
        if k in data.columns:
            data[k] = data[k].fillna(v)

    # Optional: restrict row drops to P-view basics only (avoids culling by outcome columns)
    if strict_row_drop:
        cols_present = [c for c in P_VIEW_BASICS if c in data.columns]
        if cols_present:
            miss_counts = data[cols_present].isnull().sum()
            low_missing_cols = miss_counts[(miss_counts <= 82) & (miss_counts > 0)].index.tolist()
            if low_missing_cols:
                before = len(data)
                data = data.dropna(subset=low_missing_cols)
                print(f"[RowDrop] Removed {before - len(data)} rows due to NA in low-missing P columns: {low_missing_cols}")

    # --- Step 2a: Build ADL surrogate ---
    if 'adlp' in data.columns and 'adls' in data.columns:
        data['adlp_s'] = data['adlp'].fillna(data['adls'])
    else:
        # if either missing, just ensure column exists (keeps pipeline running)
        data['adlp_s'] = data.get('adlp', pd.Series(index=data.index))

    # --- Step 2b: Drop redundant columns (keep your rationale) ---
    cols_to_drop = ['totcst', 'charges', 'surv2m', 'prg2m', 'adls', 'adlp', 'adlsc']
    data = data.drop(columns=[c for c in cols_to_drop if c in data.columns], errors='ignore')

    # --- Step 3: Encode categorical/ordinal ---
    # sfdm2
    if 'sfdm2' in data.columns:
        sfdm2_map = {"<2 mo. follow-up": 5, "no(M2 and SIP pres)": 1,
                     "adl>=4 (>=5 if sur)": 2, "SIP>=30": 3, "Coma or Intub": 4}
        data['sfdm2'] = data['sfdm2'].map(sfdm2_map)

    # income
    if 'income' in data.columns:
        data['income'] = data['income'].map({'under $11k': 1, '$11-$25k': 2, '$25-$50k': 3, '>$50k': 4})

    # edu (mode impute)
    if 'edu' in data.columns and data['edu'].isnull().any():
        data['edu'] = data['edu'].fillna(data['edu'].mode(dropna=True).iloc[0])

    # sex
    if 'sex' in data.columns:
        data['sex'] = data['sex'].map({'male': 1, 'female': 0})

    # ca (0/1/2)
    if 'ca' in data.columns:
        data['ca'] = data['ca'].map({'yes': 1, 'no': 0, 'metastatic': 2})

    # Manual OHE examples (only if originals exist)
    if 'dzgroup' in data.columns:
        dzg_vals = {
            'ARF/MOSF w/Sepsis': 'arf_mosf',
            'CHF': 'chf',
            'COPD': 'copd',
            'Lung Cancer': 'lung_cancer',
            'MOSF w/Malig': 'mosf_malig',
            'Coma': 'coma',
            'Cirrhosis': 'cirrhosis',
            'Colon Cancer': 'colon_cancer'
        }
        for k, suf in dzg_vals.items():
            data[f'dzgroup_{suf}'] = (data['dzgroup'] == k).astype('Int64')
        data = data.drop(columns=['dzgroup'])

    if 'dzclass' in data.columns:
        dzc_vals = {
            'ARF/MOSF': 'arf_mosf',
            'COPD/CHF/Cirrhosis': 'copd_chf_cirrhosis',
            'Cancer': 'cancer',
            'Coma': 'coma'
        }
        for k, suf in dzc_vals.items():
            data[f'dzclass_{suf}'] = (data['dzclass'] == k).astype('Int64')
        data = data.drop(columns=['dzclass'])

    if 'race' in data.columns:
        for r in ['white', 'black', 'hispanic', 'other', 'asian']:
            data[f'race_{r}'] = (data['race'] == r).astype('Int64')
        data = data.drop(columns=['race'])

    if 'dnr' in data.columns:
        for d in ['no dnr', 'dnr after sadm', 'dnr before sadm']:
            data[f'dnr_{d.replace(" ", "_")}'] = (data['dnr'] == d).astype('Int64')
        data = data.drop(columns=['dnr'])

    # --- Step 4: MICE on X only (exclude eid + outcomes) ---
    # Split into X (features) and Y (validation/outcomes)
    y_cols_present = [c for c in Y_COLS if c in data.columns]
    X = data.drop(columns=y_cols_present, errors='ignore')
    Y = data[y_cols_present].copy() if y_cols_present else pd.DataFrame(index=data.index)

    # Never impute over eid
    id_col = 'eid'
    if id_col not in X.columns:
        raise ValueError("Missing 'eid' after loading. It must exist before imputation.")
    X_id = X[[id_col]].copy()
    X_wo_id = X.drop(columns=[id_col])

    print("\n[MICE] Imputing X (without eid and without outcomes)...")
    imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=50, random_state=42, verbose=0)
    t0 = time.time()
    X_imp = pd.DataFrame(imputer.fit_transform(X_wo_id), columns=X_wo_id.columns, index=X_wo_id.index)
    print(f"[MICE] Done in {time.time() - t0:.2f}s.")
    dump(imputer, PROCESSED_PATH / 'imputer_X.joblib')

    processed_df = pd.concat([X_id, X_imp, Y], axis=1)

    # --- Step 4a: Round/clip discrete fields ---
    # define discrete / bounded fields present
    discrete_01 = [c for c in processed_df.columns if c.startswith(('race_', 'dnr_', 'dzgroup_', 'dzclass_'))]
    for c in ['sex']:
        if c in processed_df.columns:
            discrete_01.append(c)
    # clip to {0,1}
    for c in discrete_01:
        processed_df[c] = processed_df[c].round().clip(0, 1)

    if 'ca' in processed_df.columns:
        processed_df['ca'] = processed_df['ca'].round().clip(0, 2)

    if 'income' in processed_df.columns:
        processed_df['income'] = processed_df['income'].round().clip(1, 4)

    if 'sfdm2' in processed_df.columns:
        processed_df['sfdm2'] = processed_df['sfdm2'].round().clip(1, 5)

    if 'adlp_s' in processed_df.columns:
        # Use original (pre-impute) max if available
        try:
            orig_max = data['adlp_s'].max(skipna=True)
        except Exception:
            orig_max = processed_df['adlp_s'].max(skipna=True)
        processed_df['adlp_s'] = processed_df['adlp_s'].round().clip(lower=0, upper=orig_max if pd.notnull(orig_max) else None)

    generate_report(processed_df, "AFTER PREPROCESSING (X-imputed, Y untouched)")

    # --- Step 5: Build views (carry eid!) ---
    C_view_cols = ['eid', 'num.co', 'diabetes', 'dementia'] + \
                  [c for c in processed_df.columns if c.startswith(('dzgroup_', 'dzclass_'))] + \
                  (['ca'] if 'ca' in processed_df.columns else [])

    P_view_cols = ['eid'] + [c for c in P_VIEW_BASICS if c in processed_df.columns]

    S_view_cols = ['eid'] + [c for c in ['sex', 'income', 'edu', 'adlp_s'] if c in processed_df.columns] + \
                  [c for c in processed_df.columns if c.startswith(('race_', 'dnr_'))]

    Y_validation_cols = [c for c in Y_COLS if c in processed_df.columns]

    C_view = processed_df.loc[:, [c for c in C_view_cols if c in processed_df.columns]].copy()
    P_view = processed_df.loc[:, [c for c in P_view_cols if c in processed_df.columns]].copy()
    S_view = processed_df.loc[:, [c for c in S_view_cols if c in processed_df.columns]].copy()
    Y_validation = processed_df.loc[:, Y_validation_cols].copy() if Y_validation_cols else pd.DataFrame(index=processed_df.index)

    # Scale P-view (exclude eid)
    scaler = StandardScaler()
    P_view_scaled = P_view.copy()
    p_cols = [c for c in P_view.columns if c != 'eid']
    P_view_scaled.loc[:, p_cols] = scaler.fit_transform(P_view[p_cols])
    dump(scaler, PROCESSED_PATH / 'scaler_P.joblib')

    if save_output:
        print(f"\nSaving processed outputs to {PROCESSED_PATH} ...")
        C_view.to_csv(PROCESSED_PATH / 'C_view.csv', index=False)
        P_view_scaled.to_csv(PROCESSED_PATH / 'P_view_scaled.csv', index=False)
        S_view.to_csv(PROCESSED_PATH / 'S_view.csv', index=False)
        if not Y_validation.empty:
            Y_validation.to_csv(PROCESSED_PATH / 'Y_validation.csv', index=False)
        processed_df.to_csv(PROCESSED_PATH / 'support_preprocessed.csv', index=False)
        print("Saved.")

    return processed_df

def main():
    ap = argparse.ArgumentParser(description="Run the SUPPORT-II preprocessing pipeline.")
    ap.add_argument('--no-save', action='store_true', help="Run without saving outputs.")
    ap.add_argument('--strict-row-drop', action='store_true', help="Drop rows with NA in low-missing P columns (conservative).")
    args = ap.parse_args()
    run_preprocessing(save_output=not args.no_save, strict_row_drop=args.strict_row_drop)

if __name__ == "__main__":
    main()
