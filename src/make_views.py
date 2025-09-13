import argparse
import pandas as pd
import numpy as np
import time
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# --- Configuration ---
ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = ROOT_DIR / 'data' / '00_raw' / 'support2.csv'
PROCESSED_PATH = ROOT_DIR / 'data' / '01_processed'

# --- Helper Functions ---

def generate_report(df, stage_name):
    """Generates and prints a summary report for a DataFrame at a specific stage."""
    print("\n" + "="*80)
    print(f"REPORT: {stage_name}")
    print("="*80)
    print(f"\nShape of the dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    
    missing_df = (
        df.isnull().sum()
        .to_frame('Missing Count')
        .assign(Missing_Percent=lambda x: 100 * x['Missing Count'] / len(df))
        .query("`Missing Count` > 0")
        .sort_values('Missing Count', ascending=False)
    )
    
    print("\nMissing Data Summary:")
    if not missing_df.empty:
        print(missing_df.to_string())
    else:
        print("No missing values found.")
    print("\n" + "="*80)

def run_preprocessing(save_output=True):
    """
    Executes the full preprocessing, imputation, and view creation pipeline.
    
    Args:
        save_output (bool): If True, saves the final view files to disk.
    """
    # Load raw data
    try:
        data = pd.read_csv(RAW_DATA_PATH)
    except FileNotFoundError:
        print(f"Error: Raw data file not found at {RAW_DATA_PATH}")
        return

    # --- Initial Report ---
    generate_report(data, "BEFORE PREPROCESSING")
    
    # --- Step 1: Impute Normal Values & Drop Rows with Low Missingness ---
    print("\n[Step 1] Applying initial imputations and row deletions...")
    normal_values = {'alb': 3.5, 'pafi': 333.3, 'bili': 1.01, 'crea': 1.01, 
                     'bun': 6.51, 'wblc': 9, 'urine': 2502}
    data.fillna(value=normal_values, inplace=True)
    
    missing_counts = data.isnull().sum()
    low_missing_cols = missing_counts[(missing_counts <= 82) & (missing_counts > 0)]
    if not low_missing_cols.empty:
        print(f"Dropping rows with NAs in columns with <= 82 missing values: {list(low_missing_cols.index)}")
        initial_rows = len(data)
        data.dropna(subset=low_missing_cols.index, inplace=True)
        print(f"Removed {initial_rows - len(data)} rows.")

    # --- Step 2a: Create a Combined ADL Feature ---
    print("\n[Step 2a] Creating combined ADL feature 'adlp_s'...")
    data['adlp_s'] = data['adlp'].fillna(data['adls'])
    print("Created 'adlp_s' by filling patient ADL with surrogate ADL.")

    # --- Step 2b: Drop Highly Correlated / Redundant Columns ---
    print("\n[Step 2b] Dropping redundant or pre-imputed columns...")
    # Now we drop the original adlp and adls, keeping our new adlp_s
    cols_to_drop = ['totcst', 'charges', 'surv2m', 'prg2m', 'adls', 'adlp', 'adlsc']
    data.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    print(f"Dropped columns: {cols_to_drop}")

    # --- Step 3: Ordinal and Categorical Encoding ---
    print("\n[Step 3] Encoding categorical and ordinal features...")
    sfdm2_map = {"<2 mo. follow-up": 5, "no(M2 and SIP pres)": 1, "adl>=4 (>=5 if sur)": 2, 
                   "SIP>=30": 3, "Coma or Intub": 4}
    data["sfdm2"] = data["sfdm2"].map(sfdm2_map)

    income_map = {'under $11k': 1, '$11-$25k': 2, '$25-$50k': 3, '>$50k': 4}
    data['income'] = data['income'].map(income_map)

    data['edu'].fillna(data['edu'].mode()[0], inplace=True)
    data['sex'] = data['sex'].map({'male': 1, 'female': 0})
    data['ca'] = data['ca'].map({'yes': 1, 'no': 0, 'metastatic': 2})

    # Manual One-Hot Encoding
    data['dzgroup_arf_mosf'] = (data['dzgroup'] == 'ARF/MOSF w/Sepsis').astype(int)
    data['dzgroup_chf'] = (data['dzgroup'] == 'CHF').astype(int)
    data['dzgroup_copd'] = (data['dzgroup'] == 'COPD').astype(int)
    data['dzgroup_lung_cancer'] = (data['dzgroup'] == 'Lung Cancer').astype(int)
    data['dzgroup_mosf_malig'] = (data['dzgroup'] == 'MOSF w/Malig').astype(int)
    data['dzgroup_coma'] = (data['dzgroup'] == 'Coma').astype(int)
    data['dzgroup_cirrhosis'] = (data['dzgroup'] == 'Cirrhosis').astype(int)
    data['dzgroup_colon_cancer'] = (data['dzgroup'] == 'Colon Cancer').astype(int)
    data.drop('dzgroup', axis=1, inplace=True)

    data['dzclass_arf_mosf'] = (data['dzclass'] == 'ARF/MOSF').astype(int)
    data['dzclass_copd_chf_cirrhosis'] = (data['dzclass'] == 'COPD/CHF/Cirrhosis').astype(int)
    data['dzclass_cancer'] = (data['dzclass'] == 'Cancer').astype(int)
    data['dzclass_coma'] = (data['dzclass'] == 'Coma').astype(int)
    data.drop('dzclass', axis=1, inplace=True)

    data['race_white'] = (data['race'] == 'white').astype(int)
    data['race_black'] = (data['race'] == 'black').astype(int)
    data['race_hispanic'] = (data['race'] == 'hispanic').astype(int)
    data['race_other'] = (data['race'] == 'other').astype(int)
    data['race_asian'] = (data['race'] == 'asian').astype(int)
    data.drop('race', axis=1, inplace=True)
    
    data['dnr_no_dnr'] = (data['dnr'] == 'no dnr').astype(int)
    data['dnr_after_sadm'] = (data['dnr'] == 'dnr after sadm').astype(int)
    data['dnr_before_sadm'] = (data['dnr'] == 'dnr before sadm').astype(int)
    data.drop('dnr', axis=1, inplace=True)

    print("Encoded ordinal features and manually one-hot encoded multi-class features.")

    # --- Step 4: MICE Imputation for Remaining Missing Values ---
    print("\n[Step 4] Performing MICE imputation...")
    imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=100, random_state=42, verbose=1)
    
    start_time = time.time()
    imputed_data_array = imputer.fit_transform(data)
    end_time = time.time()

    processed_df = pd.DataFrame(imputed_data_array, columns=data.columns)
    print(f"MICE imputation completed in {end_time - start_time:.2f} seconds.")

    # --- Step 4a: Round Imputed Ordinal/Discrete Features ---
    print("\n[Step 4a] Rounding and clipping discrete features post-imputation...")
    processed_df['income'] = processed_df['income'].round().clip(1, 4)
    processed_df['sfdm2'] = processed_df['sfdm2'].round().clip(1, 5)
    processed_df['adlp_s'] = processed_df['adlp_s'].round().clip(0, data['adlp_s'].max())
    print("Rounded and clipped 'income', 'sfdm2', and 'adlp_s'.")

    # --- Step 4b: Correct Negative Cost Artifacts ---
    print("\n[Step 4b] Correcting for negative cost values from imputation...")
    if 'totmcst' in processed_df.columns:
        neg_costs_count = (processed_df['totmcst'] < 0).sum()
        if neg_costs_count > 0:
            processed_df['totmcst'] = processed_df['totmcst'].clip(lower=0)
            print(f"Corrected {neg_costs_count} instances of negative 'totmcst' by clipping at 0.")
        else:
            print("No negative costs found.")

    # --- Final Report ---
    generate_report(processed_df, "AFTER PREPROCESSING AND IMPUTATION")

    # --- Step 5: Create and Save Final Feature Views ---
    print("\n[Step 5] Creating and scaling final C/P/S views...")
    schema = {
        'C_view_cols': ['num.co', 'diabetes', 'dementia'] + [c for c in processed_df.columns if 'dzgroup_' in c or 'dzclass_' in c or 'ca' == c],
        'P_view_cols': ['age', 'scoma', 'avtisst', 'sps', 'aps', 'meanbp', 'wblc', 'hrt', 'resp', 'temp', 'pafi', 'alb', 'bili', 'crea', 'sod', 'ph', 'glucose', 'bun', 'urine'],
        'S_view_cols': ['sex', 'income', 'edu', 'adlp_s'] + [c for c in processed_df.columns if 'race_' in c or 'dnr_' in c],
        'Y_validation_cols': ['death', 'hospdead', 'd.time', 'slos', 'hday', 'sfdm2', 'surv6m', 'prg6m', 'dnrday', 'totmcst']
    }
    
    # Create View DataFrames
    C_view = processed_df[schema['C_view_cols']]
    P_view = processed_df[schema['P_view_cols']]
    S_view = processed_df[schema['S_view_cols']]
    
    # Scale P-view
    scaler = StandardScaler()
    P_view_scaled = pd.DataFrame(scaler.fit_transform(P_view), columns=schema['P_view_cols'])
    
    # Isolate Validation set
    Y_validation = processed_df[[c for c in schema['Y_validation_cols'] if c in processed_df.columns]]

    if save_output:
        print(f"\nSaving processed views to {PROCESSED_PATH}...")
        PROCESSED_PATH.mkdir(exist_ok=True)
        C_view.to_csv(PROCESSED_PATH / 'C_view.csv', index=False)
        P_view_scaled.to_csv(PROCESSED_PATH / 'P_view_scaled.csv', index=False)
        S_view.to_csv(PROCESSED_PATH / 'S_view.csv', index=False)
        Y_validation.to_csv(PROCESSED_PATH / 'Y_validation.csv', index=False)
        processed_df.to_csv(PROCESSED_PATH / 'support_preprocessed.csv', index=False)
        print("Files saved successfully.")

    return processed_df

def main():
    parser = argparse.ArgumentParser(description="Run the SUPPORT-II preprocessing pipeline.")
    parser.add_argument(
        '--no-save', 
        action='store_true',
        help="Run the script without saving the output files (dry run)."
    )
    args = parser.parse_args()
    
    run_preprocessing(save_output=not args.no_save)

if __name__ == "__main__":
    main()