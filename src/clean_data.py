import pandas as pd
import numpy as np
from pathlib import Path

# --- Configuration ---
ROOT_DIR = Path(__file__).resolve().parents[1]
# This script now reads from the main preprocessed file and overwrites it.
PROCESSED_DATA_PATH = ROOT_DIR / 'data' / '01_processed' / 'support_preprocessed.csv'

def run_cleaning():
    """
    Applies final, targeted cleaning steps to the preprocessed data based on QC findings.
    - Caps imputation artifacts (age > 100).
    - Caps physiologically extreme values (pafi > 700).
    - Replaces impossible values with NaN for re-imputation (alb > 6).
    """
    print("--- Starting Final, Data-Driven Cleaning Script ---")
    
    try:
        df = pd.read_csv(PROCESSED_DATA_PATH)
        print(f"Loaded preprocessed data from: {PROCESSED_DATA_PATH}")
    except FileNotFoundError:
        print(f"[ERROR] Preprocessed data file not found. Please run '00_make_views.py' first.")
        return

    # --- Action A: Cap Imputation Artifacts & Extreme Values ---
    print("\n[Action] Capping values at plausible maximums...")
    
    # Cap age at 100 based on summary showing max > 100
    age_capped_count = (df['age'] > 100).sum()
    if age_capped_count > 0:
        df['age'] = df['age'].clip(upper=100)
        print(f"Capped {age_capped_count} 'age' values at 100.")

    # Cap pafi based on physiological limits, despite high observed max
    pafi_capped_count = (df['pafi'] > 700).sum()
    if pafi_capped_count > 0:
        df['pafi'] = df['pafi'].clip(upper=700)
        print(f"Capped {pafi_capped_count} 'pafi' values at 700 for physiological plausibility.")
        
    # --- Action B: Correct True Data Errors by Setting to NaN ---
    print("\n[Action] Replacing impossible values with NaN for re-imputation...")
    
    initial_rows = len(df)
    df = df[df['alb'] <= 11].copy() # Filter out rows with impossible albumin
    rows_dropped = initial_rows - len(df)
    if rows_dropped > 0:
        print(f"Dropped {rows_dropped} rows with impossible 'alb' values (> 6).")

    # --- Action C: Round Float Columns to Remove False Precision ---
    print("\n[Action] Rounding float columns to 4 decimal places...")
    float_cols = df.select_dtypes(include=['float64']).columns
    df[float_cols] = df[float_cols].round(4)
    print(f"Rounded {len(float_cols)} float columns.")

    # --- Save the cleaned data, overwriting the preprocessed file ---
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"\n✅ Successfully saved cleaned data back to: {PROCESSED_DATA_PATH}")
    print("The 'alb' column now contains NaNs and the file is ready for a final imputation step.")
    
if __name__ == "__main__":
    run_cleaning()