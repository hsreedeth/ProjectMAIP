import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --- Configuration ---
ROOT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DATA_PATH = ROOT_DIR / 'data' / '01_processed' / 'support_preprocessed.csv'
QC_FIGURE_PATH = ROOT_DIR / 'reports' / 'figures' / 'qc'

# =============================================================================
# QC CONFIGURATION DICTIONARY
# Define all QC rules here. This makes them easy to manage and modify.
# =============================================================================
QC_CONFIG = {
    # --- Demographics ---
    'age': { 
        'plausible_range': [18, 100], # Data-driven: min is 18.04, max is 100.84. Capping at 100 is still sensible.
        'check_iqr_outliers': True 
    },
    # --- Physiology ---
    'meanbp': { 'plausible_range': [0, 250], 'check_iqr_outliers': True }, # Data-driven: max is 195. Buffer is reasonable.
    'wblc': { 
        'plausible_range': [0, 201], # Data-driven: max is 200. Allow a small buffer.
        'check_iqr_outliers': True 
    },
    'hrt': { 
        'plausible_range': [0, 301], # Data-driven: max is 300. Allow a small buffer.
        'check_iqr_outliers': True 
    },
    'resp': { 'plausible_range': [0, 100], 'check_iqr_outliers': True }, # Data-driven: max is 90.
    'temp': { 'plausible_range': [30, 45], 'check_iqr_outliers': True }, # Data-driven: min/max are 32.9/41.7. Original range is fine.
    'pafi': { 
        'plausible_range': [0, 700], # Data-driven: max is 890, but this is physiologically extreme. Capping at a more realistic max is safer.
        'check_iqr_outliers': True 
    },
    'alb': { 
        'plausible_range': [0, 6],   # Data-driven: max is 29, which confirms a data error. A clinical max of 6 is appropriate.
        'check_iqr_outliers': True 
    },
    'bili': { 
        'plausible_range': [0, 65],  # Data-driven: max is 63.
        'check_iqr_outliers': True 
    },
    'crea': { 
        'plausible_range': [0, 22],  # Data-driven: max is 21.5.
        'check_iqr_outliers': True 
    },
    'sod': { 
        'plausible_range': [100, 182], # Data-driven: max is 181.
        'check_iqr_outliers': True 
    },
    'ph': { 'plausible_range': [6.8, 7.8], 'check_iqr_outliers': True }, # Data-driven: min/max are 7.0/7.77. Original clinical range is fine.
    'bun': { 
        'plausible_range': [0, 301], # Data-driven: max is 300.
        'check_iqr_outliers': True 
    },
    'urine': { 'plausible_range': [0, 15000], 'check_iqr_outliers': True }, # Data-driven: max is 9000.
    # --- Scores ---
    'sps': { 'check_iqr_outliers': True },
    'aps': { 'check_iqr_outliers': True },
    'scoma': { 'check_iqr_outliers': True },
    'adlp_s': { 'check_iqr_outliers': True },
    # --- Outcomes ---
    'totmcst': { 'plausible_range': [0, None], 'check_iqr_outliers': True },
    'slos': { 'plausible_range': [0, None], 'check_iqr_outliers': True }
}

# --- QC Functions ---

def check_plausible_ranges(df):
    """Checks for values outside the hard-coded plausible ranges defined in QC_CONFIG."""
    print("\n--- 1. Checking for Biologically/Logically Implausible Values ---")
    issues_found = 0
    for col, rules in QC_CONFIG.items():
        if 'plausible_range' in rules and col in df.columns:
            min_val, max_val = rules['plausible_range']
            
            # Create boolean series for violations
            violations = pd.Series(False, index=df.index)
            if min_val is not None:
                violations |= (df[col] < min_val)
            if max_val is not None:
                violations |= (df[col] > max_val)

            if violations.any():
                issues_found += 1
                print(f"\n[ISSUE] Found {violations.sum()} implausible values in '{col}':")
                print(df.loc[violations, [col]])
    
    if issues_found == 0:
        print("✅ No implausible values found based on defined ranges.")
    return issues_found

def find_iqr_outliers(df):
    """Identifies and reports statistical outliers using the 1.5*IQR method."""
    print("\n--- 2. Checking for Statistical Outliers (1.5 * IQR Rule) ---")
    outlier_vars = []
    for col, rules in QC_CONFIG.items():
        if rules.get('check_iqr_outliers', False) and col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            
            if not outliers.empty:
                outlier_vars.append(col)
                print(f"\n[INFO] Found {len(outliers)} statistical outliers in '{col}' (Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]).")
                # Show 5 most extreme outliers
                print("Extreme examples:")
                print(outliers.sort_values(by=col, ascending=False)[[col]].head().to_string())

    if not outlier_vars:
        print("✅ No statistical outliers found in checked variables.")
    return len(outlier_vars)

def generate_visualizations(df):
    """Generates and saves a histogram and box plot for each numeric variable in the config."""
    print(f"\n--- 3. Generating Visualizations ---")
    QC_FIGURE_PATH.mkdir(parents=True, exist_ok=True)
    
    numeric_cols_to_plot = [
        col for col, rules in QC_CONFIG.items() 
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col])
    ]
    
    for col in numeric_cols_to_plot:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Histogram
        sns.histplot(df[col], kde=True, ax=axes[0])
        axes[0].set_title(f'Histogram of {col}')
        
        # Box plot
        sns.boxplot(x=df[col], ax=axes[1])
        axes[1].set_title(f'Box Plot of {col}')
        
        plt.suptitle(f'QC Plots for {col}', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save the figure
        fig_path = QC_FIGURE_PATH / f'{col}_qc_plot.png'
        plt.savefig(fig_path)
        plt.close(fig) # Close the plot to save memory
        
    print(f"✅ Saved {len(numeric_cols_to_plot)} QC plots to: {QC_FIGURE_PATH}")

def main():
    """Main function to run the QC pipeline."""
    parser = argparse.ArgumentParser(description="Run a Quality Control check on the preprocessed SUPPORT-II data.")
    parser.add_argument(
        '--visualize', 
        action='store_true',
        help="Generate and save histograms and box plots for all QC'd variables."
    )
    args = parser.parse_args()

    print("="*80)
    print("Starting Data Quality Control (QC) Script")
    print("="*80)
    
    try:
        df = pd.read_csv(PROCESSED_DATA_PATH)
        print(f"\nSuccessfully loaded preprocessed data from: {PROCESSED_DATA_PATH}")
    except FileNotFoundError:
        print(f"\n[ERROR] Preprocessed data file not found. Please run the preprocessing script first.")
        return

    check_plausible_ranges(df)
    find_iqr_outliers(df)
    
    if args.visualize:
        generate_visualizations(df)
        
    print("\n" + "="*80)
    print("QC Script Finished")
    print("="*80)

if __name__ == "__main__":
    main()