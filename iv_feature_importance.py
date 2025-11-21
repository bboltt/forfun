import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer

# --- Custom Imports ---
from optimal_spark_config.create_spark_instance import generate_spark_instance

# ================= CONFIGURATION =================
RUN_ON_FULL_DATA = True
SAMPLE_SIZE = 200000      # 200k is sufficient for stable IV and RF Importance
RF_N_SEEDS = 5            # Train 5 times to average out random noise
RF_N_ESTIMATORS = 50      # Small number of trees for speed (dummy model)
IV_BIN_MAX = 10           # Max bins for continuous variables

# Paths
HIVE_TABLE = 'dm_fraud_detection.cs_training_data_v2'
LOCAL_FILENAME = 'eda_dataframe_1.csv'
DATA_FOLDER = 'development_work/customer_score_v2/eda_data/'
SAVE_FOLDER = 'development_work/customer_score_v2/misc_figures/'
IV_FOLDER = os.path.join(SAVE_FOLDER, 'feature_importance/')

# Columns
FRAUD_COL = 'fraud_label_180'
IGNORE_COLS = ['applicationdate', 'customerssn_obfuscated', 'month_year', 'booked', FRAUD_COL]

def get_clean_sample(spark):
    """
    Fetches a robust sample for analysis.
    """
    if RUN_ON_FULL_DATA:
        print(f"--- MODE: FULL DATA (Hive: {HIVE_TABLE}) ---")
        print(f"Sampling ~{SAMPLE_SIZE} rows...")
        df = spark.table(HIVE_TABLE).sample(False, 0.05, seed=42).limit(SAMPLE_SIZE)
    else:
        print(f"--- MODE: LOCAL SAMPLE ({LOCAL_FILENAME}) ---")
        full_path = os.path.join(DATA_FOLDER, LOCAL_FILENAME)
        if not os.path.exists(full_path):
             spark.table(HIVE_TABLE).sample(0.01).toPandas().to_csv(full_path, index=False)
        df = spark.read.option("header", "true").option("inferSchema", "true").csv(full_path)

    print("Collecting sample to Driver (Pandas)...")
    pdf = df.toPandas()
    
    # Ensure fraud col is filled
    if FRAUD_COL in pdf.columns:
        pdf[FRAUD_COL] = pdf[FRAUD_COL].fillna(0).astype(int)
    
    return pdf

def calculate_iv(df, feature, target):
    """
    Calculates Information Value (IV) and Weight of Evidence (WoE) for a feature.
    Handles both Categorical (automatic) and Numeric (binning) features.
    """
    lst = []
    
    # 1. Check if numeric or categorical
    if np.issubdtype(df[feature].dtype, np.number) and df[feature].nunique() > IV_BIN_MAX:
        # Binning for continuous variables
        try:
            # qcut tries to create equal-sized buckets
            bins = pd.qcut(df[feature], IV_BIN_MAX, duplicates='drop')
        except:
            # Fallback to simple cut if qcut fails (e.g., mostly zeros)
            bins = pd.cut(df[feature], IV_BIN_MAX)
    else:
        # Categorical or low-cardinality numeric
        bins = df[feature].fillna("Missing")

    # 2. Group by Bins and Calculate Counts
    # Crosstab gives us the count of Non-Fraud (0) and Fraud (1) per bin
    ct = pd.crosstab(bins, df[target], dropna=False)
    
    # Add missing columns if a bin has pure fraud or pure non-fraud
    if 0 not in ct.columns: ct[0] = 0
    if 1 not in ct.columns: ct[1] = 0
    
    # 3. Calculate WoE and IV
    # Add small epsilon to avoid division by zero
    epsilon = 1e-6
    
    total_good = ct[0].sum() + epsilon
    total_bad = ct[1].sum() + epsilon
    
    ct['dist_good'] = (ct[0] + epsilon) / total_good
    ct['dist_bad'] = (ct[1] + epsilon) / total_bad
    
    ct['woe'] = np.log(ct['dist_good'] / ct['dist_bad'])
    ct['iv'] = (ct['dist_good'] - ct['dist_bad']) * ct['woe']
    
    return ct['iv'].sum()

def run_iv_analysis(df, feature_cols):
    print(f"\n--- Calculating Information Value (IV) for {len(feature_cols)} features ---")
    iv_stats = []
    
    for i, feat in enumerate(feature_cols):
        if i % 10 == 0: print(f"Processing IV for feature {i}/{len(feature_cols)}: {feat}")
        try:
            iv = calculate_iv(df, feat, FRAUD_COL)
            iv_stats.append({'feature': feat, 'iv': iv})
        except Exception as e:
            print(f"Error calculating IV for {feat}: {e}")
            iv_stats.append({'feature': feat, 'iv': 0.0})
            
    return pd.DataFrame(iv_stats).sort_values('iv', ascending=False)

def run_rf_importance_analysis(df, feature_cols):
    print(f"\n--- Training Dummy Random Forest ({RF_N_SEEDS} seeds) ---")
    
    # 1. Prepare Data for Sklearn
    X = df[feature_cols].copy()
    y = df[FRAUD_COL]
    
    # A. Handle Categoricals (Ordinal Encode)
    # RF handles non-linearities well, so ordinal encoding is usually fine/better than OneHot for high cardinality
    cat_cols = X.select_dtypes(exclude=[np.number]).columns
    if len(cat_cols) > 0:
        enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        # Fill NA before encoding
        X[cat_cols] = X[cat_cols].fillna("Missing")
        X[cat_cols] = enc.fit_transform(X[cat_cols])

    # B. Impute Numeric NAs (Median)
    num_cols = X.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        imp = SimpleImputer(strategy='median')
        X[num_cols] = imp.fit_transform(X[num_cols])
        
    # 2. Loop Over Seeds
    importance_df = pd.DataFrame({'feature': feature_cols})
    
    for i in range(RF_N_SEEDS):
        print(f"Training RF Seed {i+1}/{RF_N_SEEDS}...")
        
        rf = RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS,
            max_depth=10,           # Limit depth to prevent total overfitting on dummy model
            random_state=i,         # Change seed
            n_jobs=-1,              # Use all cores
            class_weight='balanced' # Handle fraud imbalance
        )
        rf.fit(X, y)
        
        # Store importance
        importance_df[f'imp_seed_{i}'] = rf.feature_importances_

    # 3. Average Results
    # Calculate Mean and Std Dev to see stability
    seed_cols = [c for c in importance_df.columns if 'imp_seed_' in c]
    importance_df['rf_importance_mean'] = importance_df[seed_cols].mean(axis=1)
    importance_df['rf_importance_std'] = importance_df[seed_cols].std(axis=1)
    
    # Sort by Mean Importance
    return importance_df[['feature', 'rf_importance_mean', 'rf_importance_std']].sort_values('rf_importance_mean', ascending=False)

def plot_importance_comparison(iv_df, rf_df, save_folder):
    """
    Creates a plot comparing IV vs RF Importance to see if they agree.
    """
    print("Generating Comparison Plots...")
    
    # Merge
    merged = pd.merge(iv_df, rf_df, on='feature')
    
    # Normalize both to 0-1 range for visualization comparison
    merged['iv_norm'] = merged['iv'] / merged['iv'].max()
    merged['rf_norm'] = merged['rf_importance_mean'] / merged['rf_importance_mean'].max()
    
    # Sort by RF Importance
    top_30 = merged.head(30)
    
    plt.figure(figsize=(12, 8))
    
    # Plot RF Importance
    sns.barplot(data=top_30, x='rf_importance_mean', y='feature', color='skyblue', label='RF Importance (Mean)')
    
    # Add Error Bars for RF Stability
    plt.errorbar(x=top_30['rf_importance_mean'], y=range(len(top_30)), 
                 xerr=top_30['rf_importance_std'], fmt='none', c='black', capsize=3)

    plt.title(f"Top 30 Features by Random Forest Importance\n(Averaged over {RF_N_SEEDS} seeds)")
    plt.xlabel("Gini Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, 'rf_feature_importance.jpg'))
    plt.close()
    
    # Scatter Plot: IV vs RF
    plt.figure(figsize=(8, 8))
    sns.scatterplot(data=merged, x='iv', y='rf_importance_mean', alpha=0.6)
    plt.title("Feature Importance Comparison: IV vs Random Forest")
    plt.xlabel("Information Value (IV)")
    plt.ylabel("RF Importance (Gini)")
    
    # Label top features
    for i in range(min(10, len(merged))):
        row = merged.iloc[i]
        plt.text(row['iv'], row['rf_importance_mean'], row['feature'], fontsize=8)
        
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, 'iv_vs_rf_scatter.jpg'))
    plt.close()

def analyze_importance():
    spark = generate_spark_instance(
        total_memory=100,
        total_vcpu=50,
        python37=True,
        appName='originations_feature_importance'
    )
    
    # 1. Get Data
    df = get_clean_sample(spark)
    feature_cols = [c for c in df.columns if c not in IGNORE_COLS]
    
    # 2. Setup Output
    os.makedirs(IV_FOLDER, exist_ok=True)
    
    # 3. Run IV Analysis
    iv_results = run_iv_analysis(df, feature_cols)
    iv_results.to_csv(os.path.join(IV_FOLDER, 'information_values.csv'), index=False)
    
    # 4. Run Random Forest Analysis (Multi-seed)
    rf_results = run_rf_importance_analysis(df, feature_cols)
    rf_results.to_csv(os.path.join(IV_FOLDER, 'rf_feature_importance.csv'), index=False)
    
    # 5. Visualize
    plot_importance_comparison(iv_results, rf_results, IV_FOLDER)
    
    print(f"\nAnalysis Complete. Results saved to {IV_FOLDER}")

if __name__ == "__main__":
    analyze_importance()
