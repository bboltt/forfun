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
SAMPLE_SIZE = 200000      # 200k is sufficient
RF_N_SEEDS = 5            # Train 5 times
RF_N_ESTIMATORS = 50      # Small number of trees
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
    
    if FRAUD_COL in pdf.columns:
        pdf[FRAUD_COL] = pdf[FRAUD_COL].fillna(0).astype(int)
    
    return pdf

def calculate_iv(df, feature, target):
    if np.issubdtype(df[feature].dtype, np.number) and df[feature].nunique() > IV_BIN_MAX:
        try:
            bins = pd.qcut(df[feature], IV_BIN_MAX, duplicates='drop')
        except:
            bins = pd.cut(df[feature], IV_BIN_MAX)
    else:
        bins = df[feature].fillna("Missing")

    ct = pd.crosstab(bins, df[target], dropna=False)
    
    if 0 not in ct.columns: ct[0] = 0
    if 1 not in ct.columns: ct[1] = 0
    
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
    
    X = df[feature_cols].copy()
    y = df[FRAUD_COL]
    
    # A. Handle Categoricals (Ordinal Encode)
    cat_cols = X.select_dtypes(exclude=[np.number]).columns
    if len(cat_cols) > 0:
        # --- FIX IS HERE: Force conversion to string to handle Date/Object types ---
        print("Encoding categorical features...")
        X[cat_cols] = X[cat_cols].astype(str)
        
        enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        # Fill NA with string "Missing" before encoding
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
            max_depth=10,
            random_state=i,
            n_jobs=-1,
            class_weight='balanced'
        )
        rf.fit(X, y)
        importance_df[f'imp_seed_{i}'] = rf.feature_importances_

    seed_cols = [c for c in importance_df.columns if 'imp_seed_' in c]
    importance_df['rf_importance_mean'] = importance_df[seed_cols].mean(axis=1)
    importance_df['rf_importance_std'] = importance_df[seed_cols].std(axis=1)
    
    return importance_df[['feature', 'rf_importance_mean', 'rf_importance_std']].sort_values('rf_importance_mean', ascending=False)

def plot_importance_comparison(iv_df, rf_df, save_folder):
    print("Generating Comparison Plots...")
    merged = pd.merge(iv_df, rf_df, on='feature')
    merged['iv_norm'] = merged['iv'] / merged['iv'].max()
    merged['rf_norm'] = merged['rf_importance_mean'] / merged['rf_importance_mean'].max()
    
    top_30 = merged.head(30)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(data=top_30, x='rf_importance_mean', y='feature', color='skyblue')
    plt.errorbar(x=top_30['rf_importance_mean'], y=range(len(top_30)), 
                 xerr=top_30['rf_importance_std'], fmt='none', c='black', capsize=3)

    plt.title(f"Top 30 Features by Random Forest Importance\n(Averaged over {RF_N_SEEDS} seeds)")
    plt.xlabel("Gini Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, 'rf_feature_importance.jpg'))
    plt.close()
    
    plt.figure(figsize=(8, 8))
    sns.scatterplot(data=merged, x='iv', y='rf_importance_mean', alpha=0.6)
    plt.title("Feature Importance Comparison: IV vs Random Forest")
    plt.xlabel("Information Value (IV)")
    plt.ylabel("RF Importance (Gini)")
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
    
    df = get_clean_sample(spark)
    feature_cols = [c for c in df.columns if c not in IGNORE_COLS]
    
    os.makedirs(IV_FOLDER, exist_ok=True)
    
    iv_results = run_iv_analysis(df, feature_cols)
    iv_results.to_csv(os.path.join(IV_FOLDER, 'information_values.csv'), index=False)
    
    rf_results = run_rf_importance_analysis(df, feature_cols)
    rf_results.to_csv(os.path.join(IV_FOLDER, 'rf_feature_importance.csv'), index=False)
    
    plot_importance_comparison(iv_results, rf_results, IV_FOLDER)
    print(f"\nAnalysis Complete. Results saved to {IV_FOLDER}")

if __name__ == "__main__":
    analyze_importance()
