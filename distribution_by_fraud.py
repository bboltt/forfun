import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import functions as F

# --- Custom Imports ---
from optimal_spark_config.create_spark_instance import generate_spark_instance

# ================= CONFIGURATION =================
RUN_ON_FULL_DATA = True
SAMPLE_SIZE = 200000  # 200k rows is plenty for distribution visualization

# Paths
HIVE_TABLE = 'dm_fraud_detection.cs_training_data_v2'
LOCAL_FILENAME = 'eda_dataframe_1.csv'
DATA_FOLDER = 'development_work/customer_score_v2/eda_data/'
SAVE_FOLDER = 'development_work/customer_score_v2/misc_figures/'
DIST_FOLDER = os.path.join(SAVE_FOLDER, 'distribution_by_fraud/')

# Columns
FRAUD_COL = 'fraud_label_180'
IGNORE_COLS = ['applicationdate', 'customerssn_obfuscated', 'month_year', 'booked', FRAUD_COL]

def get_clean_sample(spark):
    """
    Fetches a randomized sample for visualization.
    Same logic as the correlation script to ensure consistency.
    """
    if RUN_ON_FULL_DATA:
        print(f"--- MODE: FULL DATA (Hive: {HIVE_TABLE}) ---")
        print(f"Sampling ~{SAMPLE_SIZE} rows...")
        # Approximate fraction to get SAMPLE_SIZE. Adjust 0.05 if dataset is massive.
        df = spark.table(HIVE_TABLE).sample(False, 0.05, seed=42).limit(SAMPLE_SIZE)
    else:
        print(f"--- MODE: LOCAL SAMPLE ({LOCAL_FILENAME}) ---")
        full_path = os.path.join(DATA_FOLDER, LOCAL_FILENAME)
        if not os.path.exists(full_path):
             spark.table(HIVE_TABLE).sample(0.01).toPandas().to_csv(full_path, index=False)
        df = spark.read.option("header", "true").option("inferSchema", "true").csv(full_path)

    print("Collecting sample to Driver (Pandas)...")
    pdf = df.toPandas()
    
    # Ensure fraud col is present and filled (0 if null)
    if FRAUD_COL in pdf.columns:
        pdf[FRAUD_COL] = pdf[FRAUD_COL].fillna(0).astype(int)
    
    return pdf

def plot_numeric_feature(df, col, save_path):
    """
    Plots density histograms for Fraud vs Non-Fraud.
    - Clips outliers (1st and 99th percentile) for readability.
    - Uses density normalization to handle class imbalance.
    """
    plt.figure(figsize=(10, 6))
    
    # 1. Outlier Clipping (Local to this plot)
    # We clip to P1 and P99 to avoid extreme values squashing the plot
    lower = df[col].quantile(0.01)
    upper = df[col].quantile(0.99)
    clipped_data = df[(df[col] >= lower) & (df[col] <= upper)]
    
    if clipped_data.empty:
        print(f"Skipping {col} (empty after clipping)")
        plt.close()
        return

    # 2. Plotting
    # common_norm=False is CRITICAL here. It scales both histograms to have area=1.
    # Without this, the 'Fraud' histogram would be invisible due to low volume.
    sns.histplot(
        data=clipped_data,
        x=col,
        hue=FRAUD_COL,
        stat="density",      # Normalized density
        common_norm=False,   # Normalize each group independently
        element="step",      # Step outline is cleaner for overlapping
        kde=True,            # Add smooth density line
        palette={0: "skyblue", 1: "red"},
        alpha=0.4
    )
    
    # 3. Add Stats to Title
    mean_0 = df[df[FRAUD_COL] == 0][col].mean()
    mean_1 = df[df[FRAUD_COL] == 1][col].mean()
    
    plt.title(f"Distribution of {col} by Fraud Status\n(Outliers clipped 1%-99%)")
    plt.xlabel(f"{col}\nAvg Non-Fraud: {mean_0:.2f} | Avg Fraud: {mean_1:.2f}")
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_categorical_feature(df, col, save_path):
    """
    Plots normalized bar charts for categorical features.
    Limits to Top 20 categories to avoid messy charts.
    """
    # 1. Check Cardinality
    unique_vals = df[col].nunique()
    if unique_vals > 20:
        # Keep only top 20 frequent values
        top_20 = df[col].value_counts().head(20).index
        plot_df = df[df[col].isin(top_20)].copy()
        title_suffix = f"(Top 20 / {unique_vals} Categories)"
    else:
        plot_df = df.copy()
        title_suffix = ""

    plt.figure(figsize=(12, 6))
    
    # 2. Calculate Fraud Rate per Category (Better than just count)
    # We want to see if specific categories have higher fraud risk
    cat_stats = plot_df.groupby(col)[FRAUD_COL].agg(['count', 'mean']).reset_index()
    cat_stats.columns = [col, 'volume', 'fraud_rate']
    cat_stats = cat_stats.sort_values('volume', ascending=False)

    # Plot: We use a dual axis - Volume on Left, Fraud Rate on Right
    ax1 = sns.barplot(data=cat_stats, x=col, y='volume', color='lightgray', alpha=0.6)
    ax1.set_ylabel("Volume", color="gray")
    plt.xticks(rotation=45, ha='right')
    
    ax2 = ax1.twinx()
    sns.pointplot(data=cat_stats, x=col, y='fraud_rate', color='red', ax=ax2, scale=0.7)
    ax2.set_ylabel("Fraud Rate", color="red")
    
    plt.title(f"Volume & Fraud Rate by {col} {title_suffix}")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def analyze_distributions():
    spark = generate_spark_instance(
        total_memory=100,
        total_vcpu=50,
        python37=True,
        appName='originations_dist_analysis'
    )
    
    # 1. Load Sample
    df = get_clean_sample(spark)
    
    # 2. Setup Output
    os.makedirs(DIST_FOLDER, exist_ok=True)
    
    # 3. Identify Feature Types
    all_cols = [c for c in df.columns if c not in IGNORE_COLS]
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c in all_cols]
    
    # Categorical are the rest (strings, etc)
    cat_cols = [c for c in all_cols if c not in numeric_cols]

    print(f"Found {len(numeric_cols)} numeric and {len(cat_cols)} categorical features to plot.")

    # 4. Generate Plots
    # Loop Numeric
    for i, col in enumerate(numeric_cols):
        if i % 10 == 0: print(f"Plotting numeric feature {i}/{len(numeric_cols)}: {col}...")
        save_path = os.path.join(DIST_FOLDER, f"{col}.jpg")
        try:
            plot_numeric_feature(df, col, save_path)
        except Exception as e:
            print(f"Failed to plot {col}: {e}")

    # Loop Categorical
    for i, col in enumerate(cat_cols):
        print(f"Plotting categorical feature {i}/{len(cat_cols)}: {col}...")
        save_path = os.path.join(DIST_FOLDER, f"{col}.jpg")
        try:
            plot_categorical_feature(df, col, save_path)
        except Exception as e:
            print(f"Failed to plot {col}: {e}")

    print(f"Distribution Analysis Complete. Plots saved to {DIST_FOLDER}")

if __name__ == "__main__":
    analyze_distributions()
