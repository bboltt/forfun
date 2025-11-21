import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, DoubleType

# --- Custom Imports ---
# Using your existing configuration setup
from optimal_spark_config.create_spark_instance import generate_spark_instance

# ================= CONFIGURATION =================
RUN_ON_FULL_DATA = True

# Paths
HIVE_TABLE = 'dm_fraud_detection.cs_training_data_v2'
LOCAL_FILENAME = 'eda_dataframe_1.csv'
DATA_FOLDER = 'development_work/customer_score_v2/eda_data/'
SAVE_FOLDER = 'development_work/customer_score_v2/misc_figures/'

# Columns
FRAUD_COL = 'fraud_label_180'
# List of columns to IGNORE in analysis (keys, dates, label itself)
IGNORE_COLS = [FRAUD_COL, 'applicationdate', 'customerssn_obfuscated', 'booked', 'month_year']

def get_dataframe(spark):
    """
    Reuse of the robust data loader from previous steps.
    """
    if RUN_ON_FULL_DATA:
        print(f"--- MODE: FULL DATA (Hive: {HIVE_TABLE}) ---")
        df = spark.table(HIVE_TABLE)
    else:
        print(f"--- MODE: LOCAL SAMPLE ({LOCAL_FILENAME}) ---")
        full_path = os.path.join(DATA_FOLDER, LOCAL_FILENAME)
        if not os.path.exists(full_path):
            print("Downloading sample...")
            os.makedirs(DATA_FOLDER, exist_ok=True)
            spark.table(HIVE_TABLE).sample(0.01).toPandas().to_csv(full_path, index=False)
        df = spark.read.option("header", "true").option("inferSchema", "true").csv(full_path)

    # Fill target nulls with 0 just for calculation safety
    df = df.fillna({FRAUD_COL: 0})
    return df

def calculate_missing_stats(df):
    """
    Calculates missing rates and conditional fraud rates using PySpark.
    
    Returns: Pandas DataFrame with index as feature name and columns:
             ['missing_rate', 'fraud_rate_missing', 'fraud_rate_present']
    """
    print("--- Calculating Missing Values & Conditional Fraud Rates (Spark) ---")
    
    # 1. Identify Feature Columns (All numeric/string cols minus ignore list)
    feature_cols = [c for c in df.columns if c not in IGNORE_COLS]
    
    # 2. Build Aggregation Expressions
    # We want to do this in ONE pass over the data for performance.
    # For each column 'c', we calculate:
    #   - count of nulls
    #   - sum of fraud where c is null
    #   - sum of fraud where c is NOT null
    
    exprs = []
    for c in feature_cols:
        # Count Nulls
        exprs.append(F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(f"{c}_null_count"))
        # Sum Fraud when Null
        exprs.append(F.sum(F.when(F.col(c).isNull(), F.col(FRAUD_COL)).otherwise(0)).alias(f"{c}_fraud_null_sum"))
        # Sum Fraud when Present (Not Null)
        exprs.append(F.sum(F.when(F.col(c).isNotNull(), F.col(FRAUD_COL)).otherwise(0)).alias(f"{c}_fraud_present_sum"))

    # Add Total Count
    exprs.append(F.count('*').alias('total_count'))

    # 3. Execute Aggregation
    print(f"Aggregating stats for {len(feature_cols)} features...")
    # NOTE: If 200+ columns cause a DAG overflow, we can batch this loop. 
    # For < 500 columns, a single agg usually works fine in modern Spark.
    results_row = df.agg(*exprs).collect()[0]
    
    # 4. Process Results into Pandas
    total_count = results_row['total_count']
    stats_data = []

    for c in feature_cols:
        null_count = results_row[f"{c}_null_count"] or 0
        fraud_null_sum = results_row[f"{c}_fraud_null_sum"] or 0
        fraud_present_sum = results_row[f"{c}_fraud_present_sum"] or 0
        
        present_count = total_count - null_count
        
        # Metrics
        missing_rate = null_count / total_count
        
        # Avoid division by zero
        fraud_rate_missing = (fraud_null_sum / null_count) if null_count > 0 else np.nan
        fraud_rate_present = (fraud_present_sum / present_count) if present_count > 0 else np.nan
        
        stats_data.append({
            'feature': c,
            'missing_rate': missing_rate,
            'fraud_rate_missing': fraud_rate_missing,
            'fraud_rate_present': fraud_rate_present,
            'null_count': null_count
        })
        
    stats_df = pd.DataFrame(stats_data).set_index('feature')
    
    # Sort by missing rate descending for better visualization
    stats_df = stats_df.sort_values('missing_rate', ascending=False)
    
    return stats_df

def plot_missing_analysis(stats_df, save_folder):
    """
    Generates the heatmap visualization matching your original style.
    """
    print("--- Generating Missing Value Plots ---")
    os.makedirs(save_folder, exist_ok=True)
    
    # Save CSV first
    csv_path = os.path.join(save_folder, 'missing_rates.csv')
    stats_df.to_csv(csv_path)
    print(f"Detailed stats saved to {csv_path}")

    # Filter: Only show features that have AT LEAST some missing values (or > 0.1%)
    # to keep the chart readable.
    plot_df = stats_df[stats_df['missing_rate'] > 0.001].copy()
    
    if plot_df.empty:
        print("No features found with significant missing values (> 0.1%). Skipping plot.")
        return

    # Setup Figure: 2 subplots (Left: Missing Rate, Right: Fraud Rates)
    # Dynamic height based on number of features
    height = max(8, len(plot_df) * 0.25) 
    fig, ax = plt.subplots(ncols=2, figsize=(10, height), sharey=True)

    # Plot 1: Missing Rate Heatmap
    sns.heatmap(
        plot_df[['missing_rate']],
        ax=ax[0],
        cmap="Blues",
        linewidths=0.5,
        annot=True,
        fmt='.1%',
        cbar=False,
        vmin=0, vmax=1
    )
    ax[0].set_title("Missing Rate")
    
    # Plot 2: Fraud Rate Comparison (Missing vs Present)
    # We prepare a specific DF for this heatmap
    fraud_comparison = plot_df[['fraud_rate_missing', 'fraud_rate_present']]
    
    sns.heatmap(
        fraud_comparison,
        ax=ax[1],
        cmap="Reds",
        linewidths=0.5,
        annot=True,
        fmt='.2%',
        cbar=True
    )
    ax[1].set_title("Fraud Rate Impact")

    plt.suptitle("Missing Value Analysis & Effect on Fraud", y=1.02)
    plt.tight_layout()
    
    # Save
    plot_path = os.path.join(save_folder, 'missing_rates.jpg')
    plt.savefig(plot_path, bbox_inches='tight')
    print(f"Plot saved to {plot_path}")
    plt.close()

def analyze_missing_values():
    spark = generate_spark_instance(
        total_memory=100,
        total_vcpu=50,
        python37=True,
        appName='originations_missing_val_analysis'
    )
    
    # 1. Load Data
    df = get_dataframe(spark)
    
    # 2. Calculate Stats (Spark Optimized)
    stats_df = calculate_missing_stats(df)
    
    # 3. Visualize and Save
    plot_missing_analysis(stats_df, SAVE_FOLDER)
    
    print("Missing value analysis complete.")

if __name__ == "__main__":
    analyze_missing_values()
