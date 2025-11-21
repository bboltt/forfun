import os
import pandas as pd
import numpy as np
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, DoubleType, FloatType, LongType, DecimalType

# --- Custom Imports ---
from optimal_spark_config.create_spark_instance import generate_spark_instance

# ================= CONFIGURATION =================
RUN_ON_FULL_DATA = True

# Paths
HIVE_TABLE = 'dm_fraud_detection.cs_training_data_v2'
LOCAL_FILENAME = 'eda_dataframe_1.csv'
DATA_FOLDER = 'development_work/customer_score_v2/eda_data/'
SAVE_FOLDER = 'development_work/customer_score_v2/misc_figures/'

# Columns to Exclude from stats (e.g., IDs, Dates)
IGNORE_COLS = ['applicationdate', 'customerssn_obfuscated', 'month_year']

def get_dataframe(spark):
    """Load Data (Hive or Local Sample)"""
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
    return df

def analyze_numeric_features(df, numeric_cols):
    """
    Calculates Count, Mean, Std, Min, Max, and Percentiles (1%, 10%, 25%, 50%, 75%, 90%, 99%).
    Optimized for Big Data using PySpark aggregation and approxQuantile.
    """
    print(f"\n--- Analyzing {len(numeric_cols)} Numeric Features ---")
    
    # 1. Basic Stats (Count, Mean, Std, Min, Max)
    # We build one giant aggregation query to run in a single pass
    exprs = []
    for c in numeric_cols:
        exprs.extend([
            F.count(c).alias(f"{c}_count"),
            F.mean(c).alias(f"{c}_mean"),
            F.stddev(c).alias(f"{c}_std"),
            F.min(c).alias(f"{c}_min"),
            F.max(c).alias(f"{c}_max")
        ])
    
    print("Calculating basic stats (Mean, Std, Min, Max)...")
    basic_stats_row = df.agg(*exprs).collect()[0]
    
    # 2. Percentiles (ApproxQuantile is much faster for Big Data)
    # We calculate these separately or in batches if too many columns
    percentiles = [0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99]
    percentile_names = ['p1', 'p10', 'p25', 'p50', 'p75', 'p90', 'p99']
    
    stats_data = []
    
    print("Calculating percentiles (this may take a moment)...")
    for i, col_name in enumerate(numeric_cols):
        if i % 10 == 0: print(f"Processing column {i}/{len(numeric_cols)}...")
        
        # Extract Basic Stats from the aggregated row
        row_dict = {
            'feature': col_name,
            'count': basic_stats_row[f"{col_name}_count"],
            'mean': basic_stats_row[f"{col_name}_mean"],
            'std': basic_stats_row[f"{col_name}_std"],
            'min': basic_stats_row[f"{col_name}_min"],
            'max': basic_stats_row[f"{col_name}_max"]
        }
        
        # Calculate Percentiles for this column
        # approxQuantile(col, probabilities, relativeError)
        try:
            quantiles = df.stat.approxQuantile(col_name, percentiles, 0.01)
            for p_name, p_val in zip(percentile_names, quantiles):
                row_dict[p_name] = p_val
        except Exception as e:
            print(f"Warning: Could not calc percentiles for {col_name}: {e}")
        
        stats_data.append(row_dict)
        
    return pd.DataFrame(stats_data)

def analyze_categorical_features(df, cat_cols):
    """
    Calculates Count, Null Count, Distinct Count (Cardinality) for categorical features.
    Replicates Pandas 'describe(include=[O])' logic but for Big Data.
    """
    print(f"\n--- Analyzing {len(cat_cols)} Categorical Features ---")
    
    exprs = []
    for c in cat_cols:
        exprs.extend([
            F.count(c).alias(f"{c}_count"),
            F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(f"{c}_nulls"),
            F.countDistinct(c).alias(f"{c}_distinct")
        ])
        
    print("Aggregating categorical stats...")
    cat_row = df.agg(*exprs).collect()[0]
    
    stats_data = []
    for c in cat_cols:
        stats_data.append({
            'feature': c,
            'count': cat_row[f"{c}_count"],
            'null_count': cat_row[f"{c}_nulls"],
            'distinct_count': cat_row[f"{c}_distinct"]
        })
        
    return pd.DataFrame(stats_data)

def generate_summary_stats():
    spark = generate_spark_instance(
        total_memory=100,
        total_vcpu=50,
        python37=True,
        appName='originations_summary_stats'
    )
    
    df = get_dataframe(spark)
    
    # 1. Detect Column Types
    # Spark types that are numeric
    numeric_types = (IntegerType, DoubleType, FloatType, LongType, DecimalType)
    
    all_cols = [c for c in df.columns if c not in IGNORE_COLS]
    numeric_cols = [f.name for f in df.schema.fields if isinstance(f.dataType, numeric_types) and f.name in all_cols]
    cat_cols = [f.name for f in df.schema.fields if not isinstance(f.dataType, numeric_types) and f.name in all_cols]
    
    # 2. Analyze Numeric
    if numeric_cols:
        num_stats = analyze_numeric_features(df, numeric_cols)
        os.makedirs(SAVE_FOLDER, exist_ok=True)
        save_path = os.path.join(SAVE_FOLDER, 'basic_stats_numeric_features.csv')
        num_stats.set_index('feature').to_csv(save_path)
        print(f"Numeric stats saved to: {save_path}")
        
    # 3. Analyze Categorical
    if cat_cols:
        cat_stats = analyze_categorical_features(df, cat_cols)
        os.makedirs(SAVE_FOLDER, exist_ok=True)
        save_path = os.path.join(SAVE_FOLDER, 'basic_stats_categorical_features.csv')
        cat_stats.set_index('feature').to_csv(save_path)
        print(f"Categorical stats saved to: {save_path}")

    print("\nSummary Statistics Generation Complete.")

if __name__ == "__main__":
    generate_summary_stats()
