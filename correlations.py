import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType

# --- Custom Imports ---
from optimal_spark_config.create_spark_instance import generate_spark_instance

# ================= CONFIGURATION =================
RUN_ON_FULL_DATA = True
SAMPLE_SIZE = 200000  # 200k rows is statistically sufficient for stable correlation
CORRELATION_THRESHOLD = 0.7 # Filter for plots

# Paths
HIVE_TABLE = 'dm_fraud_detection.cs_training_data_v2'
LOCAL_FILENAME = 'eda_dataframe_1.csv'
DATA_FOLDER = 'development_work/customer_score_v2/eda_data/'
SAVE_FOLDER = 'development_work/customer_score_v2/misc_figures/'
CORR_FOLDER = os.path.join(SAVE_FOLDER, 'correlations/')

# Columns
FRAUD_COL = 'fraud_label_180'
IGNORE_COLS = ['applicationdate', 'customerssn_obfuscated', 'month_year', 'booked']

def get_clean_sample(spark):
    """
    Fetches a randomized sample of data for correlation analysis.
    Converts all relevant columns to float for matrix calculation.
    """
    if RUN_ON_FULL_DATA:
        print(f"--- MODE: FULL DATA (Hive: {HIVE_TABLE}) ---")
        print(f"Taking a random sample of ~{SAMPLE_SIZE} rows for correlation analysis...")
        
        # Estimate fraction to get ~SAMPLE_SIZE rows. 
        # Assuming ~100M rows, 0.005 is 500k. Adjust based on your actual volume.
        # Here we use limit() after shuffle or just a sample fraction.
        # sample(withReplacement, fraction, seed)
        df = spark.table(HIVE_TABLE).sample(False, 0.05, seed=42).limit(SAMPLE_SIZE)
    else:
        print(f"--- MODE: LOCAL SAMPLE ({LOCAL_FILENAME}) ---")
        full_path = os.path.join(DATA_FOLDER, LOCAL_FILENAME)
        if not os.path.exists(full_path):
             # Fallback if file missing
             spark.table(HIVE_TABLE).sample(0.01).toPandas().to_csv(full_path, index=False)
        df = spark.read.option("header", "true").option("inferSchema", "true").csv(full_path)

    # Convert to Pandas for Matrix math
    print("Collecting sample to Driver (Pandas)...")
    pdf = df.toPandas()
    
    # Preprocessing for Correlation:
    # 1. Drop ignored columns
    cols_to_keep = [c for c in pdf.columns if c not in IGNORE_COLS]
    pdf = pdf[cols_to_keep]
    
    # 2. Force numeric types (coalesce non-numeric to NaN then drop or fill)
    # We only correlate numeric features
    numeric_pdf = pdf.select_dtypes(include=[np.number])
    
    # 3. Fill NaNs for correlation (Correlation can't handle NaNs)
    # Simple fill with median is usually safe for finding *relationships*
    numeric_pdf = numeric_pdf.fillna(numeric_pdf.median())
    
    # Re-attach fraud label if it got dropped (it's numeric usually, but just in case)
    if FRAUD_COL not in numeric_pdf.columns and FRAUD_COL in pdf.columns:
        numeric_pdf[FRAUD_COL] = pdf[FRAUD_COL]

    print(f"Data ready for correlation. Shape: {numeric_pdf.shape}")
    return numeric_pdf

def remove_outliers(df, col, n_std=5):
    """
    Helper to remove outliers > n_std from mean for cleaner plots.
    Returns the series with outliers replaced by NaN, and the count of removed.
    """
    mean = df[col].mean()
    std = df[col].std()
    
    # Logic: Identifies rows outside the boundary
    is_outlier = np.abs(df[col] - mean) > (n_std * std)
    num_outliers = is_outlier.sum()
    
    # Replace with NaN so Seaborn ignores them
    clean_series = df[col].copy()
    clean_series[is_outlier] = np.nan
    
    return clean_series, num_outliers

def analyze_correlations():
    spark = generate_spark_instance(
        total_memory=100,
        total_vcpu=50,
        python37=True,
        appName='originations_correlation_analysis'
    )
    
    # 1. Get Data
    df = get_clean_sample(spark)
    
    # Ensure output directory exists
    os.makedirs(CORR_FOLDER, exist_ok=True)

    # 2. Calculate Matrices
    print("Calculating Pearson Correlation Matrix...")
    pcc = df.corr(method='pearson')
    pcc.to_csv(os.path.join(SAVE_FOLDER, 'correlation_all_features_pearson.csv'))
    
    print("Calculating Spearman Correlation Matrix (Rank-based)...")
    scc = df.corr(method='spearman')
    scc.to_csv(os.path.join(SAVE_FOLDER, 'correlation_all_features_spearman.csv'))

    # 3. Identify High Correlations
    print(f"Identifying pairs with correlation > {CORRELATION_THRESHOLD}...")
    
    # Get pairs to plot
    # We use the upper triangle mask to avoid duplicates (A-B and B-A) and self-correlation (A-A)
    mask = np.triu(np.ones_like(pcc, dtype=bool), k=1)
    
    # Stack matrices to get list of pairs
    pcc_stacked = pcc.where(mask).stack().reset_index()
    pcc_stacked.columns = ['feat1', 'feat2', 'pearson']
    
    scc_stacked = scc.where(mask).stack().reset_index()
    scc_stacked.columns = ['feat1', 'feat2', 'spearman']
    
    # Merge to check both criteria
    pairs = pd.merge(pcc_stacked, scc_stacked, on=['feat1', 'feat2'])
    
    # Filter
    high_corr_pairs = pairs[
        (np.abs(pairs['pearson']) > CORRELATION_THRESHOLD) | 
        (np.abs(pairs['spearman']) > CORRELATION_THRESHOLD)
    ]
    
    # Save the list of high correlations
    high_corr_pairs.to_csv(os.path.join(SAVE_FOLDER, 'correlation_high_pairs.csv'), index=False)
    print(f"Found {len(high_corr_pairs)} highly correlated pairs.")

    # 4. Generate Detailed Plots
    print("Generating scatter plots for high correlations...")
    
    # We need the customer ID for the 'only_cust_with_acct' logic from your original code,
    # but since we already sampled, we will just plot the sample data we have.
    
    plot_data = df.copy()
    
    for idx, row in high_corr_pairs.iterrows():
        f1 = row['feat1']
        f2 = row['feat2']
        p_val = row['pearson']
        s_val = row['spearman']
        
        # Skip if one of the features is the Label itself (unless you want to see it)
        if f1 == FRAUD_COL or f2 == FRAUD_COL:
            continue

        print(f"Plotting {f1} vs {f2}...")
        
        # Clean Outliers for Plotting (Replicating your logic)
        clean_f1, outliers_f1 = remove_outliers(plot_data, f1, n_std=5)
        clean_f2, outliers_f2 = remove_outliers(plot_data, f2, n_std=5)
        
        # Create temp DF for plotting
        temp_plot_df = pd.DataFrame({
            f1: clean_f1,
            f2: clean_f2,
            FRAUD_COL: plot_data[FRAUD_COL]
        }).dropna() # Drop rows where we removed outliers
        
        # Downsample for scatter if sample is still huge (e.g. > 10k points makes scatter slow)
        if len(temp_plot_df) > 10000:
            temp_plot_df = temp_plot_df.sample(10000, random_state=42)

        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.scatterplot(
            data=temp_plot_df,
            x=f1,
            y=f2,
            hue=FRAUD_COL,
            alpha=0.6,
            s=15, # size of dots
            ax=ax
        )
        
        plt.title(
            f"Correlation: {f1} vs {f2}\n"
            f"Pearson: {p_val:.3f}, Spearman: {s_val:.3f}\n"
            f"Outliers Removed (>5std): {f1}={outliers_f1}, {f2}={outliers_f2}"
        )
        
        # Save
        filename = f"{f1}-vs-{f2}.jpg".replace(" ", "_")
        plt.savefig(os.path.join(CORR_FOLDER, filename))
        plt.close()

    print("Correlation Analysis Complete.")

if __name__ == "__main__":
    analyze_correlations()
