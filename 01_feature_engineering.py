# Databricks notebook source
# MAGIC %md
# MAGIC # Feature Engineering with Unity Catalog
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/raw/main/images/product/mlops/advanced/mlops-advanced-1-feature-store.png" width="1200px">
# MAGIC
# MAGIC <!-- Tracking -->
# MAGIC <img width="1px" src="https://www.google-analytics.com/collect?v=1&gtm=GTM-NKQ8TT7&tid=UA-163989034-1&cid=555&aip=1&t=event&ec=field_demos&ea=display&dp=%2F42_field_demos%2Fml%2Fmlops_advanced%2F01_feature_engineering&dt=MLOPS_ADVANCED">
# MAGIC
# MAGIC ## Transform Raw Customer Data into ML Features
# MAGIC
# MAGIC This notebook transforms raw customer data from the bronze layer into ML-ready features.
# MAGIC
# MAGIC **What we'll do:**
# MAGIC 1. Load bronze customer data
# MAGIC 2. Clean and transform features
# MAGIC 3. One-hot encode categorical variables
# MAGIC 4. Create binary churn label
# MAGIC 5. Save to feature table with Change Data Feed
# MAGIC 6. Create train/test splits

# COMMAND ----------

# DBTITLE 1,Setup and Configuration
dbutils.widgets.dropdown("reset_all_data", "false", ["true", "false"], "Reset all data")

# COMMAND ----------

# MAGIC %run ./_resources/00-setup $reset_all_data=$reset_all_data $catalog=mlops_advanced $db=churn_demo

# COMMAND ----------

# DBTITLE 1,Load Bronze Data
from pyspark.sql import functions as F

df = spark.table(bronze_table)
record_count = df.count()

print(f"Loaded {record_count:,} customer records from {bronze_table}")

# COMMAND ----------

# DBTITLE 1,Feature Engineering Function
def create_churn_features(input_df):
    """Transform raw customer data into ML-ready features"""

    # Clean TotalCharges column
    df_clean = input_df.withColumn(
        "TotalCharges",
        F.when(F.col("TotalCharges") == " ", "0")
         .otherwise(F.col("TotalCharges"))
         .cast("double")
    )

    # Cast numeric columns
    df_clean = df_clean.withColumn("tenure", F.col("tenure").cast("int"))
    df_clean = df_clean.withColumn("MonthlyCharges", F.col("MonthlyCharges").cast("double"))
    df_clean = df_clean.withColumn("SeniorCitizen", F.col("SeniorCitizen").cast("int"))

    # Categorical columns to encode
    categorical_cols = [
        'gender', 'Partner', 'Dependents', 'PhoneService',
        'MultipleLines', 'InternetService', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies', 'Contract',
        'PaperlessBilling', 'PaymentMethod'
    ]

    # One-hot encode using pandas
    import pandas as pd
    pdf = df_clean.toPandas()
    pdf_encoded = pd.get_dummies(
        pdf,
        columns=categorical_cols,
        prefix=categorical_cols,
        drop_first=False,
        dtype=int
    )

    # Create binary churn label
    pdf_encoded['churn'] = (pdf_encoded['Churn'] == 'Yes').astype(int)

    # Clean column names
    pdf_encoded.columns = (
        pdf_encoded.columns
        .str.replace(' ', '_', regex=False)
        .str.replace('(', '_', regex=False)
        .str.replace(')', '', regex=False)
        .str.replace('-', '_', regex=False)
    )

    # Select relevant columns
    feature_cols = [col for col in pdf_encoded.columns
                   if col not in ['Churn', 'customerID']]
    pdf_features = pdf_encoded[['customerID'] + feature_cols]

    # Convert back to Spark
    features_df = spark.createDataFrame(pdf_features)
    features_df = features_df.na.fill(0)

    return features_df

# COMMAND ----------

# DBTITLE 1,Create Features
features_df = create_churn_features(df)
feature_count = len(features_df.columns) - 1

print(f"Created {features_df.count():,} feature records")
print(f"Number of features: {feature_count}")

# COMMAND ----------

# DBTITLE 1,Save to Feature Table
features_df.write \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .option("delta.enableChangeDataFeed", "true") \
    .saveAsTable(features_table)

spark.sql(f"""
    ALTER TABLE {features_table}
    SET TBLPROPERTIES (
        'delta.enableChangeDataFeed' = 'true',
        'delta.columnMapping.mode' = 'name'
    )
""")

print(f"Features saved to {features_table}")

# COMMAND ----------

# DBTITLE 1,Create Training and Test Splits
train_df, test_df = features_df.randomSplit([0.8, 0.2], seed=42)

train_df.write.mode("overwrite").saveAsTable(train_table)
test_df.write.mode("overwrite").saveAsTable(test_table)

print(f"Training set: {train_df.count():,} records")
print(f"Test set: {test_df.count():,} records")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC
# MAGIC Features are ready for model training. Proceed to:
# MAGIC - **Next:** [02 - Model Training]($./02_model_training_hpo)
