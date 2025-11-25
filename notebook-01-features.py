# Databricks notebook source
# MAGIC %md
# MAGIC # Feature Engineering with Feature Store

# COMMAND ----------

# MAGIC %run ./notebook-00-setup

# COMMAND ----------

# DBTITLE 1,Load Bronze Data
from pyspark.sql import functions as F

df = spark.table(bronze_table)
print(f"Loaded {df.count()} customer records")

# COMMAND ----------

# DBTITLE 1,Feature Engineering Function
def create_churn_features(input_df):
    """Transform raw customer data into ML-ready features"""
    
    # Clean TotalCharges
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
    
    # Categorical columns to one-hot encode
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
print(f"Created {features_df.count()} feature records")
print(f"Number of features: {len(features_df.columns) - 1}")

# COMMAND ----------

# DBTITLE 1,Save to Feature Table
features_df.write \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .option("delta.enableChangeDataFeed", "true") \
    .saveAsTable(features_table)

print(f"✅ Features saved to {features_table}")

spark.sql(f"""
    ALTER TABLE {features_table}
    SET TBLPROPERTIES (
        'delta.enableChangeDataFeed' = 'true',
        'delta.columnMapping.mode' = 'name'
    )
""")

# COMMAND ----------

# DBTITLE 1,Create Training/Test Split
train_df, test_df = features_df.randomSplit([0.8, 0.2], seed=42)

train_table = f"{catalog}.{db}.churn_features_train"
test_table = f"{catalog}.{db}.churn_features_test"

train_df.write.mode("overwrite").saveAsTable(train_table)
test_df.write.mode("overwrite").saveAsTable(test_table)

print(f"✅ Training set: {train_df.count()} records → {train_table}")
print(f"✅ Test set: {test_df.count()} records → {test_table}")