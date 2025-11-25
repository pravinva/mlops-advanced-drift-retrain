# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC # Feature Engineering with Unity Catalog Feature Store
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/raw/main/images/product/mlops/advanced/banners/mlops-advanced-2-feature-store.png" width="1200px" style="float: right" />
# MAGIC
# MAGIC ## Transform Raw Data into ML-Ready Features
# MAGIC
# MAGIC This notebook demonstrates production-grade feature engineering with Unity Catalog Feature Store integration:
# MAGIC
# MAGIC ### Key Capabilities
# MAGIC
# MAGIC 1. **Feature Store Integration** - Unity Catalog-managed features with lineage
# MAGIC 2. **One-Hot Encoding** - Categorical variables to binary features
# MAGIC 3. **Data Cleaning** - Handle missing values and type conversions
# MAGIC 4. **Feature Reusability** - Share features across models and teams
# MAGIC 5. **Change Data Feed** - Track feature evolution over time
# MAGIC
# MAGIC ### Feature Engineering Process
# MAGIC
# MAGIC ```
# MAGIC Raw Bronze Data (IBM Telco)
# MAGIC     ↓
# MAGIC Data Cleaning (TotalCharges, type casting)
# MAGIC     ↓
# MAGIC One-Hot Encoding (15 categorical columns)
# MAGIC     ↓
# MAGIC Binary Label Creation (churn: 0 or 1)
# MAGIC     ↓
# MAGIC Feature Store Registration (Unity Catalog)
# MAGIC     ↓
# MAGIC Training/Test Splits (80/20)
# MAGIC ```
# MAGIC
# MAGIC ### Features Generated
# MAGIC
# MAGIC - **Numeric:** tenure, MonthlyCharges, TotalCharges, SeniorCitizen
# MAGIC - **One-Hot Encoded:** 50+ binary columns from categorical variables
# MAGIC - **Target:** churn (binary classification label)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **Next Step:** [02-Model Training]($./notebook-02-training) to train with hyperparameter optimization

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration Setup

# COMMAND ----------

# MAGIC %run ./notebook-00-setup

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install Feature Engineering Client

# COMMAND ----------

# MAGIC %pip install databricks-feature-engineering --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Import Libraries
from databricks.feature_engineering import FeatureEngineeringClient
from pyspark.sql import functions as F
import pandas as pd

# Initialize Feature Engineering Client
fe = FeatureEngineeringClient()

print("Feature Engineering Client initialized")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Bronze Data
# MAGIC
# MAGIC Load raw customer data from the bronze layer table created in notebook 00-setup.

# COMMAND ----------

# DBTITLE 1,Load Bronze Data
df = spark.table(bronze_table)
row_count = df.count()

print(f"✅ Loaded {row_count:,} customer records from {bronze_table}")
print(f"\nSchema:")
df.printSchema()

# Sample data preview
display(df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Engineering Function
# MAGIC
# MAGIC Transform raw customer data into ML-ready features:
# MAGIC - Clean TotalCharges (handle blank values)
# MAGIC - Cast numeric columns to appropriate types
# MAGIC - One-hot encode 15 categorical columns
# MAGIC - Create binary churn label
# MAGIC - Clean column names for compatibility

# COMMAND ----------

# DBTITLE 1,Feature Engineering Function
def create_churn_features(input_df):
    """
    Transform raw customer data into ML-ready features

    Args:
        input_df: Spark DataFrame with raw customer data

    Returns:
        Spark DataFrame with engineered features
    """

    # Step 1: Clean TotalCharges (handle blank values as "0")
    df_clean = input_df.withColumn(
        "TotalCharges",
        F.when(F.col("TotalCharges") == " ", "0")
         .otherwise(F.col("TotalCharges"))
         .cast("double")
    )

    # Step 2: Cast numeric columns to proper types
    df_clean = df_clean.withColumn("tenure", F.col("tenure").cast("int"))
    df_clean = df_clean.withColumn("MonthlyCharges", F.col("MonthlyCharges").cast("double"))
    df_clean = df_clean.withColumn("SeniorCitizen", F.col("SeniorCitizen").cast("int"))

    # Step 3: Define categorical columns for encoding
    categorical_cols = [
        'gender', 'Partner', 'Dependents', 'PhoneService',
        'MultipleLines', 'InternetService', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies', 'Contract',
        'PaperlessBilling', 'PaymentMethod'
    ]

    # Step 4: One-hot encode categorical variables using pandas
    pdf = df_clean.toPandas()
    pdf_encoded = pd.get_dummies(
        pdf,
        columns=categorical_cols,
        prefix=categorical_cols,
        drop_first=False,  # Keep all categories for interpretability
        dtype=int
    )

    # Step 5: Create binary churn label
    pdf_encoded['churn'] = (pdf_encoded['Churn'] == 'Yes').astype(int)

    # Step 6: Clean column names for Delta table compatibility
    pdf_encoded.columns = (
        pdf_encoded.columns
        .str.replace(' ', '_', regex=False)
        .str.replace('(', '_', regex=False)
        .str.replace(')', '', regex=False)
        .str.replace('-', '_', regex=False)
    )

    # Step 7: Select relevant columns (exclude original Churn column)
    feature_cols = [col for col in pdf_encoded.columns
                   if col not in ['Churn', 'customerID']]
    pdf_features = pdf_encoded[['customerID'] + feature_cols]

    # Step 8: Convert back to Spark and fill any remaining nulls
    features_df = spark.createDataFrame(pdf_features)
    features_df = features_df.na.fill(0)

    return features_df

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Features
# MAGIC
# MAGIC Apply the feature engineering function to transform raw data.

# COMMAND ----------

# DBTITLE 1,Create Features
print("Creating features...")

features_df = create_churn_features(df)

feature_count = len(features_df.columns) - 1  # Exclude customerID
record_count = features_df.count()

print(f"✅ Created {record_count:,} feature records")
print(f"✅ Generated {feature_count} features")
print(f"\nFeature columns:")
print(", ".join([c for c in features_df.columns if c != 'customerID'][:10]) + "...")

# Show sample
display(features_df.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register Features to Feature Store
# MAGIC
# MAGIC ### Option 1: Feature Engineering Client (Recommended)
# MAGIC
# MAGIC Uses Unity Catalog Feature Store for:
# MAGIC - Automatic lineage tracking
# MAGIC - Feature discovery and reuse
# MAGIC - Online/offline feature serving
# MAGIC - Point-in-time correctness

# COMMAND ----------

# DBTITLE 1,Register to Feature Store (with Unity Catalog)
try:
    # Check if feature table exists
    existing_tables = spark.catalog.listTables(f"{catalog}.{db}")
    table_exists = any(t.name == features_table.split('.')[-1] for t in existing_tables)

    if table_exists:
        print(f"Feature table {features_table} already exists")
        print("Updating with new features...")

        # Write features using Feature Engineering Client
        fe.write_table(
            name=features_table,
            df=features_df,
            mode="overwrite"
        )
    else:
        print(f"Creating new feature table: {features_table}")

        # Create feature table with Feature Engineering Client
        fe.create_table(
            name=features_table,
            primary_keys=["customerID"],
            df=features_df,
            description="Customer churn prediction features with one-hot encoded categorical variables",
            tags={
                "project": "mlops-advanced",
                "version": "1.0",
                "data_source": "IBM_Telco",
                "feature_type": "customer_attributes"
            }
        )

    # Enable Change Data Feed for drift tracking
    spark.sql(f"""
        ALTER TABLE {features_table}
        SET TBLPROPERTIES (
            'delta.enableChangeDataFeed' = 'true',
            'delta.columnMapping.mode' = 'name'
        )
    """)

    print(f"✅ Features registered to Feature Store: {features_table}")
    print(f"   - Primary Key: customerID")
    print(f"   - Feature Count: {feature_count}")
    print(f"   - Change Data Feed: Enabled")

except Exception as e:
    print(f"Feature Store registration not available (requires DBR ML 13.0+)")
    print(f"Falling back to standard Delta table...")
    print(f"Error: {e}")

    # Fallback: Write as standard Delta table
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

    print(f"✅ Features saved to Delta table: {features_table}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Statistics and Quality Checks
# MAGIC
# MAGIC Validate feature quality before training.

# COMMAND ----------

# DBTITLE 1,Feature Statistics
from pyspark.sql.functions import col, count, when, isnan, isnull

# Check for nulls and data quality
feature_cols = [c for c in features_df.columns if c not in ['customerID', 'churn']]

null_counts = features_df.select([
    count(when(isnan(c) | isnull(c), c)).alias(c)
    for c in feature_cols[:10]  # Sample first 10
]).collect()[0]

print("Feature Quality Check:")
print(f"  Total Records: {features_df.count():,}")
print(f"  Total Features: {len(feature_cols)}")
print(f"  Null Counts (sample): {dict(null_counts.asDict())}")

# Churn distribution
churn_dist = features_df.groupBy("churn").count().orderBy("churn")
print(f"\nChurn Distribution:")
display(churn_dist)

churn_rate = features_df.filter(col("churn") == 1).count() / features_df.count()
print(f"\nChurn Rate: {churn_rate:.2%}")

if churn_rate < 0.10 or churn_rate > 0.50:
    print(f"⚠️ Warning: Unusual churn rate - check data quality")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Training and Test Splits
# MAGIC
# MAGIC Split features into training (80%) and test (20%) sets for model development.

# COMMAND ----------

# DBTITLE 1,Create Training/Test Split
print("Creating train/test splits...")

# Random split with seed for reproducibility
train_df, test_df = features_df.randomSplit([0.8, 0.2], seed=42)

# Define table names
train_table = f"{catalog}.{db}.churn_features_train"
test_table = f"{catalog}.{db}.churn_features_test"

# Write training set
train_df.write.mode("overwrite").saveAsTable(train_table)
train_count = train_df.count()
train_churn_rate = train_df.filter(col("churn") == 1).count() / train_count

# Write test set
test_df.write.mode("overwrite").saveAsTable(test_table)
test_count = test_df.count()
test_churn_rate = test_df.filter(col("churn") == 1).count() / test_count

print(f"\n✅ Training set: {train_count:,} records → {train_table}")
print(f"   Churn rate: {train_churn_rate:.2%}")

print(f"\n✅ Test set: {test_count:,} records → {test_table}")
print(f"   Churn rate: {test_churn_rate:.2%}")

# Validate similar distributions
if abs(train_churn_rate - test_churn_rate) > 0.05:
    print(f"\n⚠️ Warning: Train/test churn rate difference > 5%")
else:
    print(f"\n✅ Train/test distributions are balanced")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Lineage and Metadata
# MAGIC
# MAGIC View feature table metadata and lineage information.

# COMMAND ----------

# DBTITLE 1,Feature Table Metadata
print("Feature Table Metadata:")
print("=" * 60)

# Get table description
table_desc = spark.sql(f"DESCRIBE EXTENDED {features_table}").collect()

for row in table_desc:
    if row.col_name in ['Type', 'Provider', 'Location', 'Table Properties']:
        print(f"{row.col_name}: {row.data_type}")

print("\n" + "=" * 60)

# Feature count by category
numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']
categorical_features = [c for c in feature_cols if c not in numeric_features]

print(f"\nFeature Summary:")
print(f"  Numeric Features: {len(numeric_features)}")
print(f"  Categorical Features (one-hot): {len(categorical_features)}")
print(f"  Total: {len(feature_cols)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC
# MAGIC Features are now ready for model training!
# MAGIC
# MAGIC **Proceed to:** [02-Model Training]($./notebook-02-training)
# MAGIC
# MAGIC ### What's Next
# MAGIC
# MAGIC 1. **Model Training** - Hyperparameter optimization with Optuna
# MAGIC 2. **Model Registration** - Register to Unity Catalog
# MAGIC 3. **Validation** - 7-stage quality gates
# MAGIC 4. **Production Deployment** - Batch or real-time inference
# MAGIC
# MAGIC ### Feature Store Benefits
# MAGIC
# MAGIC - **Reusability** - Use these features for other churn models
# MAGIC - **Lineage** - Track which features produced which models
# MAGIC - **Online Serving** - Real-time feature lookup for low-latency predictions
# MAGIC - **Drift Detection** - Monitor feature distribution changes over time
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **Back to:** [Overview Presentation]($./00_mlops_end2end_advanced_presentation)
