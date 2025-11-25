# Databricks notebook source
# MAGIC %md
# MAGIC # Load Demo Data - IBM Telco Customer Churn
# MAGIC
# MAGIC This notebook loads the IBM Telco Customer Churn dataset for the MLOps Advanced demo.
# MAGIC
# MAGIC **Dataset:** IBM Telco Customer Churn
# MAGIC **Source:** https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv
# MAGIC **License:** Open dataset
# MAGIC **Records:** ~7,000 customers

# COMMAND ----------

# DBTITLE 1,Get Parameters
dbutils.widgets.text("catalog", "mlops_advanced", "Catalog name")
dbutils.widgets.text("db", "churn_demo", "Database name")
dbutils.widgets.text("reset_all_data", "false", "Reset all data")

catalog = dbutils.widgets.get("catalog")
db = dbutils.widgets.get("db")
reset_all_data = dbutils.widgets.get("reset_all_data") == "true"

bronze_table = f"{catalog}.{db}.bronze_customers"

# COMMAND ----------

# DBTITLE 1,Check if Data Already Loaded
try:
    existing_count = spark.table(bronze_table).count()
    if existing_count > 0 and not reset_all_data:
        print(f"✅ Data already loaded: {existing_count} records in {bronze_table}")
        print("   Skipping data load (use reset_all_data=true to reload)")
        dbutils.notebook.exit("Data already loaded")
except:
    print(f"Bronze table doesn't exist or is empty, loading data...")

# COMMAND ----------

# DBTITLE 1,Load IBM Telco Dataset
import pandas as pd
from pyspark.sql import functions as F

print("Loading IBM Telco Customer Churn dataset...")

# Dataset URL
data_url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"

try:
    # Load via pandas then convert to Spark
    print(f"  Downloading from: {data_url}")
    pdf = pd.read_csv(data_url)
    print(f"  Loaded {len(pdf)} records")

    # Convert to Spark DataFrame
    df = spark.createDataFrame(pdf)

    # Add metadata columns
    df = df.withColumn("ingestion_timestamp", F.current_timestamp())
    df = df.withColumn("source", F.lit("IBM_Telco_Public_Dataset"))

    print(f"  Dataset shape: {len(pdf)} rows × {len(pdf.columns)} columns")

except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    raise

# COMMAND ----------

# DBTITLE 1,Save to Bronze Table
print(f"\nSaving to bronze table: {bronze_table}")

# Write to Delta table
df.write \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .option("delta.enableChangeDataFeed", "true") \
    .saveAsTable(bronze_table)

# Enable Change Data Feed
spark.sql(f"""
    ALTER TABLE {bronze_table}
    SET TBLPROPERTIES (
        'delta.enableChangeDataFeed' = 'true',
        'delta.columnMapping.mode' = 'name'
    )
""")

final_count = spark.table(bronze_table).count()
print(f"✅ Data saved: {final_count} records in {bronze_table}")

# COMMAND ----------

print("\n" + "="*60)
print("DATA LOAD COMPLETE")
print("="*60)
print(f"Table: {bronze_table}")
print(f"Records: {final_count:,}")
print(f"Source: IBM Telco Public Dataset")
print("="*60)
