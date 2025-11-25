# Databricks notebook source
# MAGIC %md
# MAGIC # MLOps End-to-End - Advanced Track Setup
# MAGIC
# MAGIC Configuration and initialization for the complete MLOps pipeline

# COMMAND ----------

# DBTITLE 1,Configuration
catalog = "art_mlops"  # Change to your catalog
db = "mlops_churn_demo"
user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()

model_name = f"{catalog}.{db}.churn_model"
experiment_path = f"/Users/{user}/mlops_churn_experiments"

bronze_table = f"{catalog}.{db}.bronze_customers"
features_table = f"{catalog}.{db}.churn_features"
inference_table = f"{catalog}.{db}.churn_inference_logs"

print(f"Catalog: {catalog}")
print(f"Database: {db}")
print(f"Model: {model_name}")

# COMMAND ----------

# DBTITLE 1,Create Database
spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{db}")
spark.sql(f"USE CATALOG {catalog}")
spark.sql(f"USE SCHEMA {db}")
print("✅ Database setup complete")

# COMMAND ----------

# DBTITLE 1,Load Sample Data
import pandas as pd

data_url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
pdf = pd.read_csv(data_url)
df = spark.createDataFrame(pdf)

df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(bronze_table)
print(f"✅ Loaded {df.count()} records to {bronze_table}")

# COMMAND ----------

# DBTITLE 1,Enable Change Data Feed
spark.sql(f"""
    ALTER TABLE {bronze_table} 
    SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
""")
print("✅ Change Data Feed enabled")

# COMMAND ----------

# DBTITLE 1,MLflow Setup
import mlflow

mlflow.set_experiment(experiment_path)
mlflow.set_registry_uri("databricks-uc")
print(f"✅ MLflow configured")