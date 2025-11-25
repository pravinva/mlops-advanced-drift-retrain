# Databricks notebook source
# MAGIC %md
# MAGIC # MLOps Advanced - Setup Configuration
# MAGIC
# MAGIC This notebook initializes the environment for the MLOps Advanced demo.
# MAGIC
# MAGIC **DO NOT RUN DIRECTLY** - Called via `%run ./_resources/00-setup` from other notebooks

# COMMAND ----------

# DBTITLE 1,Get Parameters
dbutils.widgets.text("reset_all_data", "false", "Reset all data")
dbutils.widgets.text("catalog", "mlops_advanced", "Catalog name")
dbutils.widgets.text("db", "churn_demo", "Database name")

reset_all_data = dbutils.widgets.get("reset_all_data") == "true"
catalog = dbutils.widgets.get("catalog")
db = dbutils.widgets.get("db")

# COMMAND ----------

# DBTITLE 1,Call Global Setup
# This calls the Databricks global setup notebook
# It will be automatically configured when integrated into dbdemos
try:
    # Try to call global setup (will work when integrated into dbdemos)
    dbutils.notebook.run("../../../_resources/00-global-setup", 600, {
        "reset_all_data": str(reset_all_data),
        "catalog": catalog,
        "db": db
    })
except:
    # Fallback for standalone execution
    print("Running in standalone mode (not integrated with dbdemos)")
    pass

# COMMAND ----------

# DBTITLE 1,Initialize Catalog and Schema
print(f"Initializing MLOps Advanced Demo")
print(f"  Catalog: {catalog}")
print(f"  Database: {db}")
print(f"  Reset all data: {reset_all_data}")

# Create catalog if doesn't exist
spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog}")
spark.sql(f"USE CATALOG {catalog}")

# Create database/schema if doesn't exist
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{db}")
spark.sql(f"USE SCHEMA {db}")

print(f"✅ Catalog and schema ready: {catalog}.{db}")

# COMMAND ----------

# DBTITLE 1,Set Global Variables
# Get current user
user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
username_prefix = user.split('@')[0].replace('.', '_')

# Model and experiment configuration
model_name = f"{catalog}.{db}.churn_model"
experiment_path = f"/Users/{user}/mlops_churn_experiments"

# Table names
bronze_table = f"{catalog}.{db}.bronze_customers"
features_table = f"{catalog}.{db}.churn_features"
train_table = f"{catalog}.{db}.churn_features_train"
test_table = f"{catalog}.{db}.churn_features_test"
inference_table = f"{catalog}.{db}.churn_inference_logs"
predictions_table = f"{catalog}.{db}.churn_predictions"

# Storage path (for checkpoints, temp data, etc.)
cloud_storage_path = f"/Users/{user}/mlops_advanced_demo"

# Make variables available to calling notebooks
globals()['catalog'] = catalog
globals()['db'] = db
globals()['dbName'] = db  # Alias for compatibility
globals()['user'] = user
globals()['username_prefix'] = username_prefix
globals()['model_name'] = model_name
globals()['experiment_path'] = experiment_path
globals()['bronze_table'] = bronze_table
globals()['features_table'] = features_table
globals()['train_table'] = train_table
globals()['test_table'] = test_table
globals()['inference_table'] = inference_table
globals()['predictions_table'] = predictions_table
globals()['cloud_storage_path'] = cloud_storage_path

print(f"\n✅ Configuration variables set:")
print(f"   Model: {model_name}")
print(f"   Experiment: {experiment_path}")
print(f"   Bronze table: {bronze_table}")
print(f"   Features table: {features_table}")

# COMMAND ----------

# DBTITLE 1,Handle Reset All Data
if reset_all_data:
    print("\n⚠️ RESET MODE: Dropping all tables and data")

    # Drop all tables in the schema
    tables = spark.sql(f"SHOW TABLES IN {catalog}.{db}").collect()
    for table in tables:
        table_name = table.tableName
        full_table_name = f"{catalog}.{db}.{table_name}"
        print(f"   Dropping {full_table_name}")
        spark.sql(f"DROP TABLE IF EXISTS {full_table_name}")

    # Delete checkpoint and temp storage
    dbutils.fs.rm(cloud_storage_path, True)

    # Delete MLflow experiment runs (optional - commented out to preserve history)
    # try:
    #     import mlflow
    #     mlflow.set_experiment(experiment_path)
    #     experiment = mlflow.get_experiment_by_name(experiment_path)
    #     if experiment:
    #         client = mlflow.tracking.MlflowClient()
    #         runs = client.search_runs(experiment.experiment_id)
    #         for run in runs:
    #             client.delete_run(run.info.run_id)
    # except:
    #     pass

    # Try to delete model versions from Unity Catalog
    try:
        from mlflow.tracking import MlflowClient
        client = MlflowClient()

        # Delete all versions
        versions = client.search_model_versions(f"name='{model_name}'")
        for version in versions:
            print(f"   Deleting model version {version.version}")
            client.delete_model_version(model_name, version.version)

        # Delete model
        print(f"   Deleting registered model {model_name}")
        client.delete_registered_model(model_name)
    except Exception as e:
        print(f"   Note: Could not delete model (may not exist): {e}")

    print("✅ Reset complete - all data cleared")
else:
    print("\n✅ Using existing data (reset_all_data=false)")

# COMMAND ----------

# DBTITLE 1,Load Demo Data
# Call the data loading notebook
try:
    dbutils.notebook.run("./_resources/00-load-data", 600, {
        "catalog": catalog,
        "db": db,
        "reset_all_data": str(reset_all_data)
    })
    print("✅ Demo data loaded")
except Exception as e:
    print(f"⚠️ Data loading notebook not found or failed: {e}")
    print("   Will load data inline if needed")

# COMMAND ----------

# DBTITLE 1,Setup MLflow
import mlflow

# Set experiment
mlflow.set_experiment(experiment_path)

# Set Unity Catalog as registry
mlflow.set_registry_uri("databricks-uc")

print(f"✅ MLflow configured:")
print(f"   Experiment: {experiment_path}")
print(f"   Registry: databricks-uc")

# COMMAND ----------

# DBTITLE 1,Helper Functions
def is_folder_empty(folder_path):
    """Check if a folder is empty or doesn't exist"""
    try:
        return len(dbutils.fs.ls(folder_path)) == 0
    except:
        return True

def get_latest_model_version(model_name, stage=None):
    """Get latest version of a model, optionally filtered by stage"""
    from mlflow.tracking import MlflowClient
    client = MlflowClient()

    if stage:
        versions = client.get_latest_versions(model_name, stages=[stage])
    else:
        versions = client.search_model_versions(f"name='{model_name}'")
        versions = sorted(versions, key=lambda x: int(x.version), reverse=True)

    return versions[0] if versions else None

def get_model_by_alias(model_name, alias):
    """Get model version by alias (Champion, Challenger, etc.)"""
    from mlflow.tracking import MlflowClient
    client = MlflowClient()

    try:
        return client.get_model_version_by_alias(model_name, alias)
    except:
        return None

# Make functions available globally
globals()['is_folder_empty'] = is_folder_empty
globals()['get_latest_model_version'] = get_latest_model_version
globals()['get_model_by_alias'] = get_model_by_alias

print("✅ Helper functions loaded")

# COMMAND ----------

# DBTITLE 1,Setup Complete
print("\n" + "="*60)
print("MLOPS ADVANCED DEMO - SETUP COMPLETE")
print("="*60)
print(f"Catalog: {catalog}")
print(f"Database: {db}")
print(f"Model: {model_name}")
print(f"User: {user}")
print("="*60)
print("\n✅ Ready to run demo notebooks")
