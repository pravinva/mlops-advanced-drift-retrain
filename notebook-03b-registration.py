# Databricks notebook source
# MAGIC %md
# MAGIC # Register Model to Unity Catalog

# COMMAND ----------

# MAGIC %run ./notebook-00-setup

# COMMAND ----------

# DBTITLE 1,Load Best Model
import mlflow
from mlflow.tracking import MlflowClient
from datetime import datetime

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

experiment = mlflow.get_experiment_by_name(experiment_path)
runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="tags.mlflow.runName = 'final_champion_model'",
    order_by=["metrics.f1_score DESC"],
    max_results=1
)

if len(runs) == 0:
    raise Exception("No model runs found. Run notebook 02 first.")

best_run = runs.iloc[0]
run_id = best_run.run_id
f1_score = best_run["metrics.f1_score"]

print(f"Best Model Run ID: {run_id}")
print(f"F1 Score: {f1_score:.4f}")

# COMMAND ----------

# DBTITLE 1,Register Model
model_uri = f"runs:/{run_id}/model"

model_version = mlflow.register_model(
    model_uri=model_uri,
    name=model_name,
    tags={
        "model_type": "LightGBM",
        "use_case": "customer_churn",
        "training_date": datetime.now().isoformat(),
        "f1_score": str(f1_score)
    }
)

print(f"✅ Model registered: {model_name} v{model_version.version}")

# COMMAND ----------

# DBTITLE 1,Add Model Description
client.update_model_version(
    name=model_name,
    version=model_version.version,
    description=f"""
Customer Churn Prediction Model

**Model Type:** LightGBM Classifier
**Training Date:** {datetime.now().strftime('%Y-%m-%d')}
**Training Run ID:** {run_id}

**Performance Metrics:**
- F1 Score: {f1_score:.4f}
- Accuracy: {best_run['metrics.accuracy']:.4f}
- Precision: {best_run['metrics.precision']:.4f}
- Recall: {best_run['metrics.recall']:.4f}

**Use Case:** Predict customer churn for proactive retention
"""
)

# COMMAND ----------

# MAGIC %pip install lightgbm
# MAGIC
# MAGIC import mlflow
# MAGIC
# MAGIC # Load the model from Unity Catalog using MLflow and the Challenger alias
# MAGIC model = mlflow.pyfunc.load_model(f"models:/{model_name}@Challenger")
# MAGIC
# MAGIC print("✅ Model loads successfully from Unity Catalog")

# COMMAND ----------

# DBTITLE 1,Set as Challenger
client.set_registered_model_alias(
    name=model_name,
    alias="Challenger",
    version=model_version.version
)
print(f"✅ Model aliased as 'Challenger'")

# COMMAND ----------

# DBTITLE 1,Add Lineage Tags
client.set_model_version_tag(
    name=model_name,
    version=model_version.version,
    key="feature_table",
    value=features_table
)

client.set_model_version_tag(
    name=model_name,
    version=model_version.version,
    key="training_table",
    value=f"{catalog}.{db}.churn_features_train"
)

print("✅ Lineage metadata added")

# COMMAND ----------

# DBTITLE 1,Test Model Loading
model = mlflow.pyfunc.load_model(f"models:/{model_name}@Challenger")

print("✅ Model loads successfully from Unity Catalog")

# COMMAND ----------

# DBTITLE 1,Save Model Info
dbutils.jobs.taskValues.set(key="model_name", value=model_name)
dbutils.jobs.taskValues.set(key="model_version", value=str(model_version.version))
dbutils.jobs.taskValues.set(key="model_run_id", value=run_id)
print("✅ Model info saved for downstream tasks")