# Databricks notebook source
# MAGIC %md
# MAGIC # Model Registration to Unity Catalog
# MAGIC
# MAGIC <img src="https://github.com/pravinva/mlops-advanced-drift-retrain/raw/main/diagrams/mlops-advanced-3-registration.png" width="1200px">
# MAGIC
# MAGIC <!-- Tracking -->
# MAGIC <img width="1px" src="https://www.google-analytics.com/collect?v=1&gtm=GTM-NKQ8TT7&tid=UA-163989034-1&cid=555&aip=1&t=event&ec=field_demos&ea=display&dp=%2F42_field_demos%2Fml%2Fmlops_advanced%2F03_registration&dt=MLOPS_ADVANCED">
# MAGIC
# MAGIC ## Register Trained Model with Unity Catalog Governance
# MAGIC
# MAGIC This notebook registers the trained model to Unity Catalog Model Registry with full lineage tracking.
# MAGIC
# MAGIC **What we'll do:**
# MAGIC 1. Find best model from MLflow tracking
# MAGIC 2. Register model to Unity Catalog
# MAGIC 3. Add governance metadata and descriptions
# MAGIC 4. Assign "Challenger" alias for validation
# MAGIC 5. Add lineage tags for feature/training tables
# MAGIC 6. Test model loading from registry

# COMMAND ----------

# DBTITLE 1,Setup and Configuration
dbutils.widgets.dropdown("reset_all_data", "false", ["true", "false"], "Reset all data")

# COMMAND ----------

# MAGIC %run ./_resources/00-setup $reset_all_data=$reset_all_data $catalog=mlops_advanced $db=churn_demo

# COMMAND ----------

# DBTITLE 1,Install Required Libraries
# MAGIC %pip install lightgbm --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Re-run Setup After Restart
# MAGIC %run ./_resources/00-setup $reset_all_data=$reset_all_data $catalog=mlops_advanced $db=churn_demo

# COMMAND ----------

# DBTITLE 1,Find Best Model from MLflow
import mlflow
from mlflow.tracking import MlflowClient
from datetime import datetime

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

experiment = mlflow.get_experiment_by_name(experiment_path)

if experiment is None:
    raise Exception(f"Experiment not found: {experiment_path}. Run notebook 02 first.")

runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="tags.mlflow.runName = 'final_champion_model'",
    order_by=["metrics.f1_score DESC"],
    max_results=1
)

if len(runs) == 0:
    raise Exception("No model runs found with name 'final_champion_model'. Run notebook 02 first.")

best_run = runs.iloc[0]
run_id = best_run.run_id
f1_score = best_run["metrics.f1_score"]
accuracy = best_run["metrics.accuracy"]
precision = best_run["metrics.precision"]
recall = best_run["metrics.recall"]

print(f"Best Model Found:")
print(f"  Run ID: {run_id}")
print(f"  F1 Score: {f1_score:.4f}")
print(f"  Accuracy: {accuracy:.4f}")

# COMMAND ----------

# DBTITLE 1,Register Model to Unity Catalog
model_uri = f"runs:/{run_id}/model"

print(f"Registering model to Unity Catalog: {model_name}")

model_version = mlflow.register_model(
    model_uri=model_uri,
    name=model_name,
    tags={
        "model_type": "LightGBM",
        "use_case": "customer_churn",
        "training_date": datetime.now().isoformat(),
        "f1_score": str(f1_score),
        "accuracy": str(accuracy),
        "precision": str(precision),
        "recall": str(recall),
        "run_id": run_id
    }
)

print(f"✅ Model registered: {model_name} version {model_version.version}")

# COMMAND ----------

# DBTITLE 1,Add Model Description and Metadata
client.update_model_version(
    name=model_name,
    version=model_version.version,
    description=f"""
# Customer Churn Prediction Model

**Model Type:** LightGBM Binary Classifier
**Training Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Training Run ID:** `{run_id}`

## Performance Metrics

| Metric | Value |
|--------|-------|
| F1 Score | {f1_score:.4f} |
| Accuracy | {accuracy:.4f} |
| Precision | {precision:.4f} |
| Recall | {recall:.4f} |

## Use Case
Predict customer churn probability for proactive retention campaigns.

## Training Data
- **Feature Table:** `{features_table}`
- **Training Set:** `{train_table}`
- **Test Set:** `{test_table}`

## Model Governance
This model is registered with Unity Catalog for:
- Version control and lineage tracking
- Access control and audit logging
- Alias-based deployment (@Champion/@Challenger)
"""
)

print("✅ Model description updated")

# COMMAND ----------

# DBTITLE 1,Set Challenger Alias
client.set_registered_model_alias(
    name=model_name,
    alias="Challenger",
    version=model_version.version
)

print(f"✅ Model version {model_version.version} aliased as 'Challenger'")
print(f"   Load with: mlflow.pyfunc.load_model('models:/{model_name}@Challenger')")

# COMMAND ----------

# DBTITLE 1,Add Lineage Metadata Tags
lineage_tags = {
    "feature_table": features_table,
    "training_table": train_table,
    "test_table": test_table,
    "bronze_table": bronze_table,
    "experiment_path": experiment_path
}

for key, value in lineage_tags.items():
    client.set_model_version_tag(
        name=model_name,
        version=model_version.version,
        key=key,
        value=value
    )

print("✅ Lineage metadata added:")
for key, value in lineage_tags.items():
    print(f"   {key}: {value}")

# COMMAND ----------

# DBTITLE 1,Test Model Loading from Registry
model = mlflow.pyfunc.load_model(f"models:/{model_name}@Challenger")

print("✅ Model loads successfully from Unity Catalog")
print(f"   Model URI: models:/{model_name}@Challenger")
print(f"   Model Version: {model_version.version}")

# COMMAND ----------

# DBTITLE 1,Save Model Metadata for Downstream Tasks
dbutils.jobs.taskValues.set(key="model_name", value=model_name)
dbutils.jobs.taskValues.set(key="model_version", value=str(model_version.version))
dbutils.jobs.taskValues.set(key="model_run_id", value=run_id)
dbutils.jobs.taskValues.set(key="f1_score", value=float(f1_score))

print("✅ Model metadata saved for downstream tasks:")
print(f"   model_name: {model_name}")
print(f"   model_version: {model_version.version}")
print(f"   model_run_id: {run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC
# MAGIC Model registered with Challenger alias. Proceed to:
# MAGIC - **Next:** [04a - Challenger Validation]($./04a_challenger_validation)
