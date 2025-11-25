# Databricks notebook source
# MAGIC %md
# MAGIC # Real-Time Model Serving
# MAGIC
# MAGIC <img src="https://github.com/pravinva/mlops-advanced-drift-retrain/raw/main/diagrams/mlops-advanced-6-serving.png" width="1200px">
# MAGIC
# MAGIC <!-- Tracking -->
# MAGIC <img width="1px" src="https://www.google-analytics.com/collect?v=1&gtm=GTM-NKQ8TT7&tid=UA-163989034-1&cid=555&aip=1&t=event&ec=field_demos&ea=display&dp=%2F42_field_demos%2Fml%2Fmlops_advanced%2F06_serving&dt=MLOPS_ADVANCED">
# MAGIC
# MAGIC ## Deploy REST API Endpoint with Auto-Scaling
# MAGIC
# MAGIC This notebook demonstrates deploying a real-time serving endpoint for the Champion model.
# MAGIC
# MAGIC **What we'll do:**
# MAGIC 1. Load Champion model from Unity Catalog
# MAGIC 2. Create Model Serving endpoint with auto-scaling
# MAGIC 3. Enable inference table logging for monitoring
# MAGIC 4. Test endpoint with sample predictions
# MAGIC 5. Measure latency and performance
# MAGIC 6. Provide API integration examples
# MAGIC
# MAGIC **Note:** This is optional if you only need batch inference (Notebook 05).

# COMMAND ----------

# DBTITLE 1,Setup and Configuration
dbutils.widgets.dropdown("reset_all_data", "false", ["true", "false"], "Reset all data")

# COMMAND ----------

# MAGIC %run ./_resources/00-setup $reset_all_data=$reset_all_data $catalog=mlops_advanced $db=churn_demo

# COMMAND ----------

# DBTITLE 1,Install Required Libraries
# MAGIC %pip install databricks-sdk --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Re-run Setup After Restart
# MAGIC %run ./_resources/00-setup $reset_all_data=$reset_all_data $catalog=mlops_advanced $db=churn_demo

# COMMAND ----------

# DBTITLE 1,Get Champion Model
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

try:
    champion_version = client.get_model_version_by_alias(model_name, "Champion")
    print(f"Champion Model:")
    print(f"  Model: {model_name}")
    print(f"  Version: {champion_version.version}")
except Exception as e:
    raise Exception(f"No Champion model found. Run notebooks 02-04b first. Error: {e}")

# COMMAND ----------

# DBTITLE 1,Configure Serving Endpoint
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    ServedEntityInput,
    EndpointCoreConfigInput,
    AutoCaptureConfigInput
)

w = WorkspaceClient()

# Create endpoint name (must be DNS-compliant)
endpoint_name = f"churn-prediction-{username_prefix}"[:63]
endpoint_name = endpoint_name.replace('_', '-').replace('.', '-')

print(f"Endpoint Configuration:")
print(f"  Name: {endpoint_name}")
print(f"  Model: {model_name}")
print(f"  Version: {champion_version.version}")

# Check if endpoint already exists
endpoint_exists = False
try:
    existing_endpoint = w.serving_endpoints.get(name=endpoint_name)
    endpoint_exists = True
    print(f"  Status: Endpoint exists, will update")
except:
    print(f"  Status: Creating new endpoint")

# COMMAND ----------

# DBTITLE 1,Create Endpoint Configuration
# Endpoint configuration with auto-scaling and inference logging
endpoint_config = EndpointCoreConfigInput(
    name=endpoint_name,
    served_entities=[
        ServedEntityInput(
            entity_name=model_name,
            entity_version=str(champion_version.version),
            workload_size="Small",  # Small, Medium, Large
            scale_to_zero_enabled=True,  # Scale down when idle
            environment_vars={
                "MODEL_VERSION": str(champion_version.version)
            }
        )
    ],
    auto_capture_config=AutoCaptureConfigInput(
        catalog_name=catalog,
        schema_name=db,
        table_name_prefix="churn_endpoint",  # Creates inference table
        enabled=True
    )
)

print("Endpoint Features:")
print("  - Auto-scaling: Enabled")
print("  - Scale-to-zero: Enabled (cost-effective)")
print("  - Inference logging: Enabled")
print(f"  - Inference table: {catalog}.{db}.churn_endpoint_payload")

# COMMAND ----------

# DBTITLE 1,Deploy Endpoint
print(f"\nDeploying endpoint: {endpoint_name}")

if endpoint_exists:
    print("Updating existing endpoint configuration...")
    w.serving_endpoints.update_config(
        name=endpoint_name,
        served_entities=endpoint_config.served_entities,
        auto_capture_config=endpoint_config.auto_capture_config
    )
else:
    print("Creating new endpoint...")
    w.serving_endpoints.create(
        name=endpoint_name,
        config=endpoint_config
    )

print("Deployment initiated")

# COMMAND ----------

# DBTITLE 1,Wait for Endpoint Ready
import time

print("\nWaiting for endpoint to become ready...")
print("(This may take 5-15 minutes for initial deployment)\n")

max_wait = 20 * 60  # 20 minutes timeout
start = time.time()
endpoint = None

while time.time() - start < max_wait:
    try:
        endpoint = w.serving_endpoints.get(name=endpoint_name)
        status = endpoint.state.ready if endpoint.state else "UNKNOWN"

        if status == "READY":
            print(f"\nEndpoint is READY!")
            break

        elapsed = int(time.time() - start)
        mins = elapsed // 60
        secs = elapsed % 60
        print(f"Status: {status} - Elapsed: {mins}m {secs}s", end='\r')
        time.sleep(30)

    except Exception as e:
        print(f"\nError checking endpoint status: {e}")
        time.sleep(30)

if not endpoint or endpoint.state.ready != "READY":
    print("\nWarning: Endpoint not ready within timeout period")
    print("Check endpoint status in Databricks UI: Serving > Endpoints")

# COMMAND ----------

# DBTITLE 1,Endpoint Information
workspace_url = spark.conf.get("spark.databricks.workspaceUrl")
endpoint_url = f"https://{workspace_url}/serving-endpoints/{endpoint_name}/invocations"

print("\n" + "=" * 60)
print("SERVING ENDPOINT DETAILS")
print("=" * 60)
print(f"Name:                {endpoint_name}")
print(f"Status:              {endpoint.state.ready if endpoint else 'UNKNOWN'}")
print(f"Model:               {model_name}")
print(f"Version:             {champion_version.version}")
print(f"Workload Size:       Small")
print(f"Scale to Zero:       Enabled")
print(f"Inference Logging:   Enabled")
print(f"\nEndpoint URL:")
print(f"{endpoint_url}")
print("=" * 60)

# COMMAND ----------

# DBTITLE 1,Test Endpoint with Sample Data
import requests
import json

token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

# Get sample customer data
test_df = spark.table(test_table).limit(1)
test_pdf = test_df.toPandas()

feature_cols = [col for col in test_pdf.columns if col not in ['customerID', 'churn']]
test_input = test_pdf[feature_cols].to_dict(orient='records')[0]

headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
}

payload = {"dataframe_records": [test_input]}

print("Testing endpoint with sample prediction...")
response = requests.post(endpoint_url, headers=headers, json=payload)

if response.status_code == 200:
    prediction = response.json()
    pred_value = prediction['predictions'][0]

    risk_level = "High" if pred_value >= 0.7 else "Medium" if pred_value >= 0.4 else "Low"

    print(f"\nPrediction successful!")
    print(f"  Churn Probability: {pred_value:.4f}")
    print(f"  Risk Level:        {risk_level}")
else:
    print(f"\nPrediction failed: {response.status_code}")
    print(f"Error: {response.text}")

# COMMAND ----------

# DBTITLE 1,Latency Benchmark
import numpy as np

print("Running latency benchmark (20 requests)...")

latencies = []
single_record = test_pdf[feature_cols].to_dict(orient='records')[0]
payload = {"dataframe_records": [single_record]}

# Warm-up requests
print("  Warm-up...")
for _ in range(3):
    requests.post(endpoint_url, headers=headers, json=payload)

# Measure latency
print("  Measuring...")
for _ in range(20):
    start = time.time()
    requests.post(endpoint_url, headers=headers, json=payload)
    latencies.append((time.time() - start) * 1000)

print("\n" + "=" * 60)
print("LATENCY STATISTICS")
print("=" * 60)
print(f"Mean:   {np.mean(latencies):.2f}ms")
print(f"Median: {np.percentile(latencies, 50):.2f}ms")
print(f"P95:    {np.percentile(latencies, 95):.2f}ms")
print(f"P99:    {np.percentile(latencies, 99):.2f}ms")
print(f"Min:    {np.min(latencies):.2f}ms")
print(f"Max:    {np.max(latencies):.2f}ms")
print("=" * 60)

# COMMAND ----------

# DBTITLE 1,API Integration Examples
# MAGIC %md
# MAGIC ## API Integration
# MAGIC
# MAGIC ### Python Example
# MAGIC ```python
# MAGIC import requests
# MAGIC
# MAGIC ENDPOINT_URL = "{endpoint_url}"
# MAGIC DATABRICKS_TOKEN = "your_token_here"
# MAGIC
# MAGIC headers = {
# MAGIC     "Authorization": f"Bearer {DATABRICKS_TOKEN}",
# MAGIC     "Content-Type": "application/json"
# MAGIC }
# MAGIC
# MAGIC customer_data = {
# MAGIC     "tenure": 12,
# MAGIC     "MonthlyCharges": 70.50,
# MAGIC     "TotalCharges": 846.00,
# MAGIC     "SeniorCitizen": 0,
# MAGIC     # ... include all required features
# MAGIC }
# MAGIC
# MAGIC response = requests.post(
# MAGIC     ENDPOINT_URL,
# MAGIC     headers=headers,
# MAGIC     json={"dataframe_records": [customer_data]}
# MAGIC )
# MAGIC
# MAGIC if response.status_code == 200:
# MAGIC     churn_prob = response.json()['predictions'][0]
# MAGIC     print(f"Churn Probability: {churn_prob:.2%}")
# MAGIC ```
# MAGIC
# MAGIC ### cURL Example
# MAGIC ```bash
# MAGIC curl -X POST \
# MAGIC   '{endpoint_url}' \
# MAGIC   -H 'Authorization: Bearer YOUR_TOKEN' \
# MAGIC   -H 'Content-Type: application/json' \
# MAGIC   -d '{
# MAGIC     "dataframe_records": [
# MAGIC       {
# MAGIC         "tenure": 12,
# MAGIC         "MonthlyCharges": 70.50,
# MAGIC         ...
# MAGIC       }
# MAGIC     ]
# MAGIC   }'
# MAGIC ```

# COMMAND ----------

# DBTITLE 1,Save Endpoint Configuration
dbutils.jobs.taskValues.set(key="endpoint_name", value=endpoint_name)
dbutils.jobs.taskValues.set(key="endpoint_url", value=endpoint_url)

# Log to MLflow
with mlflow.start_run(run_name="model_serving_deployment"):
    mlflow.log_param("endpoint_name", endpoint_name)
    mlflow.log_param("endpoint_url", endpoint_url)
    mlflow.log_param("model_version", champion_version.version)
    mlflow.log_param("workload_size", "Small")
    mlflow.log_param("scale_to_zero", True)

    if len(latencies) > 0:
        mlflow.log_metric("avg_latency_ms", np.mean(latencies))
        mlflow.log_metric("p95_latency_ms", np.percentile(latencies, 95))
        mlflow.log_metric("p99_latency_ms", np.percentile(latencies, 99))

print("\nEndpoint configuration saved")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Endpoint Management
# MAGIC
# MAGIC ### Cost Optimization
# MAGIC - **Scale-to-zero enabled**: Endpoint scales down when idle
# MAGIC - **Workload size**: Start with Small, adjust based on load
# MAGIC - **Monitoring**: Track usage via inference tables
# MAGIC
# MAGIC ### Updating the Endpoint
# MAGIC When a new model is promoted to Champion:
# MAGIC 1. Re-run this notebook, or
# MAGIC 2. Use the Databricks UI to update the served model version
# MAGIC
# MAGIC ### Monitoring
# MAGIC - **Inference logs**: Automatically saved to `{catalog}.{db}.churn_endpoint_payload`
# MAGIC - **Latency metrics**: Available in Databricks Serving UI
# MAGIC - **Request/Error rates**: Dashboard in Serving UI
# MAGIC
# MAGIC ## Next Steps
# MAGIC
# MAGIC Serving endpoint deployed. Proceed to:
# MAGIC - **Next:** [07 - Model Monitoring]($./07_model_monitoring)
