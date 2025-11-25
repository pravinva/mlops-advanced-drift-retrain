# Databricks notebook source
# MAGIC %md
# MAGIC # Real-Time Model Serving

# COMMAND ----------

# MAGIC %run ./notebook-00-setup

# COMMAND ----------

# MAGIC %pip install databricks-sdk --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Get Champion Model
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

champion_version = client.get_model_version_by_alias(model_name, "Champion")
print(f"Champion: {model_name} v{champion_version.version}")

# COMMAND ----------

# DBTITLE 1,Create Serving Endpoint
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ServedEntityInput, EndpointCoreConfigInput, AutoCaptureConfigInput

w = WorkspaceClient()

endpoint_name = f"churn-prediction-{user.split('@')[0]}"[:63]
endpoint_name = endpoint_name.replace('_', '-').replace('.', '-')

print(f"Creating endpoint: {endpoint_name}")

# Check if exists
try:
    existing_endpoint = w.serving_endpoints.get(name=endpoint_name)
    endpoint_exists = True
    print("Endpoint exists, will update")
except:
    endpoint_exists = False
    print("Creating new endpoint")

# Configuration
endpoint_config = EndpointCoreConfigInput(
    name=endpoint_name,
    served_entities=[
        ServedEntityInput(
            entity_name=model_name,
            entity_version=str(champion_version.version),
            workload_size="Small",
            scale_to_zero_enabled=True,
            environment_vars={"MODEL_VERSION": str(champion_version.version)}
        )
    ],
    auto_capture_config=AutoCaptureConfigInput(
        catalog_name=catalog,
        schema_name=db,
        table_name_prefix="churn_endpoint",
        enabled=True
    )
)

# COMMAND ----------

# DBTITLE 1,Deploy Endpoint
if endpoint_exists:
    w.serving_endpoints.update_config(
        name=endpoint_name,
        served_entities=endpoint_config.served_entities,
        auto_capture_config=endpoint_config.auto_capture_config
    )
else:
    w.serving_endpoints.create(name=endpoint_name, config=endpoint_config)

print(f"✅ Endpoint deployment initiated: {endpoint_name}")

# COMMAND ----------

# DBTITLE 1,Wait for Ready
import time

print("Waiting for endpoint (may take 5-15 minutes)...")

max_wait = 20 * 60
start = time.time()

while time.time() - start < max_wait:
    endpoint = w.serving_endpoints.get(name=endpoint_name)
    if endpoint.state.ready == "READY":
        print("\n✅ Endpoint is READY!")
        break
    elapsed = int(time.time() - start)
    print(f"Status: {endpoint.state.ready} ({elapsed}s)", end='\r')
    time.sleep(30)

# COMMAND ----------

# DBTITLE 1,Get Endpoint Info
workspace_url = spark.conf.get("spark.databricks.workspaceUrl")
endpoint_name = "churn-prediction-pravin-varma"
endpoint_url = f"https://{workspace_url}/serving-endpoints/{endpoint_name}/invocations"

print("\n" + "="*60)
print("SERVING ENDPOINT")
print("="*60)
print(f"Name: {endpoint_name}")
print(f"Status: {endpoint.state.ready}")
print(f"Model: {model_name} v{champion_version.version}")
print(f"URL: {endpoint_url}")
print("="*60)

# COMMAND ----------

# DBTITLE 1,Test Endpoint
import requests
import json

token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

test_df = spark.table(f"{catalog}.{db}.churn_features_test").limit(1)
test_pdf = test_df.toPandas()

feature_cols = [col for col in test_pdf.columns if col not in ['customerID', 'churn']]
test_input = test_pdf[feature_cols].to_dict(orient='records')[0]

headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
}

payload = {"dataframe_records": [test_input]}

print("Testing endpoint...")
response = requests.post(endpoint_url, headers=headers, json=payload)

if response.status_code == 200:
    prediction = response.json()
    pred_value = prediction['predictions'][0]
    print(f"\n✅ Prediction successful!")
    print(f"Churn Probability: {pred_value:.4f}")
    print(f"Risk Level: {'High' if pred_value >= 0.7 else 'Medium' if pred_value >= 0.4 else 'Low'}")
else:
    print(f"\n❌ Failed: {response.status_code}")
    print(response.text)

# COMMAND ----------

# DBTITLE 1,Latency Test
import numpy as np

latencies = []
single_record = test_pdf[feature_cols].to_dict(orient='records')[0]

# Warm-up
for _ in range(3):
    requests.post(endpoint_url, headers=headers, json={"dataframe_records": [single_record]})

# Measure
for _ in range(20):
    start = time.time()
    requests.post(endpoint_url, headers=headers, json={"dataframe_records": [single_record]})
    latencies.append((time.time() - start) * 1000)

print("\nLatency Statistics:")
print(f"  Mean: {np.mean(latencies):.2f}ms")
print(f"  P50: {np.percentile(latencies, 50):.2f}ms")
print(f"  P95: {np.percentile(latencies, 95):.2f}ms")
print(f"  P99: {np.percentile(latencies, 99):.2f}ms")

# COMMAND ----------

# DBTITLE 1,API Integration Example
api_example = f"""
# Python API Integration Example

import requests

ENDPOINT_URL = "{endpoint_url}"
DATABRICKS_TOKEN = "YOUR_TOKEN"

headers = {{
    "Authorization": f"Bearer {{DATABRICKS_TOKEN}}",
    "Content-Type": "application/json"
}}

customer_data = {{
    "tenure": 12,
    "MonthlyCharges": 70.50,
    # ... include all features
}}

response = requests.post(
    ENDPOINT_URL,
    headers=headers,
    json={{"dataframe_records": [customer_data]}}
)

if response.status_code == 200:
    churn_prob = response.json()['predictions'][0]
    print(f"Churn Probability: {{churn_prob:.2%}}")
"""

print(api_example)

# COMMAND ----------

# DBTITLE 1,Save Configuration
dbutils.jobs.taskValues.set(key="endpoint_name", value=endpoint_name)
dbutils.jobs.taskValues.set(key="endpoint_url", value=endpoint_url)

with mlflow.start_run(run_name="model_serving_deployment"):
    mlflow.log_param("endpoint_name", endpoint_name)
    mlflow.log_param("model_version", champion_version.version)
    if len(latencies) > 0:
        mlflow.log_metric("avg_latency_ms", np.mean(latencies))
        mlflow.log_metric("p95_latency_ms", np.percentile(latencies, 95))

print("\n✅ Endpoint deployed and tested")