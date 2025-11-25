# Databricks notebook source
# MAGIC %md
# MAGIC # Production Model Monitoring with Drift Detection
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/raw/main/images/product/mlops/advanced/mlops-advanced-7-monitoring.png" width="1200px">
# MAGIC
# MAGIC <!-- Tracking -->
# MAGIC <img width="1px" src="https://www.google-analytics.com/collect?v=1&gtm=GTM-NKQ8TT7&tid=UA-163989034-1&cid=555&aip=1&t=event&ec=field_demos&ea=display&dp=%2F42_field_demos%2Fml%2Fmlops_advanced%2F07_monitoring&dt=MLOPS_ADVANCED">
# MAGIC
# MAGIC ## Lakehouse Monitoring with Ground Truth Integration
# MAGIC
# MAGIC This notebook demonstrates enterprise-grade monitoring with **actual accuracy tracking**.
# MAGIC
# MAGIC **What we'll do:**
# MAGIC 1. Create inference table with predictions + ground truth labels
# MAGIC 2. Set up Lakehouse Monitor with InferenceLog profile
# MAGIC 3. Track **real** F1, Precision, Recall, Confusion Matrix
# MAGIC 4. Detect feature drift (PSI, KS-test)
# MAGIC 5. Simulate realistic drift scenarios
# MAGIC 6. Calculate production accuracy vs training accuracy
# MAGIC 7. Identify retraining triggers

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

# DBTITLE 1,Load Champion Model
import mlflow
from mlflow.tracking import MlflowClient
from pyspark.sql import functions as F
from datetime import datetime, timedelta
from databricks.sdk import WorkspaceClient
from databricks import lakehouse_monitoring as lm
import time
import pandas as pd
import numpy as np

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()
w = WorkspaceClient()

try:
    champion_version = client.get_model_version_by_alias(model_name, "Champion")
    model_uri = f"models:/{model_name}@Champion"
    print(f"Champion Model: {model_name} v{champion_version.version}")
except Exception as e:
    raise Exception(f"No Champion model found. Run notebooks 02-04b first. Error: {e}")

# COMMAND ----------

# DBTITLE 1,Create Inference Table with Ground Truth
print("Creating inference table with predictions and ground truth labels...")
print("This simulates production scoring with delayed ground truth labels.\n")

inference_source_df = spark.table(test_table)

# Simulate 7 days of production scoring
all_inference_data = []

for days_back in range(7, 0, -1):
    batch_df = inference_source_df.limit(200).toPandas()
    batch_timestamp = datetime.now() - timedelta(days=days_back)

    # Get features
    feature_cols = [col for col in batch_df.columns if col not in ['customerID', 'churn']]
    X_batch = batch_df[feature_cols]

    # Make predictions
    model = mlflow.pyfunc.load_model(model_uri)
    predictions = model.predict(X_batch)

    # Store binary predictions and probabilities
    batch_df['prediction'] = (predictions >= 0.5).astype(int)  # Binary: 0 or 1
    batch_df['prediction_proba'] = predictions  # Probabilities: 0.0 to 1.0
    batch_df['ground_truth'] = batch_df['churn'].astype(int)  # Actual outcomes
    batch_df['timestamp'] = batch_timestamp
    batch_df['model_version'] = str(champion_version.version)

    all_inference_data.append(batch_df)
    print(f"  Day -{days_back}: {len(batch_df)} predictions")

inference_pdf = pd.concat(all_inference_data, ignore_index=True)
inference_df = spark.createDataFrame(inference_pdf)

# Save to inference table
inference_df.write \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .option("delta.enableChangeDataFeed", "true") \
    .saveAsTable(inference_table)

print(f"\nInference table created: {inference_table}")
print(f"Total records: {inference_df.count():,}")

# Show sample
print("\nSample Records:")
display(inference_df.select(
    "customerID", "prediction", "prediction_proba", "ground_truth", "timestamp", "model_version"
).limit(10))

# COMMAND ----------

# DBTITLE 1,Create Lakehouse Monitor
print("Setting up Lakehouse Monitor with InferenceLog profile...\n")

# Delete existing monitor if present
try:
    w.quality_monitors.delete(table_name=inference_table)
    time.sleep(10)
    print("Existing monitor deleted")
except:
    pass

# Create monitor with InferenceLog profile
monitor_params = {
    "profile_type": lm.InferenceLog(
        timestamp_col="timestamp",
        granularities=["1 day"],
        model_id_col="model_version",
        prediction_col="prediction",  # Binary predictions (0 or 1)
        label_col="ground_truth",  # Actual labels (0 or 1)
        problem_type="classification"
    ),
    "output_schema_name": f"{catalog}.{db}",
    "schedule": lm.MonitorCronSchedule(
        quartz_cron_expression="0 0 */6 * * ?",  # Every 6 hours
        timezone_id="UTC"
    )
}

print("Monitor Configuration:")
print(f"  Prediction Column: prediction")
print(f"  Label Column: ground_truth")
print(f"  Problem Type: classification")
print(f"  Schedule: Every 6 hours")

monitor_info = lm.create_monitor(table_name=inference_table, **monitor_params)

print(f"\nMonitor created successfully!")
print(f"  Profile Table: {monitor_info.profile_metrics_table_name}")
print(f"  Drift Table: {monitor_info.drift_metrics_table_name}")
print(f"  Dashboard: {monitor_info.dashboard_id if monitor_info.dashboard_id else 'N/A'}")

# COMMAND ----------

# DBTITLE 1,Wait for Monitor to Become Active
print("Waiting for monitor to become active...")

max_wait = 300  # 5 minutes
start = time.time()
status = None

while time.time() - start < max_wait:
    try:
        monitor_status = w.quality_monitors.get(table_name=inference_table)
        status = monitor_status.status

        if status == "MONITOR_STATUS_ACTIVE":
            print(f"\nMonitor is ACTIVE!")
            break
        elif status == "MONITOR_STATUS_ERROR":
            print(f"\nMonitor creation failed")
            break

        elapsed = int(time.time() - start)
        print(f"  Status: {status} ({elapsed}s)", end='\r')
        time.sleep(10)
    except:
        time.sleep(10)

if status == "MONITOR_STATUS_ACTIVE":
    print("\nTriggering initial refresh...")
    refresh = w.quality_monitors.run_refresh(table_name=inference_table)
    print(f"Refresh triggered: {refresh.refresh_id}")
    print("Refresh will take 5-10 minutes...")

# COMMAND ----------

# DBTITLE 1,Check Refresh Status
print("Checking refresh status...")

refreshes_response = w.quality_monitors.list_refreshes(table_name=inference_table)
refreshes = list(refreshes_response.refreshes)

if len(refreshes) > 0:
    latest = refreshes[0]
    print(f"\nLatest Refresh:")
    print(f"  ID: {latest.refresh_id}")
    print(f"  State: {latest.state}")

    if latest.state == "SUCCESS":
        print("\nRefresh completed successfully!")
    elif latest.state in ["PENDING", "RUNNING"]:
        print("\nStill processing... wait a few more minutes and re-run this cell")
    elif latest.state == "FAILED":
        print(f"\nRefresh failed: {latest.message if hasattr(latest, 'message') else 'Unknown error'}")
else:
    print("\nNo refresh found yet - monitor may still be initializing")

# COMMAND ----------

# DBTITLE 1,Query Production Metrics
print("\n" + "=" * 60)
print("PRODUCTION MODEL PERFORMANCE METRICS")
print("=" * 60)

profile_table = monitor_info.profile_metrics_table_name

metrics_query = f"""
SELECT
    window.start as window_start,
    window.end as window_end,
    model_version,
    accuracy_score,
    precision.weighted as precision_weighted,
    recall.weighted as recall_weighted,
    f1_score.weighted as f1_weighted,
    roc_auc_score.one_vs_one.weighted as roc_auc,
    confusion_matrix,
    log_loss
FROM {profile_table}
WHERE column_name = ':table'
ORDER BY window_start DESC
"""

try:
    metrics_df = spark.sql(metrics_query)

    if metrics_df.count() > 0:
        print(f"\nRetrieved {metrics_df.count()} metric records\n")
        display(metrics_df)

        latest = metrics_df.first()

        print("\n" + "=" * 60)
        print("LATEST PRODUCTION PERFORMANCE")
        print("=" * 60)
        print(f"Time Window: {latest.window_start} to {latest.window_end}")
        print(f"Model Version: {latest.model_version}\n")

        if latest.accuracy_score is not None:
            print("Classification Metrics:")
            print(f"  Accuracy:  {latest.accuracy_score:.4f}")
            print(f"  Precision: {latest.precision_weighted:.4f}")
            print(f"  Recall:    {latest.recall_weighted:.4f}")
            print(f"  F1 Score:  {latest.f1_weighted:.4f}")
            if latest.roc_auc is not None:
                print(f"  ROC AUC:   {latest.roc_auc:.4f}")

            # Parse confusion matrix
            if latest.confusion_matrix is not None and len(latest.confusion_matrix) > 0:
                cm_dict = {}
                for row in latest.confusion_matrix:
                    row_dict = row.asDict()
                    key = f"{row_dict['prediction']}_{row_dict['label']}"
                    cm_dict[key] = row_dict['count']

                tp = cm_dict.get('1_1', 0)
                fp = cm_dict.get('1_0', 0)
                fn = cm_dict.get('0_1', 0)
                tn = cm_dict.get('0_0', 0)

                print(f"\nConfusion Matrix:")
                print(f"  TP: {tp:4d}  FP: {fp:4d}")
                print(f"  FN: {fn:4d}  TN: {tn:4d}")

                if (tp + fn) > 0:
                    fnr = fn / (tp + fn)
                    print(f"\n  False Negative Rate: {fnr:.2%}")

            print("=" * 60)
        else:
            print("Metrics still processing - re-run after refresh completes")

    else:
        print("No metrics found yet - wait for refresh to complete")

except Exception as e:
    print(f"Error querying metrics: {e}")

# COMMAND ----------

# DBTITLE 1,Simulate Feature Drift
print("Simulating progressive feature drift over 14 days...\n")

base_df = spark.table(test_table).toPandas()
feature_cols = [col for col in base_df.columns if col not in ['customerID', 'churn']]
original_dtypes = base_df[feature_cols].dtypes.to_dict()

all_drift_data = []

for days_back in range(14, 0, -1):
    batch_df = base_df.sample(n=100, replace=True, random_state=days_back).copy()
    batch_timestamp = datetime.now() - timedelta(days=days_back)

    drift_factor = (14 - days_back) / 14  # 0 to 1

    # Apply progressive drift
    batch_df['MonthlyCharges'] = batch_df['MonthlyCharges'] * (1 + drift_factor * 0.3)  # +30% increase
    batch_df['tenure'] = batch_df['tenure'] * (1 - drift_factor * 0.25)  # -25% decrease

    # Shift to month-to-month contracts
    if drift_factor > 0.5:
        mask = np.random.random(len(batch_df)) < (drift_factor * 0.3)
        batch_df.loc[mask, 'Contract_Month_to_month'] = 1
        batch_df.loc[mask, 'Contract_One_year'] = 0
        batch_df.loc[mask, 'Contract_Two_year'] = 0

    # Restore dtypes
    for col, dtype in original_dtypes.items():
        if col in batch_df.columns:
            batch_df[col] = batch_df[col].astype(dtype)

    # Predict
    X_batch = batch_df[feature_cols]
    predictions = model.predict(X_batch)

    batch_df['prediction'] = (predictions >= 0.5).astype(int)
    batch_df['prediction_proba'] = predictions
    batch_df['ground_truth'] = batch_df['churn'].astype(int)
    batch_df['timestamp'] = batch_timestamp
    batch_df['model_version'] = str(champion_version.version)

    all_drift_data.append(batch_df)

    if days_back % 4 == 0:
        print(f"Day -{days_back:2d}: MonthlyCharges=${batch_df['MonthlyCharges'].mean():.2f}, "
              f"Tenure={batch_df['tenure'].mean():.1f}mo, "
              f"Churn={batch_df['prediction'].mean():.1%}")

# Append drift data
drift_pdf = pd.concat(all_drift_data, ignore_index=True)
drift_df = spark.createDataFrame(drift_pdf)
drift_df.write.mode("append").saveAsTable(inference_table)

print(f"\nAdded {drift_df.count():,} records with progressive drift")

# COMMAND ----------

# DBTITLE 1,Refresh Monitor to Detect Drift
print("Triggering monitor refresh to detect drift...")

refresh = w.quality_monitors.run_refresh(table_name=inference_table)
print(f"Refresh triggered: {refresh.refresh_id}")
print("\nWaiting for refresh to complete (this may take 5-10 minutes)...")

max_wait = 900  # 15 minutes
start = time.time()

while time.time() - start < max_wait:
    try:
        refresh_status = w.quality_monitors.get_refresh(
            table_name=inference_table,
            refresh_id=refresh.refresh_id
        )

        if refresh_status.state == "SUCCESS":
            duration = int(time.time() - start)
            print(f"\nRefresh complete! ({duration}s)")
            break
        elif refresh_status.state == "FAILED":
            print(f"\nRefresh failed")
            break

        elapsed = int(time.time() - start)
        print(f"  Status: {refresh_status.state} ({elapsed}s)", end='\r')
        time.sleep(15)
    except:
        time.sleep(15)

# COMMAND ----------

# DBTITLE 1,View Drift Detection Results
print("\n" + "=" * 60)
print("DRIFT DETECTION RESULTS")
print("=" * 60)

drift_table = monitor_info.drift_metrics_table_name

drift_query = f"""
SELECT
    window.start as window_start,
    column_name,
    ks_test.pvalue as ks_pvalue,
    wasserstein_distance
FROM {drift_table}
WHERE column_name IN ('MonthlyCharges', 'tenure', 'Contract_Month_to_month', 'prediction')
  AND drift_type = 'CONSECUTIVE'
ORDER BY window_start DESC, column_name
"""

try:
    drift_df = spark.sql(drift_query)

    if drift_df.count() > 0:
        print(f"\nDrift metrics found: {drift_df.count()} records\n")
        display(drift_df)

        # Significant drift (p-value < 0.05)
        significant_drift = drift_df.filter(F.col("ks_pvalue") < 0.05)

        if significant_drift.count() > 0:
            print("\n" + "=" * 60)
            print("SIGNIFICANT DRIFT DETECTED")
            print("=" * 60)
            display(significant_drift)

            print("\nInterpretation:")
            print("  - p-value < 0.05 = statistically significant drift")
            print("  - Wasserstein distance = magnitude of distribution change")
            print("  - Higher values = more severe drift")
        else:
            print("\nNo significant drift detected (all p-values > 0.05)")

    else:
        print("No drift metrics found yet - refresh may still be processing")

except Exception as e:
    print(f"Error querying drift: {e}")

# COMMAND ----------

# DBTITLE 1,Compare Production vs Training Performance
print("\n" + "=" * 60)
print("PRODUCTION vs TRAINING COMPARISON")
print("=" * 60)

# Get training metrics from Champion model run
champion_run = mlflow.get_run(champion_version.run_id)
training_metrics = champion_run.data.metrics

# Get latest production metrics
try:
    prod_metrics = spark.sql(metrics_query).first()

    if prod_metrics and prod_metrics.f1_weighted is not None:
        print("\nMetric Comparison:")
        print(f"{'Metric':<15} {'Training':<12} {'Production':<12} {'Delta':<12}")
        print("-" * 60)

        for metric in ['f1_score', 'accuracy', 'precision', 'recall']:
            train_val = training_metrics.get(metric, 0)
            prod_val = getattr(prod_metrics, metric + '_weighted' if metric != 'accuracy_score' else metric, 0)
            if prod_val is None:
                prod_val = getattr(prod_metrics, metric.replace('_score', '_weighted'), 0)
            delta = prod_val - train_val if prod_val else 0

            print(f"{metric:<15} {train_val:<12.4f} {prod_val:<12.4f} {delta:+.4f}")

        f1_degradation = training_metrics.get('f1_score', 0) - (prod_metrics.f1_weighted or 0)

        print("\n" + "=" * 60)
        if f1_degradation > 0.05:
            print("RETRAINING RECOMMENDED")
            print("=" * 60)
            print(f"  F1 Score degraded by {f1_degradation:.1%}")
            print("  Trigger: Production F1 < Training F1 by >5%")
        else:
            print("MODEL PERFORMANCE STABLE")
            print("=" * 60)

    else:
        print("Production metrics not available yet")

except Exception as e:
    print(f"Error comparing metrics: {e}")

# COMMAND ----------

# DBTITLE 1,Save Monitoring Results
dbutils.jobs.taskValues.set(key="monitoring_table", value=inference_table)
dbutils.jobs.taskValues.set(key="profile_table", value=profile_table)
dbutils.jobs.taskValues.set(key="drift_table", value=drift_table)

print("\nMonitoring setup complete")
print(f"  Inference Table: {inference_table}")
print(f"  Profile Metrics: {profile_table}")
print(f"  Drift Metrics: {drift_table}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC
# MAGIC Production monitoring is active. Proceed to:
# MAGIC - **Next:** [08 - Automated Retraining]($./08_automated_retraining)
