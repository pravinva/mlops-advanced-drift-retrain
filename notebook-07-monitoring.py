# Databricks notebook source
# MAGIC %md
# MAGIC # Model Monitoring
# MAGIC
# MAGIC ** GENERATES F1, PRECISION, RECALL METRICS**

# COMMAND ----------

# MAGIC %pip install lightgbm

# COMMAND ----------

# MAGIC %run ./notebook-00-setup

# COMMAND ----------

# DBTITLE 1,Setup
model_name = "art_mlops.mlops_churn_demo.churn_model"  # Update if your model name is different
from databricks.sdk import WorkspaceClient
from databricks import lakehouse_monitoring as lm
import mlflow
from mlflow.tracking import MlflowClient
from pyspark.sql import functions as F
from datetime import datetime, timedelta
import time
import pandas as pd

w = WorkspaceClient()
mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

champion_version = client.get_model_version_by_alias(model_name, "Champion")
model_uri = f"models:/{model_name}@Champion"

print(f"Setting up monitoring for: {model_name} v{champion_version.version}")

# COMMAND ----------

# COMMAND ----------

# DBTITLE 1,Create Inference Table with Ground Truth - FIXED
print("Creating inference table with predictions and ground truth...")

inference_source_df = spark.table(f"{catalog}.{db}.churn_features_test")

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
    
    # CRITICAL FIX: Use binary predictions for confusion matrix
    batch_df['prediction'] = (predictions >= 0.5).astype(int)  # Binary: 0 or 1
    batch_df['prediction_proba'] = predictions  # Keep probabilities too
    batch_df['ground_truth'] = batch_df['churn'].astype(int)  # Ensure int type
    batch_df['timestamp'] = batch_timestamp
    batch_df['model_version'] = str(champion_version.version)
    
    all_inference_data.append(batch_df)
    print(f"  Batch for {batch_timestamp.date()}: {len(batch_df)} records")

inference_pdf = pd.concat(all_inference_data, ignore_index=True)
inference_df = spark.createDataFrame(inference_pdf)

inference_table = f"{catalog}.{db}.churn_inference_with_labels"

inference_df.write \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .option("delta.enableChangeDataFeed", "true") \
    .saveAsTable(inference_table)

print(f"\n✅ Inference table created: {inference_table}")
print(f"   Total records: {inference_df.count()}")

# Verify the data
print("\nSample data:")
display(inference_df.select(
    "customerID", 
    "prediction",  # Should be 0 or 1
    "prediction_proba",  # Should be 0.0 to 1.0
    "ground_truth",  # Should be 0 or 1
    "timestamp",
    "model_version"
).limit(10))

# Verify prediction and ground_truth match
verify = inference_df.select([
    F.countDistinct("prediction").alias("unique_predictions"),
    F.countDistinct("ground_truth").alias("unique_labels"),
    F.min("prediction").alias("min_pred"),
    F.max("prediction").alias("max_pred"),
    F.min("ground_truth").alias("min_label"),
    F.max("ground_truth").alias("max_label")
]).collect()[0]

print("\nData Validation:")
print(f"  Unique predictions: {verify.unique_predictions} (should be 2: [0, 1])")
print(f"  Unique labels: {verify.unique_labels} (should be 2: [0, 1])")
print(f"  Prediction range: [{verify.min_pred}, {verify.max_pred}]")
print(f"  Label range: [{verify.min_label}, {verify.max_label}]")

if verify.unique_predictions == 2 and verify.max_pred == 1:
    print("\n✅ Predictions are binary - confusion matrix will work!")
else:
    print("\n❌ WARNING: Predictions may not be binary!")

# COMMAND ----------

# DBTITLE 1,Verify Ground Truth
check = inference_df.select([
    F.count("*").alias("total"),
    F.count("ground_truth").alias("non_null_gt"),
    F.avg("ground_truth").alias("churn_rate")
]).collect()[0]

print("\nGround Truth Validation:")
print(f"  Total Records: {check.total}")
print(f"  Non-null Ground Truth: {check.non_null_gt}")
print(f"  Churn Rate: {check.churn_rate:.2%}")

if check.non_null_gt == check.total:
    print("\n✅ Ground truth complete - metrics will work!")
else:
    print("\n❌ WARNING: Missing ground truth!")

# COMMAND ----------

# DBTITLE 1,Create Monitor with InferenceLog
print("\nCreating Lakehouse Monitor...")

# Delete existing
try:
    w.quality_monitors.delete(table_name=inference_table)
    time.sleep(10)
    print("Deleted existing monitor")
except:
    pass

# CRITICAL CONFIGURATION for F1, Precision, Recall
monitor_params = {
    "profile_type": lm.InferenceLog(
        timestamp_col="timestamp",
        granularities=["1 day"],
        model_id_col="model_version",
        prediction_col="prediction",
        label_col="ground_truth",  # MUST have actual labels
        problem_type="classification"
    ),
    "output_schema_name": f"{catalog}.{db}",
    "schedule": lm.MonitorCronSchedule(
        quartz_cron_expression="0 0 */6 * * ?",
        timezone_id="UTC"
    )
}

print("\nMonitor Config:")
print(f"  Prediction Column: prediction")
print(f"  Label Column: ground_truth")
print(f"  Problem Type: classification")

monitor_info = lm.create_monitor(table_name=inference_table, **monitor_params)

print(f"\n✅ Monitor created!")
print(f"   Profile Table: {monitor_info.profile_metrics_table_name}")
print(f"   Drift Table: {monitor_info.drift_metrics_table_name}")
print(f"   Dashboard: {monitor_info.dashboard_id}")

# COMMAND ----------

# COMMAND ----------

# DBTITLE 1,Create Monitor and Wait for Ready
print("\nDeleting and recreating monitor...")

try:
    w.quality_monitors.delete(table_name=inference_table)
    time.sleep(10)
    print("✅ Old monitor deleted")
except:
    pass

# Create monitor
monitor_params = {
    "profile_type": lm.InferenceLog(
        timestamp_col="timestamp",
        granularities=["1 day"],
        model_id_col="model_version",
        prediction_col="prediction",  # Binary predictions (0 or 1)
        label_col="ground_truth",  # Binary labels (0 or 1)
        problem_type="classification"
    ),
    "output_schema_name": f"{catalog}.{db}",
    "schedule": lm.MonitorCronSchedule(
        quartz_cron_expression="0 0 */6 * * ?",
        timezone_id="UTC"
    )
}

print("Creating monitor...")
monitor_info = lm.create_monitor(table_name=inference_table, **monitor_params)

print(f"✅ Monitor created!")
print(f"   Profile Table: {monitor_info.profile_metrics_table_name}")

# CRITICAL: Wait for monitor to become ACTIVE before triggering refresh
print("\nWaiting for monitor to be ready...")

max_wait = 300  # 5 minutes
start = time.time()

while time.time() - start < max_wait:
    try:
        monitor_status = w.quality_monitors.get(table_name=inference_table)
        status = monitor_status.status
        
        print(f"  Monitor status: {status}", end='\r')
        
        if status == "MONITOR_STATUS_ACTIVE":
            print(f"\n✅ Monitor is ACTIVE!")
            break
        elif status == "MONITOR_STATUS_ERROR":
            print(f"\n❌ Monitor creation failed")
            break
        
        time.sleep(10)
    except:
        time.sleep(10)

# Now trigger refresh
if status == "MONITOR_STATUS_ACTIVE":
    print("\nTriggering refresh...")
    refresh = w.quality_monitors.run_refresh(table_name=inference_table)
    print(f"✅ Refresh triggered: {refresh.refresh_id}")
    print("\nRefresh will take 10-15 minutes")
    print("Re-run the 'Check Refresh Status' cell to monitor progress")
else:
    print(f"\n⚠️ Monitor status: {status}")
    print("Cannot trigger refresh yet - monitor is not active")

# COMMAND ----------

# COMMAND ----------

# DBTITLE 1,Check Refresh Status
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

# Get all refreshes for the table
refreshes_response = w.quality_monitors.list_refreshes(table_name=inference_table)
refreshes = refreshes_response.refreshes

print("Recent Refreshes:")
print("="*60)
for refresh in list(refreshes):
    print(f"Refresh ID: {refresh.refresh_id}")
    print(f"  State: {refresh.state}")
    print(f"  Start Time: {refresh.start_time_ms}")
    if refresh.end_time_ms:
        duration = (refresh.end_time_ms - refresh.start_time_ms) / 1000
        print(f"  Duration: {duration:.0f}s")
    print()

# Check latest refresh
try:
    latest = list(refreshes)[0]
    print(f"\nLatest Status: {latest.state}")
    
    if latest.state == "SUCCESS":
        print("✅ Refresh completed successfully!")
    elif latest.state == "PENDING" or latest.state == "RUNNING":
        print("⏳ Still processing... wait a few more minutes")
    elif latest.state == "FAILED":
        print(f"❌ Refresh failed: {latest.message}")
except:
    print("No refresh status found")

# COMMAND ----------

# DBTITLE 1,Query Metrics - CHECK F1, PRECISION, RECALL
# COMMAND ----------

# DBTITLE 1,Query Metrics - FIXED Row Access
print("\n" + "="*60)
print("MODEL PERFORMANCE METRICS")
print("="*60)

profile_table = monitor_info.profile_metrics_table_name

metrics_query = f"""
SELECT 
    window.start as window_start,
    window.end as window_end,
    model_version,
    granularity,
    column_name,
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
        print(f"\n✅ Retrieved {metrics_df.count()} metric records\n")
        display(metrics_df)
        
        latest = metrics_df.first()
        
        print("\n" + "="*60)
        print("LATEST MODEL PERFORMANCE")
        print("="*60)
        print(f"Time Window: {latest.window_start} to {latest.window_end}")
        print(f"Model Version: {latest.model_version}")
        print()
        
        if latest.accuracy_score is not None:
            print("✅ CLASSIFICATION METRICS:")
            print(f"  Accuracy:        {latest.accuracy_score:.4f}")
            print(f"  Precision (wtd): {latest.precision_weighted:.4f}")
            print(f"  Recall (wtd):    {latest.recall_weighted:.4f}")
            print(f"  F1 Score (wtd):  {latest.f1_weighted:.4f}")
            
            if latest.roc_auc is not None:
                print(f"  ROC AUC:         {latest.roc_auc:.4f}")
            
            if latest.log_loss is not None:
                print(f"  Log Loss:        {latest.log_loss:.4f}")
            
            # FIXED: Parse confusion matrix - use asDict() for Row objects
            if latest.confusion_matrix is not None and len(latest.confusion_matrix) > 0:
                print(f"\nConfusion Matrix:")
                
                # Build lookup dict
                cm_dict = {}
                for row in latest.confusion_matrix:
                    # Convert Row to dict
                    row_dict = row.asDict()
                    pred = row_dict['prediction']
                    label = row_dict['label']
                    count_val = row_dict['count']
                    
                    key = f"{pred}_{label}"
                    cm_dict[key] = count_val
                
                # Extract values
                tp = cm_dict.get('1_1', 0)
                fp = cm_dict.get('1_0', 0)
                fn = cm_dict.get('0_1', 0)
                tn = cm_dict.get('0_0', 0)
                
                print(f"  True Positives:  {tp}")
                print(f"  False Positives: {fp}")
                print(f"  True Negatives:  {tn}")
                print(f"  False Negatives: {fn}")
                
                # Calculate rates
                total = tp + fp + tn + fn
                print(f"\n  Total Predictions: {total}")
                
                if (tp + fn) > 0:
                    tpr = tp / (tp + fn)
                    fnr = fn / (tp + fn)
                    print(f"  Recall (TPR):     {tpr:.2%}")
                    print(f"  False Neg Rate:   {fnr:.2%}")
                
                if (fp + tn) > 0:
                    fpr = fp / (fp + tn)
                    print(f"  False Pos Rate:   {fpr:.2%}")
                
                if (tp + fp) > 0:
                    prec = tp / (tp + fp)
                    print(f"  Precision:        {prec:.2%}")
            
            status = "✅ SUCCESS"
        else:
            print("❌ Metrics are NULL")
            status = "⚠️ PENDING"
        
        print("="*60)
    else:
        print("⚠️ No metrics found")
        status = "⚠️ PENDING"
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    status = "❌ ERROR"

print(f"\nStatus: {status}")

# COMMAND ----------

# COMMAND ----------

# DBTITLE 1,Check Dashboard Status
workspace_url = spark.conf.get("spark.databricks.workspaceUrl")

print("Checking dashboard status...")
print("="*60)

# Check if dashboard was created
dashboard_id = monitor_info.dashboard_id

if dashboard_id and dashboard_id != "None":
    dashboard_url = f"https://{workspace_url}/sql/dashboardsv3/{dashboard_id}"
    print(f"✅ Dashboard ID: {dashboard_id}")
    print(f"URL: {dashboard_url}")
    
    # Test if accessible
    displayHTML(f'''
    <div style="padding:20px; background:#f0f8ff; border:2px solid #4CAF50; border-radius:5px;">
        <h3>📊 Monitoring Dashboard</h3>
        <a href="{dashboard_url}" target="_blank" 
           style="background:#4CAF50; color:white; padding:10px 20px; 
                  text-decoration:none; border-radius:5px; display:inline-block;">
            Open Dashboard
        </a>
    </div>
    ''')
else:
    print("⚠️ No auto-generated dashboard found")
    print("\nAlternative: Query metrics directly using SQL")
    print("="*60)
    
    # Show how to query metrics directly
    query_example = f"""
-- Query metrics directly in Databricks SQL or notebook:

SELECT 
    window.start as time_window,
    model_version,
    accuracy_score,
    precision.weighted as precision,
    recall.weighted as recall,
    f1_score.weighted as f1_score,
    log_loss
FROM {profile_table}
WHERE column_name = ':table'
ORDER BY window.start DESC
"""
    
    print("\nSQL Query to View Metrics:")
    print(query_example)
    
    print("\n✅ Metrics are in the profile table - you can query them anytime!")
    print(f"   Table: {profile_table}")

print("="*60)

# COMMAND ----------

# COMMAND ----------

# DBTITLE 1,Create Custom Monitoring Dashboard (Optional)
# Since auto-dashboard may not exist, create a simple query-based view

print("Creating custom monitoring view...")

# Query for dashboard
dashboard_query = f"""
SELECT 
    window.start as metric_date,
    model_version,
    accuracy_score,
    precision.weighted as precision,
    recall.weighted as recall,
    f1_score.weighted as f1_score,
    roc_auc_score.one_vs_one.weighted as roc_auc,
    log_loss,
    count as sample_count
FROM {profile_table}
WHERE column_name = ':table'
ORDER BY window.start DESC
"""

# Display metrics over time
print("\n📊 Model Performance Over Time:")
monitoring_view = spark.sql(dashboard_query)
display(monitoring_view)

# Summary of latest metrics
print("\n✅ Use this query in Databricks SQL to create visualizations:")
print(dashboard_query)

# COMMAND ----------

# COMMAND ----------

# DBTITLE 1,Simulate Drift - With Dtype Fix
print("Creating data with drift...")
print("="*60)

from datetime import timedelta
import pandas as pd
import numpy as np

base_df = spark.table(f"{catalog}.{db}.churn_features_test").toPandas()
feature_cols = [col for col in base_df.columns if col not in ['customerID', 'churn']]

# Save original dtypes BEFORE any modifications
original_dtypes = base_df[feature_cols].dtypes.to_dict()

def restore_dtypes(df, dtype_map):
    for col, dtype in dtype_map.items():
        if col in df.columns:
            if dtype in ['int32', 'int64']:
                df[col] = df[col].astype(dtype)
            elif dtype in ['float64', 'float32']:
                df[col] = df[col].astype('float64')
    return df

# Load model once
model = mlflow.pyfunc.load_model(model_uri)

all_drift_data = []

for days_back in range(14, 0, -1):
    batch_df = base_df.sample(n=100, replace=True, random_state=days_back).copy()
    batch_timestamp = datetime.now() - timedelta(days=days_back)
    
    drift_factor = (14 - days_back) / 14
    
    # Apply drift
    batch_df['MonthlyCharges'] = batch_df['MonthlyCharges'] * (1 + drift_factor * 0.3)
    batch_df['tenure'] = batch_df['tenure'] * (1 - drift_factor * 0.25)
    
    if drift_factor > 0.5:
        month_mask = np.random.random(len(batch_df)) < (drift_factor * 0.3)
        batch_df.loc[month_mask, 'Contract_Month_to_month'] = 1
        batch_df.loc[month_mask, 'Contract_One_year'] = 0
        batch_df.loc[month_mask, 'Contract_Two_year'] = 0
    
    # CRITICAL FIX: Restore original dtypes after modifications
    X_batch = batch_df[feature_cols].copy()
    X_batch = restore_dtypes(X_batch, original_dtypes)
    # Also restore dtypes in batch_df for all columns
    batch_df = restore_dtypes(batch_df, original_dtypes)
    
    # Predict
    predictions = model.predict(X_batch)
    
    batch_df['prediction'] = (predictions >= 0.5).astype(int)
    batch_df['prediction_proba'] = predictions
    batch_df['ground_truth'] = batch_df['churn'].astype(int)
    batch_df['timestamp'] = batch_timestamp
    batch_df['model_version'] = str(champion_version.version)
    
    all_drift_data.append(batch_df)
    
    if days_back % 3 == 0:
        avg_monthly = batch_df['MonthlyCharges'].mean()
        avg_tenure = batch_df['tenure'].mean()
        churn_rate = batch_df['prediction'].mean()
        print(f"Day -{days_back:2d}: MonthlyCharges=${avg_monthly:.2f}, "
              f"Tenure={avg_tenure:.1f}, Churn={churn_rate:.1%}")

# Combine
drift_pdf = pd.concat(all_drift_data, ignore_index=True)

# Print dtypes before writing
print("\nDtypes before writing to Delta table:")
print(drift_pdf.dtypes)

# Ensure 'tenure' column matches original dtype
drift_pdf['tenure'] = drift_pdf['tenure'].astype(original_dtypes['tenure'])

drift_df = spark.createDataFrame(drift_pdf)

# Append
drift_df.write.mode("append").saveAsTable(inference_table)

print(f"\n✅ Added {drift_df.count()} records with progressive drift")
print("="*60)

# COMMAND ----------

# COMMAND ----------

# DBTITLE 1,Refresh Monitor to Detect Drift
print("Triggering monitor refresh to detect drift...")

refresh = w.quality_monitors.run_refresh(table_name=inference_table)
print(f"✅ Refresh triggered: {refresh.refresh_id}")

print("\nWaiting for refresh to complete...")

max_wait = 900  # 15 minutes
start = time.time()

while time.time() - start < max_wait:
    try:
        status = w.quality_monitors.get_refresh(
            table_name=inference_table,
            refresh_id=refresh.refresh_id
        )
        
        if status.state == "SUCCESS":
            duration = (time.time() - start)
            print(f"\n✅ Refresh complete! ({duration:.0f}s)")
            break
        elif status.state == "FAILED":
            print(f"\n❌ Refresh failed")
            break
        
        elapsed = int(time.time() - start)
        print(f"  Status: {status.state} ({elapsed}s)", end='\r')
        time.sleep(15)
    except:
        time.sleep(15)

# COMMAND ----------

# COMMAND ----------

# DBTITLE 1,View Drift Metrics
print("\n" + "="*60)
print("DRIFT DETECTION RESULTS")
print("="*60)

drift_table = monitor_info.drift_metrics_table_name

# Query drift for key features
drift_query = f"""
SELECT 
    window.start as window_start,
    window.end as window_end,
    column_name,
    drift_type,
    chi_squared_test.pvalue as chi_squared_pvalue,
    ks_test.pvalue as ks_pvalue,
    wasserstein_distance
FROM {drift_table}
WHERE column_name IN ('MonthlyCharges', 'tenure', 'Contract_Month_to_month', 'prediction')
  AND drift_type = 'CONSECUTIVE'  -- Compare to previous time window
ORDER BY window_start DESC, column_name
"""

drift_df = spark.sql(drift_query)

if drift_df.count() > 0:
    print(f"\n✅ Drift metrics found: {drift_df.count()} records\n")
    display(drift_df)
    
    # Show significant drift (p-value < 0.05 means significant change)
    significant_drift = drift_df.filter(
        (F.col("ks_pvalue") < 0.05) | (F.col("chi_squared_pvalue") < 0.05)
    )
    
    if significant_drift.count() > 0:
        print("\n⚠️ SIGNIFICANT DRIFT DETECTED:")
        print("="*60)
        display(significant_drift.select(
            "window_start", 
            "column_name", 
            "ks_pvalue",
            "wasserstein_distance"
        ))
        
        print("\nInterpretation:")
        print("  - p-value < 0.05 indicates statistically significant drift")
        print("  - Wasserstein distance shows magnitude of distribution change")
        print("  - Higher values = more drift")
    else:
        print("\n✅ No significant drift detected (all p-values > 0.05)")
else:
    print("⚠️ No drift metrics found - refresh may still be processing")

# COMMAND ----------

# COMMAND ----------

# DBTITLE 1,Visualize Feature Drift
print("Feature Distribution Changes Over Time")
print("="*60)

# Query feature statistics over time
feature_drift_query = f"""
SELECT 
    window.start as date,
    column_name,
    avg as mean_value,
    stddev,
    min,
    max
FROM {profile_table}
WHERE column_name IN ('MonthlyCharges', 'tenure', 'TotalCharges')
ORDER BY window.start ASC, column_name
"""

feature_stats_df = spark.sql(feature_drift_query)

if feature_stats_df.count() > 0:
    print("\n✅ Feature statistics over time:")
    display(feature_stats_df)
    
    # Convert to pandas for visualization
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    stats_pdf = feature_stats_df.toPandas()
    
    # Plot trends
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, feature in enumerate(['MonthlyCharges', 'tenure', 'TotalCharges']):
        feature_data = stats_pdf[stats_pdf['column_name'] == feature]
        
        if len(feature_data) > 0:
            ax = axes[idx]
            ax.plot(feature_data['date'], feature_data['mean_value'], marker='o', linewidth=2)
            ax.fill_between(
                feature_data['date'],
                feature_data['mean_value'] - feature_data['stddev'],
                feature_data['mean_value'] + feature_data['stddev'],
                alpha=0.3
            )
            ax.set_title(f'{feature} Over Time', fontsize=12, fontweight='bold')
            ax.set_xlabel('Date')
            ax.set_ylabel('Value')
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    print("\n📈 Trend Analysis:")
    for feature in ['MonthlyCharges', 'tenure']:
        feature_data = stats_pdf[stats_pdf['column_name'] == feature].sort_values('date')
        if len(feature_data) > 1:
            first_val = feature_data.iloc[0]['mean_value']
            last_val = feature_data.iloc[-1]['mean_value']
            change_pct = ((last_val - first_val) / first_val) * 100
            print(f"  {feature}: {first_val:.2f} → {last_val:.2f} ({change_pct:+.1f}%)")
else:
    print("⚠️ Not enough data points yet for trend analysis")

# COMMAND ----------

# COMMAND ----------

# DBTITLE 1,Model Performance Drift Over Time
print("Model Performance Trends")
print("="*60)

# Query model metrics over time
performance_query = f"""
SELECT 
    window.start as date,
    model_version,
    accuracy_score,
    precision.weighted as precision,
    recall.weighted as recall,
    f1_score.weighted as f1_score,
    log_loss
FROM {profile_table}
WHERE column_name = ':table'
ORDER BY window.start ASC
"""

perf_df = spark.sql(performance_query)

if perf_df.count() > 1:
    print(f"\n✅ Performance metrics over {perf_df.count()} time windows\n")
    display(perf_df)
    
    # Plot performance trends
    perf_pdf = perf_df.toPandas()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    metrics_to_plot = [
        ('accuracy_score', 'Accuracy'),
        ('precision', 'Precision'),
        ('recall', 'Recall'),
        ('f1_score', 'F1 Score')
    ]
    
    for idx, (metric_col, metric_name) in enumerate(metrics_to_plot):
        ax = axes[idx // 2, idx % 2]
        ax.plot(perf_pdf['date'], perf_pdf[metric_col], 
                marker='o', linewidth=2, markersize=8)
        ax.axhline(y=perf_pdf[metric_col].mean(), 
                   color='r', linestyle='--', alpha=0.5, label='Average')
        ax.set_title(f'{metric_name} Over Time', fontsize=12, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel(metric_name)
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # Calculate performance degradation
    print("\n📉 Performance Degradation Analysis:")
    first_metrics = perf_pdf.iloc[0]
    last_metrics = perf_pdf.iloc[-1]
    
    for metric_col, metric_name in metrics_to_plot:
        first_val = first_metrics[metric_col]
        last_val = last_metrics[metric_col]
        change = last_val - first_val
        change_pct = (change / first_val) * 100 if first_val > 0 else 0
        
        status = "⚠️ DEGRADED" if change < -0.05 else "✅ STABLE"
        print(f"  {metric_name}: {first_val:.4f} → {last_val:.4f} ({change_pct:+.1f}%) {status}")
    
    # Check if retraining is needed
    f1_degradation = first_metrics['f1_score'] - last_metrics['f1_score']
    if f1_degradation > 0.05:
        print("\n⚠️ ALERT: F1 Score degraded by >5% - consider retraining!")
    else:
        print("\n✅ Model performance is stable")
else:
    print("⚠️ Need more time windows for trend analysis")
    print("   Current windows:", perf_df.count())

# COMMAND ----------

# COMMAND ----------

# DBTITLE 1,Prediction Distribution Drift
print("Prediction Distribution Changes")
print("="*60)

# Query prediction distribution over time
pred_dist_query = f"""
SELECT 
    window.start as date,
    avg as avg_prediction,
    stddev as std_prediction,
    min as min_prediction,
    max as max_prediction,
    median
FROM {profile_table}
WHERE column_name = 'prediction'
ORDER BY window.start ASC
"""

pred_dist_df = spark.sql(pred_dist_query)

if pred_dist_df.count() > 0:
    print(f"\n✅ Prediction distribution over {pred_dist_df.count()} windows\n")
    display(pred_dist_df)
    
    # Plot
    pred_pdf = pred_dist_df.toPandas()
    
    plt.figure(figsize=(12, 6))
    plt.plot(pred_pdf['date'], pred_pdf['avg_prediction'], 
             marker='o', linewidth=2, markersize=8, label='Mean Prediction')
    plt.fill_between(
        pred_pdf['date'],
        pred_pdf['avg_prediction'] - pred_pdf['std_prediction'],
        pred_pdf['avg_prediction'] + pred_pdf['std_prediction'],
        alpha=0.3, label='±1 Std Dev'
    )
    plt.axhline(y=pred_pdf['avg_prediction'].iloc[0], 
                color='r', linestyle='--', alpha=0.5, label='Initial Mean')
    plt.title('Churn Prediction Rate Over Time', fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Average Churn Prediction')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Analyze drift
    first_pred = pred_pdf.iloc[0]['avg_prediction']
    last_pred = pred_pdf.iloc[-1]['avg_prediction']
    pred_change = last_pred - first_pred
    
    print(f"\n📊 Prediction Drift:")
    print(f"  Initial avg: {first_pred:.4f}")
    print(f"  Current avg: {last_pred:.4f}")
    print(f"  Change: {pred_change:+.4f} ({pred_change/first_pred*100:+.1f}%)")
    
    if abs(pred_change) > 0.1:
        print("\n⚠️ ALERT: Significant change in prediction distribution!")
        print("   This could indicate:")
        print("   - Customer base is changing")
        print("   - Market conditions shifted")
        print("   - Model may need retraining")
else:
    print("⚠️ No prediction distribution data yet")

# COMMAND ----------

# COMMAND ----------

# DBTITLE 1,Drift Summary Report
print("\n" + "="*70)
print("DRIFT MONITORING SUMMARY REPORT")
print("="*70)

# Get drift metrics summary
drift_summary_query = f"""
SELECT 
    column_name,
    COUNT(DISTINCT window.start) as num_windows,
    AVG(CASE WHEN ks_test.pvalue < 0.05 THEN 1 ELSE 0 END) as pct_significant_drift,
    AVG(wasserstein_distance) as avg_wasserstein_distance
FROM {drift_table}
WHERE drift_type = 'CONSECUTIVE'
  AND column_name IN ('MonthlyCharges', 'tenure', 'TotalCharges', 'prediction', 'ground_truth')
GROUP BY column_name
ORDER BY avg_wasserstein_distance DESC
"""

try:
    drift_summary = spark.sql(drift_summary_query)
    
    if drift_summary.count() > 0:
        print("\nFeatures Ranked by Drift Severity:")
        display(drift_summary)
        
        # Identify high-drift features
        high_drift = drift_summary.filter(F.col("avg_wasserstein_distance") > 0.1).collect()
        
        if len(high_drift) > 0:
            print("\n⚠️ HIGH DRIFT DETECTED IN:")
            for row in high_drift:
                print(f"  - {row.column_name}: Wasserstein={row.avg_wasserstein_distance:.4f}")
            
            print("\nRECOMMENDATIONS:")
            print("  1. Investigate root cause of distribution changes")
            print("  2. Check if business conditions changed")
            print("  3. Consider retraining model with recent data")
            print("  4. Validate model performance hasn't degraded")
        else:
            print("\n✅ All features show acceptable drift levels")
    else:
        print("No drift summary available yet")
        
except Exception as e:
    print(f"Error: {e}")

print("="*70)

# COMMAND ----------

# DBTITLE 1,Setup Alerts Example
alert_example = f"""
# Create SQL Alert for Performance Degradation

Query:
SELECT 
    window.end as check_time,
    f1_score.double as f1_score,
    accuracy
FROM {profile_table}
WHERE column_name = ':table'
  AND window.end >= current_date() - 1
  AND f1_score.double < 0.65
ORDER BY window.end DESC
LIMIT 1

Alert when: Query returns results
Frequency: Daily
Notification: Email/Slack
"""

print(alert_example)

# COMMAND ----------

# DBTITLE 1,Final Summary
print("\n" + "="*60)
print("MONITORING SETUP COMPLETE")
print("="*60)
print(f"Status: {status}")
print(f"\nComponents:")
print(f"  Inference Table: {inference_table}")
print(f"  Profile Metrics: {profile_table}")
print(f"  Dashboard: {monitor_info.dashboard_id}")
print(f"\nFeatures:")
print(f"  ✅ F1, Precision, Recall tracking")
print(f"  ✅ Data drift detection")
print(f"  ✅ Automated refresh (every 6 hours)")
print(f"  ✅ Historical trends")
print("="*60)

dbutils.jobs.taskValues.set(key="monitoring_table", value=inference_table)
dbutils.jobs.taskValues.set(key="dashboard_id", value=monitor_info.dashboard_id)
dbutils.jobs.taskValues.set(key="metrics_status", value=status)

print("\n🎉 MLOps Pipeline Complete!")

# COMMAND ----------

# COMMAND ----------

# DBTITLE 1,Verify Prediction Drift Data Exists

inference_table = "art_mlops.mlops_churn_demo.churn_inference_with_labels"
print("Checking for prediction drift over time...")
print("="*60)

# Query predictions grouped by day
prediction_by_day = spark.sql(f"""
SELECT 
    DATE(timestamp) as date,
    COUNT(*) as num_predictions,
    AVG(prediction) as avg_prediction,
    AVG(prediction_proba) as avg_probability,
    STDDEV(prediction_proba) as std_probability
FROM {inference_table}
GROUP BY DATE(timestamp)
ORDER BY date ASC
""")

print(f"\nPredictions by day:")
display(prediction_by_day)

# Check if we have multiple days
num_days = prediction_by_day.count()
print(f"\nNumber of unique days: {num_days}")

if num_days > 1:
    pdf = prediction_by_day.toPandas()
    first_day_pred = pdf.iloc[0]['avg_prediction']
    last_day_pred = pdf.iloc[-1]['avg_prediction']
    change = last_day_pred - first_day_pred
    
    print(f"\nPrediction Drift:")
    print(f"  First day avg: {first_day_pred:.4f}")
    print(f"  Last day avg: {last_day_pred:.4f}")
    print(f"  Change: {change:+.4f}")
    
    if abs(change) > 0.01:
        print(f"  ✅ Drift detected: {change/first_day_pred*100:+.1f}%")
    else:
        print(f"  ⚠️ Minimal drift: {change/first_day_pred*100:+.1f}%")
else:
    print("⚠️ Need data across multiple days")

print("="*60)

# COMMAND ----------

# COMMAND ----------

# DBTITLE 1,Visualize Prediction Drift Over Time
print("Prediction Drift Visualization")
print("="*60)

# Get daily prediction statistics
daily_preds = spark.sql(f"""
SELECT 
    DATE(timestamp) as date,
    AVG(prediction) as avg_binary_prediction,
    AVG(prediction_proba) as avg_probability,
    STDDEV(prediction_proba) as std_probability,
    MIN(prediction_proba) as min_probability,
    MAX(prediction_proba) as max_probability,
    COUNT(*) as sample_count
FROM {inference_table}
GROUP BY DATE(timestamp)
ORDER BY date ASC
""")

if daily_preds.count() > 1:
    pdf = daily_preds.toPandas()
    
    # Create visualization
    import matplotlib.pyplot as plt
    import seaborn as sns
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Average Churn Probability Over Time
    ax1 = axes[0]
    ax1.plot(pdf['date'], pdf['avg_probability'], 
             marker='o', linewidth=2, markersize=8, color='#ff6b6b', label='Avg Churn Probability')
    ax1.fill_between(
        pdf['date'],
        pdf['avg_probability'] - pdf['std_probability'],
        pdf['avg_probability'] + pdf['std_probability'],
        alpha=0.3, color='#ff6b6b'
    )
    
    # Add baseline
    baseline = pdf['avg_probability'].iloc[0]
    ax1.axhline(y=baseline, color='blue', linestyle='--', linewidth=2, 
                alpha=0.7, label=f'Baseline ({baseline:.3f})')
    
    # Add threshold
    ax1.axhline(y=0.5, color='red', linestyle=':', linewidth=2, 
                alpha=0.5, label='Decision Threshold (0.5)')
    
    ax1.set_title('Churn Probability Drift Over Time', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Average Churn Probability', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Predicted Churn Rate (Binary) Over Time
    ax2 = axes[1]
    ax2.plot(pdf['date'], pdf['avg_binary_prediction'], 
             marker='s', linewidth=2, markersize=8, color='#4ecdc4', label='Predicted Churn Rate')
    
    baseline_binary = pdf['avg_binary_prediction'].iloc[0]
    ax2.axhline(y=baseline_binary, color='blue', linestyle='--', linewidth=2, 
                alpha=0.7, label=f'Baseline ({baseline_binary:.1%})')
    
    ax2.set_title('Predicted Churn Rate Over Time', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Churn Rate (%)', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    # Format y-axis as percentage
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    plt.tight_layout()
    plt.show()
    
    # Print drift analysis
    print("\n📈 PREDICTION DRIFT ANALYSIS:")
    print("="*60)
    print(f"Time Period: {pdf['date'].min()} to {pdf['date'].max()}")
    print(f"Days: {len(pdf)}")
    print()
    
    first = pdf.iloc[0]
    last = pdf.iloc[-1]
    
    prob_change = last['avg_probability'] - first['avg_probability']
    rate_change = last['avg_binary_prediction'] - first['avg_binary_prediction']
    
    print(f"Churn Probability Drift:")
    print(f"  Initial: {first['avg_probability']:.4f}")
    print(f"  Latest:  {last['avg_probability']:.4f}")
    print(f"  Change:  {prob_change:+.4f} ({prob_change/first['avg_probability']*100:+.1f}%)")
    print()
    
    print(f"Churn Rate Drift:")
    print(f"  Initial: {first['avg_binary_prediction']:.1%}")
    print(f"  Latest:  {last['avg_binary_prediction']:.1%}")
    print(f"  Change:  {rate_change:+.1%}")
    print()
    
    if prob_change > 0.05:
        print("⚠️ ALERT: Significant upward drift in churn predictions!")
        print("   Possible causes:")
        print("   - Customer base deteriorating")
        print("   - Price increases driving churn")
        print("   - Service quality issues")
        print("\n   RECOMMENDATION: Investigate and consider intervention")
    elif prob_change < -0.05:
        print("✅ Positive trend: Churn predictions decreasing")
    else:
        print("✅ Predictions stable within acceptable range")
    
    print("="*60)
else:
    print("⚠️ Need data across multiple days")

# COMMAND ----------

# COMMAND ----------

# DBTITLE 1,Feature Changes vs Prediction Changes
print("Correlating Feature Drift with Prediction Changes")
print("="*60)

# Get feature trends
feature_trends = spark.sql(f"""
SELECT 
    DATE(timestamp) as date,
    AVG(MonthlyCharges) as avg_monthly_charges,
    AVG(tenure) as avg_tenure,
    AVG(Contract_Month_to_month) as pct_month_to_month,
    AVG(prediction_proba) as avg_churn_prob
FROM {inference_table}
GROUP BY DATE(timestamp)
ORDER BY date ASC
""")

if feature_trends.count() > 1:
    trends_pdf = feature_trends.toPandas()
    
    # Create multi-axis chart
    fig, ax1 = plt.subplots(figsize=(14, 6))
    
    # Churn probability on left axis
    color = '#ff6b6b'
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Churn Probability', color=color, fontsize=12)
    ax1.plot(trends_pdf['date'], trends_pdf['avg_churn_prob'], 
             marker='o', linewidth=3, markersize=8, color=color, label='Churn Probability')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Monthly charges on right axis
    ax2 = ax1.twinx()
    color = '#4ecdc4'
    ax2.set_ylabel('Avg Monthly Charges ($)', color=color, fontsize=12)
    ax2.plot(trends_pdf['date'], trends_pdf['avg_monthly_charges'], 
             marker='s', linewidth=2, markersize=6, color=color, 
             linestyle='--', label='Monthly Charges')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Churn Predictions vs Monthly Charges Over Time', 
              fontsize=14, fontweight='bold')
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    plt.show()
    
    # Correlation analysis
    print("\n📊 Drift Correlation Analysis:")
    print("="*60)
    
    corr_monthly = trends_pdf['avg_monthly_charges'].corr(trends_pdf['avg_churn_prob'])
    corr_tenure = trends_pdf['avg_tenure'].corr(trends_pdf['avg_churn_prob'])
    
    print(f"Monthly Charges ↔ Churn Probability: {corr_monthly:+.3f}")
    print(f"Tenure ↔ Churn Probability: {corr_tenure:+.3f}")
    print()
    
    if corr_monthly > 0.7:
        print("✅ Strong positive correlation: Higher prices → More churn")
    if corr_tenure < -0.5:
        print("✅ Negative correlation: Lower tenure → More churn")
    
    print("="*60)
else:
    print("Need more days of data")

# COMMAND ----------

# COMMAND ----------

# DBTITLE 1,Where to View Drift Visually
workspace_url = spark.conf.get("spark.databricks.workspaceUrl")

# Direct link to Quality tab
quality_url = f"https://{workspace_url}/explore/data/{catalog}/{db}/churn_inference_with_labels?o=&activeTab=quality"

print("\n" + "="*60)
print("VIEW DRIFT IN DATABRICKS UI")
print("="*60)
print("\n1. Catalog Explorer - Quality Tab:")
print(f"   {quality_url}")
print("\n2. What you'll see:")
print("   - Feature distribution charts over time")
print("   - Drift detection alerts")
print("   - Statistical test results")
print("   - Performance metric trends")
print("\n3. Metrics Tables:")
print(f"   - Profile: {profile_table}")
print(f"   - Drift: {drift_table}")
print("="*60)

displayHTML(f'''
<div style="padding:20px; background:#e3f2fd; border:2px solid #2196F3; border-radius:5px;">
    <h3>📊 View Drift Monitoring</h3>
    <p><strong>Simulated Drift:</strong></p>
    <ul>
        <li>MonthlyCharges: +30% increase over 14 days</li>
        <li>Tenure: -25% decrease (more new customers)</li>
        <li>Contract shifts: More month-to-month contracts</li>
    </ul>
    <p><a href="{quality_url}" target="_blank" 
          style="background:#2196F3; color:white; padding:10px 20px; 
                 text-decoration:none; border-radius:5px; display:inline-block;">
        Open Quality Tab in Catalog
    </a></p>
</div>
''')