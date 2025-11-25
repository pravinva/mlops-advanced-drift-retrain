# Testing Guide: MLOps Advanced Enhanced Demo

## Overview

This document provides comprehensive testing procedures for validating the MLOps Advanced demo enhancements.

## Test Categories

### 1. Unit Tests
### 2. Integration Tests
### 3. End-to-End Tests
### 4. Performance Tests
### 5. Validation Framework Tests

---

## 1. Unit Tests

### Feature Engineering Tests

**Test:** Data Cleaning
```python
# Notebook: 01_feature_engineering
# Validate TotalCharges cleaning
assert df.filter(col("TotalCharges").isNull()).count() == 0
assert df.filter(col("TotalCharges") < 0).count() == 0
```

**Test:** One-Hot Encoding
```python
# Verify all categorical columns encoded
categorical_cols = ['gender', 'Partner', 'Dependents', ...]
for cat_col in categorical_cols:
    assert cat_col not in df.columns, f"{cat_col} should be encoded"
```

**Test:** Train/Test Split
```python
# Validate 80/20 split
train_count = spark.table(train_table).count()
test_count = spark.table(test_table).count()
total_count = train_count + test_count

assert abs((train_count / total_count) - 0.80) < 0.02
assert abs((test_count / total_count) - 0.20) < 0.02
```

### Model Training Tests

**Test:** Optuna Trials Complete
```python
# Notebook: 02_model_training_hpo
# Verify 50 trials completed
import mlflow
runs = mlflow.search_runs(experiment_id=experiment_id)
assert len(runs) >= 50, "Should have at least 50 HPO trials"
```

**Test:** Model Artifacts Logged
```python
# Verify final model has required artifacts
run = mlflow.get_run(run_id)
artifacts = run.data.tags

assert 'mlflow.log-model.history' in artifacts
assert run.data.params.get('num_leaves') is not None
assert run.data.metrics.get('f1_score') is not None
```

### Model Registration Tests

**Test:** Unity Catalog Registration
```python
# Notebook: 03_model_registration
from mlflow.tracking import MlflowClient

client = MlflowClient()
model_versions = client.search_model_versions(f"name='{model_name}'")
assert len(model_versions) > 0, "Model should be registered"
```

**Test:** Challenger Alias Assigned
```python
# Verify Challenger alias exists
challenger = client.get_model_version_by_alias(model_name, "Challenger")
assert challenger is not None
assert challenger.version is not None
```

---

## 2. Integration Tests

### End-to-End Pipeline Test

**Test:** Feature → Train → Register → Validate → Approve

```python
# Run notebooks in sequence
%run ./01_feature_engineering $reset_all_data=false
%run ./02_model_training_hpo $reset_all_data=false
%run ./03_model_registration $reset_all_data=false
%run ./04a_challenger_validation $reset_all_data=false
%run ./04b_challenger_approval $reset_all_data=false

# Verify Champion exists
champion = client.get_model_version_by_alias(model_name, "Champion")
assert champion is not None, "Champion should be promoted after approval"
```

### Batch Inference Integration Test

**Test:** Champion Model Scoring

```python
# Notebook: 05_batch_inference
predictions_df = spark.table(f"{catalog}.{db}.churn_predictions")

# Verify predictions exist
assert predictions_df.count() > 0

# Verify probability range [0, 1]
stats = predictions_df.select("churn_probability").summary("min", "max")
min_val = float(stats.filter(col("summary") == "min").collect()[0][1])
max_val = float(stats.filter(col("summary") == "max").collect()[0][1])

assert 0 <= min_val <= 1
assert 0 <= max_val <= 1

# Verify risk categories exist
risk_counts = predictions_df.groupBy("risk_category").count()
assert risk_counts.count() == 3  # High, Medium, Low
```

### Monitoring Integration Test

**Test:** Lakehouse Monitor Creation

```python
# Notebook: 07_model_monitoring
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
monitor_info = w.quality_monitors.get(
    table_name=f"{catalog}.{db}.churn_inference_with_labels"
)

assert monitor_info is not None
assert monitor_info.status == "ACTIVE"
```

**Test:** Drift Metrics Generated

```python
# Verify drift metrics table exists and has data
drift_table = f"{catalog}.{db}.churn_inference_with_labels_drift_metrics"
drift_df = spark.table(drift_table)

assert drift_df.count() > 0
assert "ks_test" in drift_df.columns
assert "wasserstein_distance" in drift_df.columns
```

---

## 3. End-to-End Tests

### Complete Workflow Test

**Scenario:** First-time deployment of model

```python
# Step 1: Reset environment (optional)
%run ./00_mlops_end2end_advanced_presentation $reset_all_data=true

# Step 2: Run complete pipeline
notebooks = [
    "01_feature_engineering",
    "02_model_training_hpo",
    "03_model_registration",
    "04a_challenger_validation",
    "04b_challenger_approval",
    "05_batch_inference",
    "07_model_monitoring"
]

for notebook in notebooks:
    print(f"Running {notebook}...")
    %run ./{notebook} $reset_all_data=false
    print(f" {notebook} completed")

# Step 3: Verify final state
# - Champion model exists
# - Predictions generated
# - Monitoring active
```

**Expected Results:**
-  Feature tables created with CDF enabled
-  50 Optuna trials completed
-  Model registered with version ≥ 1
-  7 validation tests passed
-  Champion alias assigned
-  Predictions table populated
-  Lakehouse Monitor active

### Retraining Workflow Test

**Scenario:** Trigger automated retraining

```python
# Step 1: Create initial Champion (use workflow test above)

# Step 2: Simulate drift
# (Insert data with distribution shift)

# Step 3: Run retraining
%run ./08_automated_retraining $reset_all_data=false

# Step 4: Verify new Challenger created
versions = client.search_model_versions(f"name='{model_name}'")
assert len(versions) >= 2  # At least Champion + new Challenger

challenger = client.get_model_version_by_alias(model_name, "Challenger")
assert challenger.tags.get("retrain_reason") is not None
```

---

## 4. Performance Tests

### Batch Inference Performance

**Test:** Latency for large-scale scoring

```python
import time

# Load Champion model
model_uri = f"models:/{model_name}@Champion"
model = mlflow.pyfunc.load_model(model_uri)

# Get test data
test_df = spark.table(test_table).limit(10000)

# Measure batch scoring time
start = time.time()
predictions = model.predict(test_df.toPandas())
elapsed = time.time() - start

# Assertions
records_per_second = 10000 / elapsed
assert records_per_second > 1000, f"Too slow: {records_per_second:.0f} records/sec"
print(f" Batch performance: {records_per_second:.0f} records/sec")
```

### Model Serving Latency (if using notebook 06)

**Test:** P95 latency < 200ms

```python
import requests
import numpy as np

latencies = []
for _ in range(100):
    start = time.time()
    response = requests.post(endpoint_url, headers=headers, json=payload)
    latencies.append((time.time() - start) * 1000)

p95 = np.percentile(latencies, 95)
assert p95 < 200, f"P95 latency too high: {p95:.0f}ms"
print(f" P95 latency: {p95:.0f}ms")
```

---

## 5. Validation Framework Tests

### Test 1: Performance Metrics

```python
# Notebook: 04a_challenger_validation
# Test passes if:
# - F1 score >= 0.55
# - Accuracy >= 0.70
# - Precision >= 0.55
# - Recall >= 0.50

# Run validation
%run ./04a_challenger_validation

# Check results
validation_passed = dbutils.jobs.taskValues.get(
    taskKey="challenger_validation",
    key="validation_passed"
)
assert validation_passed == True, "Performance validation should pass"
```

### Test 2: Confusion Matrix

```python
# Test passes if:
# - False Negative Rate <= 50%

# Verify FNR calculation
from sklearn.metrics import confusion_matrix

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
fnr = fn / (fn + tp)
assert fnr <= 0.50, f"FNR too high: {fnr:.2%}"
```

### Test 3: Class Performance

```python
# Test passes if:
# - Minority class F1 >= 0.50

from sklearn.metrics import f1_score

minority_f1 = f1_score(y_test, y_pred, pos_label=1)
assert minority_f1 >= 0.50, f"Minority F1 too low: {minority_f1:.3f}"
```

### Test 4: Prediction Distribution

```python
# Test passes if:
# - Minority class prediction rate >= 10%

churn_pred_rate = np.mean(y_pred == 1)
assert churn_pred_rate >= 0.10, f"Churn prediction rate too low: {churn_pred_rate:.2%}"
```

### Test 5: Inference Latency

```python
# Test passes if:
# - Single prediction < 50ms

import time

sample = X_test.iloc[0:1]
latencies = []

for _ in range(20):
    start = time.time()
    model.predict(sample)
    latencies.append((time.time() - start) * 1000)

avg_latency = np.mean(latencies)
assert avg_latency < 50, f"Latency too high: {avg_latency:.1f}ms"
```

### Test 6: Champion Comparison

```python
# Test passes if:
# - Challenger F1 >= Champion F1 - 0.02

champion_f1 = 0.60  # From Champion metadata
challenger_f1 = 0.59  # From current evaluation

assert challenger_f1 >= champion_f1 - 0.02, "F1 degraded too much"
```

### Test 7: Robustness

```python
# Test passes if:
# - Model handles 10% missing values

# Introduce missing values
X_test_missing = X_test.copy()
mask = np.random.rand(*X_test.shape) < 0.10
X_test_missing[mask] = np.nan

# Predict
try:
    preds = model.predict(X_test_missing)
    assert len(preds) == len(X_test_missing)
    assert all((preds >= 0) & (preds <= 1))
    print(" Robustness test passed")
except Exception as e:
    assert False, f"Model failed with missing values: {e}"
```

---

## 6. Drift Detection Tests

### Feature Drift Simulation Test

```python
# Simulate drift over 14 days
import pandas as pd
from datetime import datetime, timedelta

base_date = datetime.now()
drift_data = []

for day in range(14):
    date = base_date + timedelta(days=day)

    # Progressive drift
    monthly_charges_shift = 1.0 + (day / 14) * 0.30  # +30% over 14 days
    tenure_shift = 1.0 - (day / 14) * 0.25  # -25% over 14 days

    # Generate records
    for _ in range(100):
        record = {
            'prediction_timestamp': date,
            'MonthlyCharges': np.random.normal(70, 20) * monthly_charges_shift,
            'tenure': np.random.normal(30, 15) * tenure_shift,
            'churn_probability': np.random.rand(),
            'churn_actual': np.random.choice([0, 1])
        }
        drift_data.append(record)

# Insert into inference table
drift_df = spark.createDataFrame(drift_data)
drift_df.write.mode("append").saveAsTable(inference_table)

# Refresh monitor
w.quality_monitors.run_refresh(table_name=inference_table)

# Wait for completion
time.sleep(60)

# Verify drift detected
drift_metrics = spark.table(f"{inference_table}_drift_metrics")
drift_detected = drift_metrics.filter(
    (col("column_name") == "MonthlyCharges") &
    (col("ks_test.pvalue") < 0.05)
).count()

assert drift_detected > 0, "Drift should be detected for MonthlyCharges"
```

---

## 7. Deployment Job Tests

### Job Creation Test

```python
# Notebook: 09_deployment_job
%run ./09_deployment_job

# Verify job created
job_id = dbutils.jobs.taskValues.get(taskKey="deployment_job", key="job_id")
assert job_id is not None

# Verify job configuration
from databricks.sdk import WorkspaceClient
w = WorkspaceClient()
job = w.jobs.get(job_id=job_id)

assert len(job.settings.tasks) == 6
assert job.settings.max_concurrent_runs == 1
```

### Job Execution Test

```python
# Trigger job run
run = w.jobs.run_now(job_id=job_id)

# Wait for completion (max 2 hours)
import time
timeout = 7200  # 2 hours
start = time.time()

while time.time() - start < timeout:
    run_status = w.jobs.get_run(run_id=run.run_id)

    if run_status.state.life_cycle_state in ["TERMINATED", "SKIPPED"]:
        assert run_status.state.result_state == "SUCCESS"
        print(f" Job completed successfully in {int(time.time() - start)}s")
        break

    time.sleep(30)
else:
    assert False, "Job timed out"
```

---

## Test Execution Checklist

### Pre-Testing Setup

- [ ] Databricks workspace accessible
- [ ] Unity Catalog enabled
- [ ] ML Runtime 14.3+ cluster running
- [ ] Required libraries installed (lightgbm, optuna, scikit-learn, databricks-sdk)
- [ ] Appropriate permissions (catalog/schema creation, job creation)

### Unit Tests

- [ ] Feature engineering data cleaning
- [ ] One-hot encoding validation
- [ ] Train/test split ratios
- [ ] Optuna trials completion
- [ ] Model artifacts logging
- [ ] Unity Catalog registration
- [ ] Alias assignment

### Integration Tests

- [ ] End-to-end pipeline (01 → 05)
- [ ] Batch inference predictions
- [ ] Monitoring setup and activation
- [ ] Drift metrics generation

### End-to-End Tests

- [ ] Complete workflow (first deployment)
- [ ] Retraining workflow (model update)
- [ ] Job orchestration

### Performance Tests

- [ ] Batch inference latency
- [ ] Model serving latency (if applicable)
- [ ] Scalability validation

### Validation Framework Tests

- [ ] Test 1: Performance metrics
- [ ] Test 2: Confusion matrix
- [ ] Test 3: Class performance
- [ ] Test 4: Prediction distribution
- [ ] Test 5: Inference latency
- [ ] Test 6: Champion comparison
- [ ] Test 7: Robustness

### Drift Detection Tests

- [ ] Feature drift simulation
- [ ] Drift detection accuracy
- [ ] Monitor refresh functionality

### Deployment Tests

- [ ] Job creation
- [ ] Job execution
- [ ] Task dependencies
- [ ] Email notifications

---

## Troubleshooting

### Common Issues

**Issue:** Validation tests fail
- Check quality gate thresholds in `04a_challenger_validation`
- Verify test data distribution matches training data
- Review feature engineering consistency

**Issue:** Drift not detected
- Ensure monitoring refresh completed
- Check inference table has multiple days of data
- Verify drift threshold settings (p-value < 0.05)

**Issue:** Job execution fails
- Review task logs in Databricks UI
- Check cluster configuration
- Verify notebook paths are correct
- Ensure dependencies are installed

**Issue:** Performance tests fail
- Check cluster size and worker count
- Verify no resource contention
- Review Spark configuration

---

## Continuous Testing

### Automated Test Suite

Create a Databricks Job for automated testing:

```python
# Job: mlops_advanced_test_suite
# Tasks:
1. Unit Tests (notebook: tests/unit_tests.py)
2. Integration Tests (notebook: tests/integration_tests.py)
3. Validation Tests (notebook: tests/validation_tests.py)
4. Performance Tests (notebook: tests/performance_tests.py)

# Schedule: Daily at 2 AM
# Notifications: Email on failure
```

### CI/CD Integration

Integrate with GitHub Actions or Databricks Repos:

```yaml
# .github/workflows/test.yml
name: MLOps Advanced Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run Databricks Tests
        env:
          DATABRICKS_HOST: ${{ secrets.DATABRICKS_HOST }}
          DATABRICKS_TOKEN: ${{ secrets.DATABRICKS_TOKEN }}
        run: |
          databricks jobs run-now --job-id <test_job_id>
```

---

## Test Reporting

### Metrics to Track

- Test pass rate (%)
- Test execution time
- Code coverage
- Performance benchmarks
- Drift detection accuracy

### Dashboard

Create a SQL dashboard in Databricks to track test results over time.

---

## Next Steps

After testing is complete:
1. Document any issues found
2. Update thresholds if needed
3. Submit enhancement proposal (see `DBDEMOS_SUBMISSION.md`)
4. Monitor production deployment

---

**Last Updated:** 2024-11-25
**Version:** 1.0
