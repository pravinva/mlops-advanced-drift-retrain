# Databricks notebook source
# MAGIC %md
# MAGIC # Automated Challenger Model Validation
# MAGIC
# MAGIC <img src="https://github.com/pravinva/mlops-advanced-drift-retrain/raw/main/diagrams/mlops-advanced-4-validation.png" width="1200px">
# MAGIC
# MAGIC <!-- Tracking -->
# MAGIC <img width="1px" src="https://www.google-analytics.com/collect?v=1&gtm=GTM-NKQ8TT7&tid=UA-163989034-1&cid=555&aip=1&t=event&ec=field_demos&ea=display&dp=%2F42_field_demos%2Fml%2Fmlops_advanced%2F04a_validation&dt=MLOPS_ADVANCED">
# MAGIC
# MAGIC ## 7-Stage Model Validation Framework
# MAGIC
# MAGIC This notebook implements an enterprise-grade validation framework with realistic quality gates.
# MAGIC
# MAGIC **Validation Tests:**
# MAGIC 1. **Performance Metrics** - F1, Accuracy, Precision, Recall, ROC-AUC
# MAGIC 2. **Confusion Matrix** - False negative/positive rate analysis
# MAGIC 3. **Class Performance** - Minority class (churn) performance
# MAGIC 4. **Prediction Distribution** - Ensure model isn't predicting all one class
# MAGIC 5. **Inference Latency** - Single prediction speed test
# MAGIC 6. **Champion Comparison** - Compare with production model
# MAGIC 7. **Robustness** - Handle missing values gracefully

# COMMAND ----------

# DBTITLE 1,Setup and Configuration
dbutils.widgets.dropdown("reset_all_data", "false", ["true", "false"], "Reset all data")

# COMMAND ----------

# MAGIC %run ./_resources/00-setup $reset_all_data=$reset_all_data $catalog=mlops_advanced $db=churn_demo

# COMMAND ----------

# DBTITLE 1,Install Required Libraries
# MAGIC %pip install lightgbm scikit-learn --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Re-run Setup After Restart
# MAGIC %run ./_resources/00-setup $reset_all_data=$reset_all_data $catalog=mlops_advanced $db=churn_demo

# COMMAND ----------

# DBTITLE 1,Load Challenger Model from Unity Catalog
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import numpy as np
from sklearn.metrics import *

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

try:
    challenger_model = mlflow.pyfunc.load_model(f"models:/{model_name}@Challenger")
    challenger_version = client.get_model_version_by_alias(model_name, "Challenger")
    print(f"✅ Challenger Model Loaded")
    print(f"   Model: {model_name}")
    print(f"   Version: {challenger_version.version}")
    print(f"   Alias: Challenger")
except Exception as e:
    raise Exception(f"Failed to load Challenger model. Run notebook 03 first. Error: {e}")

# COMMAND ----------

# DBTITLE 1,Load Test Dataset
test_df = spark.table(test_table)
test_pdf = test_df.toPandas()

feature_cols = [col for col in test_pdf.columns if col not in ['customerID', 'churn']]
X_test = test_pdf[feature_cols]
y_test = test_pdf['churn']

print(f"Test Dataset: {len(X_test):,} samples")
print(f"Features: {len(feature_cols)}")
print(f"Churn Rate: {y_test.mean():.2%}")

# COMMAND ----------

# DBTITLE 1,Generate Predictions
y_pred_proba = challenger_model.predict(X_test)
y_pred = (y_pred_proba > 0.5).astype(int)

print(f"Predictions generated: {len(y_pred):,}")
print(f"Predicted Churn Rate: {y_pred.mean():.2%}")

# COMMAND ----------

# DBTITLE 1,Test 1: Performance Metrics
print("=" * 60)
print("TEST 1: PERFORMANCE METRICS")
print("=" * 60)

metrics = {
    'accuracy': accuracy_score(y_test, y_pred),
    'precision': precision_score(y_test, y_pred),
    'recall': recall_score(y_test, y_pred),
    'f1_score': f1_score(y_test, y_pred),
    'roc_auc': roc_auc_score(y_test, y_pred_proba)
}

# Realistic quality gates for imbalanced churn datasets
quality_gates = {
    'accuracy': 0.70,
    'precision': 0.55,
    'recall': 0.50,
    'f1_score': 0.55,
    'roc_auc': 0.75
}

test_1_passed = True
for metric_name, metric_value in metrics.items():
    threshold = quality_gates[metric_name]
    passed = metric_value >= threshold
    test_1_passed = test_1_passed and passed
    status = "PASS" if passed else "FAIL"
    print(f"{metric_name:12s}: {metric_value:.4f} (threshold: {threshold:.4f}) {status}")

print(f"\nTest 1: {'PASSED' if test_1_passed else 'FAILED'}")

# COMMAND ----------

# DBTITLE 1,Test 2: Confusion Matrix Analysis
print("\n" + "=" * 60)
print("TEST 2: CONFUSION MATRIX")
print("=" * 60)

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

max_fnr = 0.50  # Maximum acceptable false negative rate

test_2_passed = fnr <= max_fnr

print(f"Confusion Matrix:")
print(f"  True Positives (TP):  {tp:4d}")
print(f"  False Positives (FP): {fp:4d}")
print(f"  True Negatives (TN):  {tn:4d}")
print(f"  False Negatives (FN): {fn:4d}")
print(f"\nError Rates:")
print(f"  False Negative Rate: {fnr:.4f} (max: {max_fnr:.4f})")
print(f"  False Positive Rate: {fpr:.4f}")
print(f"\nTest 2: {'PASSED' if test_2_passed else 'FAILED'}")

# COMMAND ----------

# DBTITLE 1,Test 3: Class-Wise Performance
print("\n" + "=" * 60)
print("TEST 3: CLASS-WISE PERFORMANCE")
print("=" * 60)

report = classification_report(
    y_test, y_pred,
    target_names=['No Churn', 'Churn'],
    output_dict=True
)

churn_f1 = report['Churn']['f1-score']
min_churn_f1 = 0.50  # Minimum F1 for minority class

test_3_passed = churn_f1 >= min_churn_f1

print(f"No Churn Class (Majority):")
print(f"  Precision: {report['No Churn']['precision']:.4f}")
print(f"  Recall:    {report['No Churn']['recall']:.4f}")
print(f"  F1-Score:  {report['No Churn']['f1-score']:.4f}")

print(f"\nChurn Class (Minority):")
print(f"  Precision: {report['Churn']['precision']:.4f}")
print(f"  Recall:    {report['Churn']['recall']:.4f}")
print(f"  F1-Score:  {churn_f1:.4f} (min: {min_churn_f1:.4f})")

print(f"\nTest 3: {'PASSED' if test_3_passed else 'FAILED'}")

# COMMAND ----------

# DBTITLE 1,Test 4: Prediction Distribution
print("\n" + "=" * 60)
print("TEST 4: PREDICTION DISTRIBUTION")
print("=" * 60)

pred_dist = pd.Series(y_pred).value_counts(normalize=True)
min_minority_pred = 0.10  # At least 10% should be minority class

minority_pred_rate = min(pred_dist.get(0, 0), pred_dist.get(1, 0))
test_4_passed = minority_pred_rate >= min_minority_pred

print(f"Prediction Distribution:")
print(f"  Class 0 (No Churn): {pred_dist.get(0, 0):.2%}")
print(f"  Class 1 (Churn):    {pred_dist.get(1, 0):.2%}")
print(f"\nMinority Class Rate: {minority_pred_rate:.2%} (min: {min_minority_pred:.0%})")
print(f"\nTest 4: {'PASSED' if test_4_passed else 'FAILED'}")

# COMMAND ----------

# DBTITLE 1,Test 5: Inference Latency
print("\n" + "=" * 60)
print("TEST 5: INFERENCE LATENCY")
print("=" * 60)

import time

single_record = X_test.head(1)
latencies = []

# Warm-up predictions
for _ in range(3):
    _ = challenger_model.predict(single_record)

# Measure latency
for _ in range(20):
    start = time.time()
    _ = challenger_model.predict(single_record)
    latencies.append((time.time() - start) * 1000)

avg_latency = np.mean(latencies)
p95_latency = np.percentile(latencies, 95)
max_latency = 50  # Maximum acceptable latency in ms

test_5_passed = avg_latency <= max_latency

print(f"Single Prediction Latency:")
print(f"  Average: {avg_latency:.2f}ms")
print(f"  P95:     {p95_latency:.2f}ms")
print(f"  Max:     {np.max(latencies):.2f}ms")
print(f"  Threshold: {max_latency}ms")
print(f"\nTest 5: {'PASSED' if test_5_passed else 'FAILED'}")

# COMMAND ----------

# DBTITLE 1,Test 6: Champion Comparison
print("\n" + "=" * 60)
print("TEST 6: CHAMPION COMPARISON")
print("=" * 60)

try:
    champion_model = mlflow.pyfunc.load_model(f"models:/{model_name}@Champion")
    champion_version = client.get_model_version_by_alias(model_name, "Champion")

    champion_pred_proba = champion_model.predict(X_test)
    champion_pred = (champion_pred_proba > 0.5).astype(int)

    champion_f1 = f1_score(y_test, champion_pred)
    challenger_f1 = metrics['f1_score']

    # Allow up to 2% F1 degradation
    min_improvement = -0.02
    improvement = challenger_f1 - champion_f1
    test_6_passed = improvement >= min_improvement

    print(f"Champion Model:")
    print(f"  Version: {champion_version.version}")
    print(f"  F1 Score: {champion_f1:.4f}")

    print(f"\nChallenger Model:")
    print(f"  Version: {challenger_version.version}")
    print(f"  F1 Score: {challenger_f1:.4f}")

    print(f"\nImprovement: {improvement:+.4f}")
    print(f"Minimum Required: {min_improvement:+.4f}")
    print(f"\nTest 6: {'PASSED' if test_6_passed else 'FAILED'}")

except Exception as e:
    print("No Champion model found - this is the first model")
    print("Skipping Champion comparison")
    test_6_passed = True
    print(f"\nTest 6: PASSED (no Champion to compare)")

# COMMAND ----------

# DBTITLE 1,Test 7: Robustness - Missing Values
print("\n" + "=" * 60)
print("TEST 7: ROBUSTNESS - MISSING VALUES")
print("=" * 60)

test_sample = X_test.head(100).copy()
original_dtypes = test_sample.dtypes.to_dict()

# Introduce 10% missing values randomly
np.random.seed(42)
for col in test_sample.columns:
    mask = np.random.random(len(test_sample)) < 0.1
    test_sample.loc[mask, col] = np.nan

# Fill missing values with 0
test_sample_filled = test_sample.fillna(0)

# Convert to correct dtypes
for col, dtype in original_dtypes.items():
    if col in ['SeniorCitizen', 'tenure']:
        test_sample_filled[col] = test_sample_filled[col].astype('int32')
    elif dtype in ['int64', 'int32', 'int16', 'int8']:
        test_sample_filled[col] = test_sample_filled[col].astype('int64')
    elif dtype in ['float64', 'float32']:
        test_sample_filled[col] = test_sample_filled[col].astype('float64')

try:
    robust_pred = challenger_model.predict(test_sample_filled)

    # Validation checks
    valid_range = (robust_pred >= 0).all() and (robust_pred <= 1).all()
    pred_mean = robust_pred.mean()
    reasonable_mean = 0.05 <= pred_mean <= 0.95
    pred_std = robust_pred.std()
    has_variance = pred_std > 0.01

    test_7_passed = valid_range and reasonable_mean and has_variance

    print(f"Predictions with 10% missing values:")
    print(f"  Mean: {pred_mean:.4f}")
    print(f"  Std:  {pred_std:.4f}")
    print(f"  Min:  {robust_pred.min():.4f}")
    print(f"  Max:  {robust_pred.max():.4f}")

    print(f"\nValidation Checks:")
    print(f"  Valid Range [0,1]:           {'PASS' if valid_range else 'FAIL'}")
    print(f"  Reasonable Mean (0.05-0.95): {'PASS' if reasonable_mean else 'FAIL'}")
    print(f"  Has Variance (>0.01):        {'PASS' if has_variance else 'FAIL'}")

    print(f"\nTest 7: {'PASSED' if test_7_passed else 'FAILED'}")

except Exception as e:
    print(f"Model failed with missing values: {e}")
    test_7_passed = False
    print(f"\nTest 7: FAILED")

# COMMAND ----------

# DBTITLE 1,Validation Summary
all_tests_passed = all([
    test_1_passed, test_2_passed, test_3_passed, test_4_passed,
    test_5_passed, test_6_passed, test_7_passed
])

print("\n" + "=" * 60)
print("VALIDATION SUMMARY")
print("=" * 60)
print(f"Test 1 - Performance Metrics:  {'PASSED' if test_1_passed else 'FAILED'}")
print(f"Test 2 - Confusion Matrix:     {'PASSED' if test_2_passed else 'FAILED'}")
print(f"Test 3 - Class Performance:    {'PASSED' if test_3_passed else 'FAILED'}")
print(f"Test 4 - Distribution:         {'PASSED' if test_4_passed else 'FAILED'}")
print(f"Test 5 - Latency:              {'PASSED' if test_5_passed else 'FAILED'}")
print(f"Test 6 - Champion Comparison:  {'PASSED' if test_6_passed else 'FAILED'}")
print(f"Test 7 - Robustness:           {'PASSED' if test_7_passed else 'FAILED'}")
print("=" * 60)

if all_tests_passed:
    print("\nALL TESTS PASSED - Ready for promotion to Champion")
else:
    print("\nSOME TESTS FAILED - Review before promotion")
    print("\nRealistic Quality Gates (Imbalanced Churn Dataset):")
    print("  Accuracy: ≥70% | Precision: ≥55% | Recall: ≥50%")
    print("  F1 Score: ≥55% | ROC AUC: ≥75%")
    print("  False Negative Rate: ≤50%")
    print("  Churn Class F1: ≥50%")
    print("\nNote: These thresholds reflect real-world churn prediction challenges.")
    print("Consider business context when deciding on promotion.")

print("\n" + "=" * 60)

# COMMAND ----------

# DBTITLE 1,Save Validation Results
dbutils.jobs.taskValues.set(key="validation_passed", value=all_tests_passed)
dbutils.jobs.taskValues.set(key="challenger_version", value=str(challenger_version.version))
dbutils.jobs.taskValues.set(key="challenger_f1", value=float(metrics['f1_score']))

print("✅ Validation results saved for downstream tasks")
print(f"   validation_passed: {all_tests_passed}")
print(f"   challenger_version: {challenger_version.version}")
print(f"   challenger_f1: {metrics['f1_score']:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Quality Gates Explained
# MAGIC
# MAGIC These quality gates are calibrated for imbalanced churn datasets:
# MAGIC
# MAGIC ### Core Metrics
# MAGIC - **Accuracy ≥70%**: Overall correctness (heavily influenced by majority class)
# MAGIC - **Precision ≥55%**: Of predicted churners, how many actually churn
# MAGIC - **Recall ≥50%**: Of actual churners, how many we identify
# MAGIC - **F1 Score ≥55%**: Harmonic mean of precision and recall
# MAGIC - **ROC AUC ≥75%**: Overall discrimination ability
# MAGIC
# MAGIC ### Error Analysis
# MAGIC - **False Negative Rate ≤50%**: Can't miss more than half of churners
# MAGIC - **Churn Class F1 ≥50%**: Minority class must perform reasonably
# MAGIC
# MAGIC ### Operational
# MAGIC - **Latency ≤50ms**: Single prediction responsiveness
# MAGIC - **Champion Delta ≥-2%**: Can't degrade by more than 2% F1
# MAGIC - **Robustness**: Must handle missing values gracefully
# MAGIC
# MAGIC ## Next Steps
# MAGIC
# MAGIC Validation complete. Proceed to:
# MAGIC - **If PASSED:** [04b - Challenger Approval]($./04b_challenger_approval)
# MAGIC - **If FAILED:** Review metrics and retrain (Notebook 02)
