# Databricks notebook source
# MAGIC %md
# MAGIC # Automated Model Retraining
# MAGIC
# MAGIC <img src="https://github.com/pravinva/mlops-advanced-drift-retrain/raw/main/diagrams/mlops-advanced-8-retraining.png" width="1200px">
# MAGIC
# MAGIC <!-- Tracking -->
# MAGIC <img width="1px" src="https://www.google-analytics.com/collect?v=1&gtm=GTM-NKQ8TT7&tid=UA-163989034-1&cid=555&aip=1&t=event&ec=field_demos&ea=display&dp=%2F42_field_demos%2Fml%2Fmlops_advanced%2F08_retraining&dt=MLOPS_ADVANCED">
# MAGIC
# MAGIC ## Complete Retraining Workflow Triggered by Drift
# MAGIC
# MAGIC This notebook demonstrates enterprise-grade automated retraining in response to performance degradation or drift.
# MAGIC
# MAGIC **Retraining Workflow:**
# MAGIC 1. Assess current model performance from monitoring
# MAGIC 2. Prepare fresh training data (recent + historical)
# MAGIC 3. Retrain with proven hyperparameters from Champion
# MAGIC 4. Compare retrained model vs current Champion
# MAGIC 5. Register as new Challenger if improved
# MAGIC 6. Trigger validation and approval workflow

# COMMAND ----------

# DBTITLE 1,Setup and Configuration
dbutils.widgets.dropdown("reset_all_data", "false", ["true", "false"], "Reset all data")

# COMMAND ----------

# MAGIC %run ./_resources/00-setup $reset_all_data=$reset_all_data $catalog=mlops_advanced $db=churn_demo

# COMMAND ----------

# DBTITLE 1,Install Required Libraries
# MAGIC %pip install lightgbm optuna scikit-learn --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Re-run Setup After Restart
# MAGIC %run ./_resources/00-setup $reset_all_data=$reset_all_data $catalog=mlops_advanced $db=churn_demo

# COMMAND ----------

# DBTITLE 1,Import Libraries and Setup
import mlflow
from mlflow.tracking import MlflowClient
import lightgbm as lgb
from sklearn.metrics import *
import pandas as pd
import numpy as np
from pyspark.sql import functions as F
from datetime import datetime, timedelta

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()
mlflow.set_experiment(experiment_path)

print(f"Model: {model_name}")
print(f"Experiment: {experiment_path}")

# COMMAND ----------

# DBTITLE 1,Step 1: Assess Current Model Performance
print("\n" + "=" * 70)
print("STEP 1: ASSESS CURRENT MODEL PERFORMANCE")
print("=" * 70)

# Get current Champion
try:
    champion_version = client.get_model_version_by_alias(model_name, "Champion")
    print(f"\nCurrent Champion: Version {champion_version.version}")
except Exception as e:
    raise Exception(f"No Champion model found. Run notebooks 02-04b first. Error: {e}")

# Get recent performance from monitoring
profile_table = f"{catalog}.{db}.{inference_table.split('.')[-1]}_profile_metrics"

try:
    # Get latest and baseline metrics
    latest_metrics = spark.sql(f"""
    SELECT
        window.start as metric_date,
        accuracy_score,
        precision.weighted as precision,
        recall.weighted as recall,
        f1_score.weighted as f1_score
    FROM {profile_table}
    WHERE column_name = ':table'
    ORDER BY window.start DESC
    LIMIT 1
    """).first()

    baseline_metrics = spark.sql(f"""
    SELECT
        window.start as metric_date,
        accuracy_score,
        precision.weighted as precision,
        recall.weighted as recall,
        f1_score.weighted as f1_score
    FROM {profile_table}
    WHERE column_name = ':table'
    ORDER BY window.start ASC
    LIMIT 1
    """).first()

    print("\nPerformance Comparison:")
    print(f"{'Metric':<15} {'Baseline':<12} {'Current':<12} {'Change':<12} {'Status'}")
    print("-" * 70)

    retrain_needed = False
    degradation_threshold = 0.05  # 5% degradation

    for metric in ['accuracy_score', 'precision', 'recall', 'f1_score']:
        baseline_val = getattr(baseline_metrics, metric)
        current_val = getattr(latest_metrics, metric)
        change = current_val - baseline_val

        degraded = change < -degradation_threshold
        if degraded:
            retrain_needed = True

        status = "RETRAIN" if degraded else "OK"
        print(f"{metric:<15} {baseline_val:<12.4f} {current_val:<12.4f} {change:<+12.4f} {status}")

    print("=" * 70)

    if retrain_needed:
        print("\nRETRAINING RECOMMENDED: Performance degraded >5%")
    else:
        print("\nPerformance acceptable, retraining with fresh data for continuous improvement")

except Exception as e:
    print(f"\nCould not get monitoring metrics: {e}")
    print("Proceeding with retraining based on time-based policy")
    retrain_needed = True

# COMMAND ----------

# DBTITLE 1,Step 2: Prepare Fresh Training Data
print("\n" + "=" * 70)
print("STEP 2: PREPARE FRESH TRAINING DATA")
print("=" * 70)

# Get recent inference data with ground truth
recent_inference = spark.table(inference_table).select([
    F.col("customerID"),
    F.col("SeniorCitizen"),
    F.col("tenure"),
    F.col("MonthlyCharges"),
    F.col("TotalCharges"),
    F.col("ground_truth").alias("churn")
] + [
    F.col(c) for c in spark.table(inference_table).columns
    if c.startswith(('gender_', 'Partner_', 'Dependents_', 'PhoneService_',
                     'MultipleLines_', 'InternetService_', 'OnlineSecurity_',
                     'OnlineBackup_', 'DeviceProtection_', 'TechSupport_',
                     'StreamingTV_', 'StreamingMovies_', 'Contract_',
                     'PaperlessBilling_', 'PaymentMethod_'))
])

# Get historical features
historical_features = spark.table(features_table).select(recent_inference.columns)

# Combine datasets (70% recent, 30% historical)
recent_count = recent_inference.count()
historical_count = int(recent_count * 0.3 / 0.7)

fresh_data = recent_inference.union(
    historical_features.limit(historical_count)
).dropDuplicates(['customerID'])

print(f"\nFresh Training Dataset:")
print(f"  Recent observations: {recent_count:,}")
print(f"  Historical data: {historical_count:,}")
print(f"  Total unique: {fresh_data.count():,}")

# Verify data
if 'churn' in fresh_data.columns:
    churn_rate = fresh_data.select(F.avg("churn")).first()[0]
    print(f"  Churn rate: {churn_rate:.2%}")
    print("  Label column verified")
else:
    raise Exception("Error: churn column missing!")

# Create train/test split
train_fresh, test_fresh = fresh_data.randomSplit([0.8, 0.2], seed=43)

# Save to tables
retrain_train_table = f"{catalog}.{db}.churn_features_retrain_train"
retrain_test_table = f"{catalog}.{db}.churn_features_retrain_test"

train_fresh.write.mode("overwrite").saveAsTable(retrain_train_table)
test_fresh.write.mode("overwrite").saveAsTable(retrain_test_table)

print(f"\nFresh data splits created:")
print(f"  Train: {train_fresh.count():,} records")
print(f"  Test: {test_fresh.count():,} records")

# COMMAND ----------

# DBTITLE 1,Step 3: Load Fresh Data
print("\n" + "=" * 70)
print("STEP 3: LOAD FRESH DATA FOR TRAINING")
print("=" * 70)

train_pdf = spark.table(retrain_train_table).toPandas()
test_pdf = spark.table(retrain_test_table).toPandas()

feature_cols = [c for c in train_pdf.columns if c not in ['customerID', 'churn']]

X_train = train_pdf[feature_cols]
y_train = train_pdf['churn']
X_test = test_pdf[feature_cols]
y_test = test_pdf['churn']

print(f"Training samples: {len(X_train):,}")
print(f"Test samples: {len(X_test):,}")
print(f"Features: {len(feature_cols)}")
print(f"Churn rate (train): {y_train.mean():.2%}")
print(f"Churn rate (test): {y_test.mean():.2%}")

# COMMAND ----------

# DBTITLE 1,Step 4: Get Champion Hyperparameters
print("\n" + "=" * 70)
print("STEP 4: RETRIEVE CHAMPION HYPERPARAMETERS")
print("=" * 70)

# Get hyperparameters from current Champion
champion_run = mlflow.get_run(champion_version.run_id)
champion_params = champion_run.data.params

# Use proven hyperparameters
best_params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'verbosity': -1,
    'boosting_type': 'gbdt'
}

# Extract numeric parameters
param_mapping = {
    'num_leaves': int,
    'learning_rate': float,
    'feature_fraction': float,
    'bagging_fraction': float,
    'bagging_freq': int,
    'min_child_samples': int,
    'max_depth': int,
    'lambda_l1': float,
    'lambda_l2': float
}

defaults = {
    'num_leaves': 50, 'learning_rate': 0.1, 'feature_fraction': 0.8,
    'bagging_fraction': 0.8, 'bagging_freq': 5, 'min_child_samples': 20,
    'max_depth': 8, 'lambda_l1': 0, 'lambda_l2': 0
}

for param_name, param_type in param_mapping.items():
    if param_name in champion_params:
        try:
            best_params[param_name] = param_type(champion_params[param_name])
        except:
            best_params[param_name] = defaults[param_name]
    else:
        best_params[param_name] = defaults[param_name]

print("\nUsing hyperparameters from Champion:")
for k, v in sorted(best_params.items()):
    if k not in ['objective', 'metric', 'verbosity', 'boosting_type']:
        print(f"  {k}: {v}")

# COMMAND ----------

# DBTITLE 1,Step 5: Retrain Model
print("\n" + "=" * 70)
print("STEP 5: RETRAIN MODEL WITH FRESH DATA")
print("=" * 70)

with mlflow.start_run(run_name="retrained_on_drift") as run:
    # Create datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    # Train
    retrained_model = lgb.train(
        best_params,
        train_data,
        valid_sets=[valid_data],
        num_boost_round=300,
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=True)]
    )

    # Evaluate
    y_pred_proba = retrained_model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int)

    new_metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }

    # Log everything
    mlflow.log_params(best_params)
    mlflow.log_metrics(new_metrics)
    mlflow.log_param("retrain_reason", "drift_detected")
    mlflow.log_param("retrain_date", datetime.now().isoformat())
    mlflow.log_param("training_data_source", "recent_plus_historical")

    # Log model
    from mlflow.models.signature import infer_signature
    signature = infer_signature(X_train, y_pred_proba)

    mlflow.lightgbm.log_model(
        retrained_model,
        "model",
        signature=signature,
        input_example=X_train.iloc[:5]
    )

    retrain_run_id = run.info.run_id

    print("\n" + "=" * 70)
    print("RETRAINED MODEL PERFORMANCE")
    print("=" * 70)
    for metric_name, metric_value in new_metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")
    print("=" * 70)
    print(f"\nMLflow Run ID: {retrain_run_id}")

# COMMAND ----------

# DBTITLE 1,Step 6: Compare with Champion
print("\n" + "=" * 70)
print("STEP 6: COMPARE RETRAINED vs CURRENT CHAMPION")
print("=" * 70)

# Get Champion's original metrics
champion_metrics = {
    'accuracy': champion_run.data.metrics.get('accuracy', 0),
    'precision': champion_run.data.metrics.get('precision', 0),
    'recall': champion_run.data.metrics.get('recall', 0),
    'f1_score': champion_run.data.metrics.get('f1_score', 0),
    'roc_auc': champion_run.data.metrics.get('roc_auc', 0)
}

print(f"\n{'Metric':<15} {'Champion':<12} {'Retrained':<12} {'Change':<12} {'Status'}")
print("-" * 70)

should_register = False
improvements = {}

for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']:
    champ_val = champion_metrics[metric]
    new_val = new_metrics[metric]
    change = new_val - champ_val

    improvements[metric] = change

    if change > 0.01:
        status = "BETTER"
    elif change > -0.02:
        status = "SIMILAR"
    else:
        status = "WORSE"

    print(f"{metric:<15} {champ_val:<12.4f} {new_val:<12.4f} {change:<+12.4f} {status}")

print("=" * 70)

# Decision criteria
f1_improvement = improvements['f1_score']

if f1_improvement > 0:
    print("\nRetrained model is BETTER!")
    print(f"  F1 Score improved by: +{f1_improvement:.4f}")
    should_register = True
elif f1_improvement > -0.02:
    print("\nRetrained model is similar (within 2%)")
    print("  Will register for A/B testing")
    should_register = True
else:
    print("\nRetrained model performs worse")
    print(f"  F1 Score degraded by: {f1_improvement:.4f}")
    print("  Will NOT register")
    should_register = False

# COMMAND ----------

# DBTITLE 1,Step 7: Register as New Challenger
if should_register:
    print("\n" + "=" * 70)
    print("STEP 7: REGISTER RETRAINED MODEL AS CHALLENGER")
    print("=" * 70)

    model_uri = f"runs:/{retrain_run_id}/model"

    # Register to Unity Catalog
    new_version = mlflow.register_model(
        model_uri=model_uri,
        name=model_name,
        tags={
            "retrain_reason": "drift_detected",
            "retrain_date": datetime.now().isoformat(),
            "f1_score": str(new_metrics['f1_score']),
            "training_strategy": "recent_plus_historical"
        }
    )

    print(f"\nModel registered: Version {new_version.version}")

    # Add description
    client.update_model_version(
        name=model_name,
        version=new_version.version,
        description=f"""
# Retrained Model - Response to Drift

**Retrain Trigger:** Performance drift detected
**Retrain Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Run ID:** `{retrain_run_id}`

## Performance Metrics
| Metric | Value |
|--------|-------|
| F1 Score | {new_metrics['f1_score']:.4f} |
| Accuracy | {new_metrics['accuracy']:.4f} |
| Precision | {new_metrics['precision']:.4f} |
| Recall | {new_metrics['recall']:.4f} |
| ROC AUC | {new_metrics['roc_auc']:.4f} |

## Comparison with Previous Champion (v{champion_version.version})
| Metric | Change |
|--------|--------|
| F1 Score | {f1_improvement:+.4f} |
| Accuracy | {improvements['accuracy']:+.4f} |
| Precision | {improvements['precision']:+.4f} |
| Recall | {improvements['recall']:+.4f} |

## Training Data
- Source: Recent observations + historical data
- Training samples: {len(X_train):,}
- Test samples: {len(X_test):,}
- Churn rate: {y_train.mean():.2%}
"""
    )

    # Remove old Challenger if exists
    try:
        old_challenger = client.get_model_version_by_alias(model_name, "Challenger")
        client.delete_registered_model_alias(model_name, "Challenger")
        print(f"Removed old Challenger (v{old_challenger.version})")
    except:
        pass

    # Set as Challenger
    client.set_registered_model_alias(
        name=model_name,
        alias="Challenger",
        version=new_version.version
    )

    print(f"Set as Challenger: Version {new_version.version}")

    # Save for downstream tasks
    dbutils.jobs.taskValues.set(key="retrained_model_version", value=str(new_version.version))
    dbutils.jobs.taskValues.set(key="retrain_run_id", value=retrain_run_id)
    dbutils.jobs.taskValues.set(key="should_validate", value=True)

    print("\nRegistration complete!")

else:
    print("\nSkipping registration - model did not improve")
    dbutils.jobs.taskValues.set(key="should_validate", value=False)

# COMMAND ----------

# DBTITLE 1,Retraining Summary
print("\n" + "=" * 70)
print("RETRAINING WORKFLOW COMPLETE")
print("=" * 70)

if should_register:
    print("\nNew Challenger Model Ready!")
    print(f"  Version: {new_version.version}")
    print(f"  F1 Score: {new_metrics['f1_score']:.4f}")
    print(f"  Improvement: {f1_improvement:+.4f}")

    print("\nNEXT STEPS:")
    print("  1. Run notebook 04a (challenger_validation)")
    print("  2. Run notebook 04b (challenger_approval)")
    print("  3. Model will auto-deploy if approved")
else:
    print("\nRetrained model not registered")
    print("  Performance did not improve sufficiently")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Retraining Complete
# MAGIC
# MAGIC ### What Happened:
# MAGIC 1. Assessed current model performance from monitoring
# MAGIC 2. Prepared fresh training data (recent + historical)
# MAGIC 3. Retrained using proven hyperparameters from Champion
# MAGIC 4. Compared retrained model vs current Champion
# MAGIC 5. Registered as new Challenger (if improved)
# MAGIC
# MAGIC ### Next Steps:
# MAGIC - **If Registered:** Run [04a - Challenger Validation]($./04a_challenger_validation)
# MAGIC - **If Not Registered:** Review data quality and hyperparameters
