# Databricks notebook source
# MAGIC %md
# MAGIC # Model Retraining Based on Drift
# MAGIC
# MAGIC Retrain model when drift or performance degradation is detected:
# MAGIC - Assess current model performance
# MAGIC - Prepare fresh training data from recent observations
# MAGIC - Retrain with updated data
# MAGIC - Register as new Challenger
# MAGIC - Trigger validation workflow

# COMMAND ----------

# DBTITLE 1,Install Dependencies
# MAGIC %pip install lightgbm optuna scikit-learn --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./notebook-00-setup

# COMMAND ----------

# DBTITLE 1,Setup After Restart
import mlflow
from mlflow.tracking import MlflowClient
import lightgbm as lgb
import optuna
from sklearn.metrics import *
import pandas as pd
import numpy as np
from pyspark.sql import functions as F
from datetime import datetime, timedelta

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()


model_name = f"{catalog}.{db}.churn_model"
features_table = f"{catalog}.{db}.churn_features"
inference_table = f"{catalog}.{db}.churn_inference_with_labels"
experiment_path = f"/Users/{dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()}/mlops_churn_experiments"

mlflow.set_experiment(experiment_path)

print(f"Model: {model_name}")
print(f"Experiment: {experiment_path}")

# COMMAND ----------

# DBTITLE 1,Step 1: Assess Current Model Performance
print("\n" + "="*70)
print("STEP 1: ASSESS CURRENT MODEL PERFORMANCE")
print("="*70)

# Get current Champion
champion_version = client.get_model_version_by_alias(model_name, "Champion")
print(f"\nCurrent Champion: Version {champion_version.version}")

# Get recent performance from monitoring
profile_table = f"{catalog}.{db}.churn_inference_with_labels_profile_metrics"

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
    print("-"*70)
    
    retrain_needed = False
    degradation_threshold = 0.05  # 5% degradation
    
    for metric in ['accuracy_score', 'precision', 'recall', 'f1_score']:
        baseline_val = getattr(baseline_metrics, metric)
        current_val = getattr(latest_metrics, metric)
        change = current_val - baseline_val
        
        degraded = change < -degradation_threshold
        if degraded:
            retrain_needed = True
        
        status = "❌ RETRAIN" if degraded else "✅ OK"
        metric_name = metric.replace('_', ' ').title()
        print(f"{metric_name:<15} {baseline_val:<12.4f} {current_val:<12.4f} {change:<+12.4f} {status}")
    
    print("="*70)
    
    if retrain_needed:
        print("\n⚠️ RETRAINING RECOMMENDED: Performance degraded >5%")
    else:
        print("\n✅ Performance acceptable, but retraining with fresh data anyway")
    
except Exception as e:
    print(f"\n⚠️ Could not get monitoring metrics: {e}")
    print("Proceeding with retraining based on time-based policy")
    retrain_needed = True

# COMMAND ----------

# DBTITLE 1,Step 2: Prepare Fresh Training Data
# COMMAND ----------

# DBTITLE 1,Step 2: Prepare Fresh Training Data - FIXED
print("\n" + "="*70)
print("STEP 2: PREPARE FRESH TRAINING DATA")
print("="*70)

# Get recent inference data - explicitly select and rename columns
recent_inference = spark.table(inference_table).select([
    F.col("customerID"),
    F.col("SeniorCitizen"),
    F.col("tenure"),
    F.col("MonthlyCharges"),
    F.col("TotalCharges"),
    F.col("ground_truth").alias("churn")  # Rename ground_truth to churn
] + [
    F.col(c) for c in spark.table(inference_table).columns 
    if c.startswith(('gender_', 'Partner_', 'Dependents_', 'PhoneService_', 
                     'MultipleLines_', 'InternetService_', 'OnlineSecurity_',
                     'OnlineBackup_', 'DeviceProtection_', 'TechSupport_',
                     'StreamingTV_', 'StreamingMovies_', 'Contract_',
                     'PaperlessBilling_', 'PaymentMethod_'))
])

# Get historical features - select same columns
historical_features = spark.table(features_table).select(
    recent_inference.columns
)

print("Column alignment:")
print(f"  Recent inference columns: {len(recent_inference.columns)}")
print(f"  Historical columns: {len(historical_features.columns)}")
print(f"  Columns match: {set(recent_inference.columns) == set(historical_features.columns)}")

# Combine datasets
recent_count = recent_inference.count()
historical_count = int(recent_count * 0.3 / 0.7)

fresh_data = recent_inference.union(
    historical_features.limit(historical_count)
).dropDuplicates(['customerID'])

print(f"\nFresh Training Dataset:")
print(f"  Recent observations: {recent_count}")
print(f"  Historical data: {historical_count}")
print(f"  Total unique: {fresh_data.count()}")

# Verify churn column exists
if 'churn' in fresh_data.columns:
    churn_rate = fresh_data.select(F.avg("churn")).first()[0]
    print(f"  Churn rate: {churn_rate:.2%}")
    print("  ✅ Label column verified")
else:
    print("  ❌ ERROR: churn column missing!")

# Create train/test split
train_fresh, test_fresh = fresh_data.randomSplit([0.8, 0.2], seed=43)

# Save
retrain_train_table = f"{catalog}.{db}.churn_features_retrain_train"
retrain_test_table = f"{catalog}.{db}.churn_features_retrain_test"

train_fresh.write.mode("overwrite").saveAsTable(retrain_train_table)
test_fresh.write.mode("overwrite").saveAsTable(retrain_test_table)

print(f"\n✅ Fresh data splits created:")
print(f"   Train: {train_fresh.count()} → {retrain_train_table}")
print(f"   Test: {test_fresh.count()} → {retrain_test_table}")
print("="*70)

# COMMAND ----------

# DBTITLE 1,Step 3: Load Fresh Data for Training
print("\n" + "="*70)
print("STEP 3: LOAD FRESH DATA")
print("="*70)

train_pdf = spark.table(retrain_train_table).toPandas()
test_pdf = spark.table(retrain_test_table).toPandas()

feature_cols = [c for c in train_pdf.columns if c not in ['customerID', 'churn']]

X_train = train_pdf[feature_cols]
y_train = train_pdf['churn']

X_test = test_pdf[feature_cols]
y_test = test_pdf['churn']

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Features: {len(feature_cols)}")
print(f"Churn rate (train): {y_train.mean():.2%}")
print(f"Churn rate (test): {y_test.mean():.2%}")

# COMMAND ----------

# DBTITLE 1,Step 4: Get Best Hyperparameters from Current Champion
print("\n" + "="*70)
print("STEP 4: RETRIEVE CHAMPION HYPERPARAMETERS")
print("="*70)

# Get hyperparameters from current Champion
champion_run = mlflow.get_run(champion_version.run_id)
champion_params = champion_run.data.params

# Use proven hyperparameters as starting point
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

for param_name, param_type in param_mapping.items():
    if param_name in champion_params:
        try:
            best_params[param_name] = param_type(champion_params[param_name])
        except:
            # Use default if conversion fails
            defaults = {
                'num_leaves': 50, 'learning_rate': 0.1, 'feature_fraction': 0.8,
                'bagging_fraction': 0.8, 'bagging_freq': 5, 'min_child_samples': 20,
                'max_depth': 8, 'lambda_l1': 0, 'lambda_l2': 0
            }
            best_params[param_name] = defaults[param_name]

print("\nUsing hyperparameters from Champion:")
for k, v in sorted(best_params.items()):
    if k not in ['objective', 'metric', 'verbosity', 'boosting_type']:
        print(f"  {k}: {v}")

# COMMAND ----------

# DBTITLE 1,Step 5: Retrain Model with Fresh Data
print("\n" + "="*70)
print("STEP 5: RETRAIN MODEL")
print("="*70)

print("Training model with fresh data and proven hyperparameters...")

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
    mlflow.log_param("training_data_source", "recent_observations_plus_historical")
    
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
    
    print("\n" + "="*70)
    print("RETRAINED MODEL PERFORMANCE")
    print("="*70)
    for metric_name, metric_value in new_metrics.items():
        print(f"{metric_name.upper()}: {metric_value:.4f}")
    print("="*70)
    print(f"\nMLflow Run ID: {retrain_run_id}")

# COMMAND ----------

# DBTITLE 1,Step 6: Compare with Current Champion
print("\n" + "="*70)
print("STEP 6: COMPARE RETRAINED vs CURRENT CHAMPION")
print("="*70)

# Get Champion's original metrics
champion_run = mlflow.get_run(champion_version.run_id)
champion_metrics = {
    'accuracy': champion_run.data.metrics.get('accuracy', 0),
    'precision': champion_run.data.metrics.get('precision', 0),
    'recall': champion_run.data.metrics.get('recall', 0),
    'f1_score': champion_run.data.metrics.get('f1_score', 0),
    'roc_auc': champion_run.data.metrics.get('roc_auc', 0)
}

print(f"\n{'Metric':<15} {'Champion':<12} {'Retrained':<12} {'Change':<12} {'Status'}")
print("-"*70)

should_register = False
improvements = {}

for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']:
    champ_val = champion_metrics[metric]
    new_val = new_metrics[metric]
    change = new_val - champ_val
    
    improvements[metric] = change
    
    # Determine status
    if change > 0.01:
        status = "✅ BETTER"
    elif change > -0.02:
        status = "➡️ SIMILAR"
    else:
        status = "❌ WORSE"
    
    print(f"{metric.upper():<15} {champ_val:<12.4f} {new_val:<12.4f} {change:<+12.4f} {status}")

print("="*70)

# Decision criteria
f1_improvement = improvements['f1_score']

if f1_improvement > 0:
    print("\n🎉 RETRAINED MODEL IS BETTER!")
    print(f"   F1 Score improved by: +{f1_improvement:.4f}")
    should_register = True
elif f1_improvement > -0.02:
    print("\n⚠️ Retrained model is similar (within 2%)")
    print("   Will register for A/B testing")
    should_register = True
else:
    print("\n❌ Retrained model performs worse")
    print(f"   F1 Score degraded by: {f1_improvement:.4f}")
    print("   Will NOT register - needs investigation")
    should_register = False

print("="*70)

# COMMAND ----------

# DBTITLE 1,Step 7: Register as New Challenger
if should_register:
    print("\n" + "="*70)
    print("STEP 7: REGISTER RETRAINED MODEL")
    print("="*70)
    
    model_uri = f"runs:/{retrain_run_id}/model"
    
    # Register to Unity Catalog
    new_version = mlflow.register_model(
        model_uri=model_uri,
        name=model_name,
        tags={
            "retrain_reason": "drift_detected",
            "retrain_date": datetime.now().isoformat(),
            "f1_score": str(new_metrics['f1_score']),
            "training_strategy": "recent_plus_historical",
            "champion_version_replaced": str(champion_version.version)
        }
    )
    
    print(f"\n✅ Model registered: Version {new_version.version}")
    
    # Add detailed description
    client.update_model_version(
        name=model_name,
        version=new_version.version,
        description=f"""
Retrained Model - Response to Drift

**Retrain Trigger:** Performance drift detected in production monitoring
**Retrain Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Run ID:** {retrain_run_id}

**Performance Metrics:**
- F1 Score: {new_metrics['f1_score']:.4f}
- Accuracy: {new_metrics['accuracy']:.4f}
- Precision: {new_metrics['precision']:.4f}
- Recall: {new_metrics['recall']:.4f}
- ROC AUC: {new_metrics['roc_auc']:.4f}

**Comparison with Previous Champion (v{champion_version.version}):**
- F1 Score: {f1_improvement:+.4f}
- Accuracy: {improvements['accuracy']:+.4f}
- Precision: {improvements['precision']:+.4f}
- Recall: {improvements['recall']:+.4f}

**Training Data:**
- Source: Recent 30-day observations + historical data
- Training samples: {len(X_train)}
- Test samples: {len(X_test)}
- Churn rate: {y_train.mean():.2%}

**Next Steps:**
- Run validation tests (04a_challenger_validation)
- Get approval (04b_challenger_approval)
- Deploy to production if approved
"""
    )
    
    # Set as Challenger
    # First remove old Challenger if exists
    try:
        old_challenger = client.get_model_version_by_alias(model_name, "Challenger")
        client.delete_registered_model_alias(model_name, "Challenger")
        print(f"Removed old Challenger (v{old_challenger.version})")
    except:
        pass
    
    client.set_registered_model_alias(
        name=model_name,
        alias="Challenger",
        version=new_version.version
    )
    
    print(f"✅ Set as Challenger: Version {new_version.version}")
    
    # Add lineage
    client.set_model_version_tag(
        name=model_name,
        version=new_version.version,
        key="training_table",
        value=retrain_train_table
    )
    
    print("\n✅ Registration complete!")
    
else:
    print("\n❌ Skipping registration - model did not improve")
    print("   Review training process and data quality")

# COMMAND ----------

# DBTITLE 1,Step 8: Model Comparison Summary
print("\n" + "="*70)
print("MODEL REGISTRY STATUS")
print("="*70)

all_versions = client.search_model_versions(f"name='{model_name}'")

print("\nAll Model Versions:")
for mv in sorted(all_versions, key=lambda x: int(x.version), reverse=True)[:5]:
    aliases = mv.aliases if hasattr(mv, 'aliases') and mv.aliases is not None else []
    alias_str = f" [{', '.join(aliases if isinstance(aliases, list) else [])}]" if aliases else ""
    
    try:
        run = mlflow.get_run(mv.run_id)
        f1 = run.data.metrics.get('f1_score', 'N/A')
        f1_str = f"F1={f1:.4f}" if f1 != 'N/A' else "No metrics"
    except:
        f1_str = "No metrics"
    
    print(f"  Version {mv.version}{alias_str}: {f1_str}")

print("="*70)

# COMMAND ----------

# DBTITLE 1,Step 9: Next Steps
print("\n" + "="*70)
print("RETRAINING WORKFLOW COMPLETE")
print("="*70)

if should_register:
    print("\n✅ New Challenger Model Ready!")
    print(f"   Version: {new_version.version}")
    print(f"   F1 Score: {new_metrics['f1_score']:.4f}")
    print(f"   Improvement: {f1_improvement:+.4f}")
    
    print("\n📋 NEXT STEPS:")
    print("   1. Run notebook 04a (challenger_validation)")
    print("      → Runs 7 automated validation tests")
    print()
    print("   2. Run notebook 04b (challenger_approval)")
    print("      → Promotes to Champion if tests pass")
    print()
    print("   3. Update production endpoints")
    print("      → New Champion will be used for inference")
    print()
    print("   4. Continue monitoring")
    print("      → Watch for new drift patterns")
    
    # Save for downstream notebooks
    dbutils.jobs.taskValues.set(key="retrained_model_version", value=str(new_version.version))
    dbutils.jobs.taskValues.set(key="retrain_run_id", value=retrain_run_id)
    dbutils.jobs.taskValues.set(key="should_validate", value=True)
    
else:
    print("\n⚠️ Retrained Model NOT Registered")
    print("   Performance did not improve sufficiently")
    
    print("\n📋 TROUBLESHOOTING STEPS:")
    print("   1. Review data quality in fresh dataset")
    print("   2. Try hyperparameter optimization (run with Optuna)")
    print("   3. Investigate feature engineering improvements")
    print("   4. Check for data leakage or quality issues")
    
    dbutils.jobs.taskValues.set(key="should_validate", value=False)

print("="*70)

# COMMAND ----------

# DBTITLE 1,Optional: Auto-Trigger Validation
# Uncomment to automatically trigger validation workflow

# if should_register:
#     print("\n🚀 Auto-triggering validation workflow...")
#     
#     try:
#         # Run validation
#         validation_result = dbutils.notebook.run(
#             "04a_challenger_validation",
#             timeout_seconds=1800
#         )
#         
#         print("✅ Validation complete")
#         
#         # Run approval if validation passed
#         approval_result = dbutils.notebook.run(
#             "04b_challenger_approval",
#             timeout_seconds=600
#         )
#         
#         print("✅ Approval workflow complete")
#         
#     except Exception as e:
#         print(f"⚠️ Automated workflow error: {e}")
#         print("   Run validation notebooks manually")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Retraining Complete! 🎉
# MAGIC
# MAGIC ### What Happened:
# MAGIC 1. ✅ Assessed current model performance
# MAGIC 2. ✅ Prepared fresh training data (recent + historical)
# MAGIC 3. ✅ Retrained using proven hyperparameters
# MAGIC 4. ✅ Registered as new Challenger (if improved)
# MAGIC
# MAGIC ### Manual Next Steps:
# MAGIC 1. Run **[04a_challenger_validation](04a_challenger_validation)** to validate
# MAGIC 2. Run **[04b_challenger_approval](04b_challenger_approval)** to promote
# MAGIC
# MAGIC ### Automation Options:
# MAGIC - Uncomment auto-trigger cell above for hands-free workflow
# MAGIC - Schedule this notebook to run weekly
# MAGIC - Trigger based on drift alerts