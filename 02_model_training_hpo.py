# Databricks notebook source
# MAGIC %md
# MAGIC # Model Training with Hyperparameter Optimization
# MAGIC
# MAGIC <img src="https://github.com/pravinva/mlops-advanced-drift-retrain/raw/main/diagrams/mlops-advanced-2-training.png" width="1200px">
# MAGIC
# MAGIC <!-- Tracking -->
# MAGIC <img width="1px" src="https://www.google-analytics.com/collect?v=1&gtm=GTM-NKQ8TT7&tid=UA-163989034-1&cid=555&aip=1&t=event&ec=field_demos&ea=display&dp=%2F42_field_demos%2Fml%2Fmlops_advanced%2F02_training&dt=MLOPS_ADVANCED">
# MAGIC
# MAGIC ## Train LightGBM Model with Optuna
# MAGIC
# MAGIC This notebook trains a churn prediction model using LightGBM with automated hyperparameter optimization.
# MAGIC
# MAGIC **What we'll do:**
# MAGIC 1. Load training and test data
# MAGIC 2. Run Optuna hyperparameter optimization (50 trials)
# MAGIC 3. Train final model with best parameters
# MAGIC 4. Log model and metrics to MLflow
# MAGIC 5. Save model metadata for registration

# COMMAND ----------

# DBTITLE 1,Setup and Configuration
dbutils.widgets.dropdown("reset_all_data", "false", ["true", "false"], "Reset all data")

# COMMAND ----------

# MAGIC %run ./_resources/00-setup $reset_all_data=$reset_all_data $catalog=mlops_advanced $db=churn_demo

# COMMAND ----------

# DBTITLE 1,Install Required Libraries
# MAGIC %pip install optuna lightgbm scikit-learn --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Re-run Setup After Restart
# MAGIC %run ./_resources/00-setup $reset_all_data=$reset_all_data $catalog=mlops_advanced $db=churn_demo

# COMMAND ----------

# DBTITLE 1,Load Training Data
import pandas as pd
import numpy as np
import mlflow
import mlflow.lightgbm

train_df = spark.table(train_table)
test_df = spark.table(test_table)

train_pdf = train_df.toPandas()
test_pdf = test_df.toPandas()

feature_cols = [col for col in train_pdf.columns
                if col not in ['customerID', 'churn', 'ingestion_timestamp', 'source']]

X_train = train_pdf[feature_cols]
y_train = train_pdf['churn']
X_test = test_pdf[feature_cols]
y_test = test_pdf['churn']

print(f"Training samples: {len(X_train):,}")
print(f"Test samples: {len(X_test):,}")
print(f"Features: {len(feature_cols)}")

# COMMAND ----------

# DBTITLE 1,Hyperparameter Optimization with Optuna
import optuna
import lightgbm as lgb
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score

mlflow.set_experiment(experiment_path)

def objective(trial):
    """Optuna objective function for hyperparameter optimization"""
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'lambda_l1': trial.suggest_float('lambda_l1', 0.0, 10.0),
        'lambda_l2': trial.suggest_float('lambda_l2', 0.0, 10.0),
    }

    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        valid_sets=[valid_data],
        num_boost_round=200,
        callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
    )

    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int)
    f1 = f1_score(y_test, y_pred)

    with mlflow.start_run(nested=True):
        mlflow.log_params(params)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))

    return f1

print("Starting hyperparameter optimization (50 trials)...")

with mlflow.start_run(run_name="optuna_optimization") as parent_run:
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50, show_progress_bar=True)

    mlflow.log_params(study.best_params)
    mlflow.log_metric("best_f1_score", study.best_value)

    print(f"Best F1 Score: {study.best_value:.4f}")
    print("Best Parameters:", study.best_params)

# COMMAND ----------

# DBTITLE 1,Train Final Model with Best Parameters
best_params = study.best_params
best_params.update({
    'objective': 'binary',
    'metric': 'binary_logloss',
    'verbosity': -1,
    'boosting_type': 'gbdt'
})

print("Training final model with best parameters...")

with mlflow.start_run(run_name="final_champion_model") as run:
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    final_model = lgb.train(
        best_params,
        train_data,
        valid_sets=[valid_data],
        num_boost_round=300,
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=True)]
    )

    y_pred_proba = final_model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int)

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }

    mlflow.log_params(best_params)
    mlflow.log_metrics(metrics)

    from mlflow.models.signature import infer_signature
    signature = infer_signature(X_train, y_pred_proba)

    mlflow.lightgbm.log_model(
        final_model,
        "model",
        signature=signature,
        input_example=X_train.iloc[:5]
    )

    final_run_id = run.info.run_id

    print("\nFinal Model Metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")

# COMMAND ----------

# DBTITLE 1,Save Model Metadata
dbutils.jobs.taskValues.set(key="model_run_id", value=final_run_id)
dbutils.jobs.taskValues.set(key="f1_score", value=float(metrics['f1_score']))

print(f"Model run ID: {final_run_id}")
print("Model metadata saved for downstream tasks")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC
# MAGIC Model trained and logged to MLflow. Proceed to:
# MAGIC - **Next:** [03 - Model Registration]($./03_model_registration)
