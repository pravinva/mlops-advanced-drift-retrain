# Databricks notebook source
# MAGIC %md
# MAGIC # Challenger Model Approval Workflow

# COMMAND ----------

# MAGIC %run ./notebook-00-setup

# COMMAND ----------

# DBTITLE 1,Setup
# Load configuration

user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()

print(f"User: {user}")
print(f"Model: {model_name}")

# COMMAND ----------

# DBTITLE 1,Load Validation Results
import mlflow
from mlflow.tracking import MlflowClient
from datetime import datetime

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()



challenger_version = client.get_model_version_by_alias(model_name, "Challenger")
print(f"Challenger Model: {model_name} v{challenger_version.version}")

# Try to get validation results from previous task (if running as a job)
validation_passed = None
try:
    validation_passed = dbutils.jobs.taskValues.get(taskKey="model_validation", key="validation_passed")
    print(f"Validation Status (from job): {'✅ PASSED' if validation_passed else '❌ FAILED'}")
except Exception as e:
    # If not running as a job, check MLflow for recent validation runs
    print("⚠️ Not running as part of a job, checking MLflow for validation results...")
    
    try:
        # Search for recent validation runs
        experiment_path = f"/Users/{dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()}/mlops_churn_experiments"
        experiment = mlflow.get_experiment_by_name(experiment_path)
        
        if experiment:
            # Get recent validation runs
            validation_runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string="tags.mlflow.runName = 'challenger_validation'",
                order_by=["start_time DESC"],
                max_results=1
            )
            
            if len(validation_runs) > 0:
                latest_validation = validation_runs.iloc[0]
                validation_passed = latest_validation.get('metrics.validation_passed', None)
                
                if validation_passed is not None:
                    validation_passed = bool(validation_passed)
                    print(f"✅ Found recent validation run (from MLflow)")
                    print(f"   Validation Status: {'✅ PASSED' if validation_passed else '❌ FAILED'}")
                    print(f"   Run Time: {latest_validation['start_time']}")
    except Exception as mlflow_error:
        print(f"Could not find validation results in MLflow: {mlflow_error}")

# If still no validation results, ask user or default to manual review
if validation_passed is None:
    print("\n" + "="*60)
    print("⚠️  NO VALIDATION RESULTS FOUND")
    print("="*60)
    print("Options:")
    print("  1. Run notebook 04a (challenger_validation) first")
    print("  2. Or set validation_passed manually below:")
    print("     validation_passed = True  # If you reviewed and approved")
    print("     validation_passed = False # If validation failed")
    print("="*60)
    
    # Default to requiring manual review
    validation_passed = None
    print("\nDefaulting to: MANUAL REVIEW REQUIRED")

# COMMAND ----------

# DBTITLE 1,Get Model Metrics
run = mlflow.get_run(challenger_version.run_id)
metrics = run.data.metrics

print("\nChallenger Performance:")
print(f"  F1 Score: {metrics.get('f1_score', 0):.4f}")
print(f"  Accuracy: {metrics.get('accuracy', 0):.4f}")
print(f"  Precision: {metrics.get('precision', 0):.4f}")
print(f"  Recall: {metrics.get('recall', 0):.4f}")

# COMMAND ----------



# COMMAND ----------

# DBTITLE 1,Manual Override (Optional)
# MAGIC %md
# MAGIC ## Manual Approval Override
# MAGIC
# MAGIC If validation results aren't available but you want to proceed, uncomment and set:
# MAGIC ```python
# MAGIC # Manual approval after reviewing model metrics
# MAGIC validation_passed = True  # Set to True to approve, False to reject
# MAGIC ```

# COMMAND ----------

# Uncomment to manually override validation
validation_passed = True

# COMMAND ----------

# DBTITLE 1,Compare with Champion
try:
    champion_version = client.get_model_version_by_alias(model_name, "Champion")
    champion_run = mlflow.get_run(champion_version.run_id)
    champion_metrics = champion_run.data.metrics
    
    print("\nChampion vs Challenger:")
    for metric in ['f1_score', 'accuracy', 'precision', 'recall']:
        champ_val = champion_metrics.get(metric, 0)
        chall_val = metrics.get(metric, 0)
        improvement = chall_val - champ_val
        print(f"  {metric}: {champ_val:.4f} → {chall_val:.4f} ({improvement:+.4f})")
    
    has_champion = True
    f1_improvement = metrics.get('f1_score', 0) - champion_metrics.get('f1_score', 0)
except:
    print("\nNo Champion found - this will be the first Champion")
    has_champion = False
    f1_improvement = 0

# COMMAND ----------

# DBTITLE 1,Approval Decision
auto_promote = False

print("\n" + "="*60)
print("APPROVAL DECISION")
print("="*60)

# Check if we have validation results
if validation_passed is None:
    print("⚠️  No automated validation results available")
    print("    Manual review required before promotion")
    print("\nTo manually approve, uncomment and run:")
    print("    # validation_passed = True")
    print("    # Then re-run cells below")
    auto_promote = False
elif validation_passed:
    print("✅ Validation: PASSED")
    if has_champion:
        if f1_improvement > 0:
            print(f"✅ Performance: Better than Champion (+{f1_improvement:.4f})")
            auto_promote = True
        else:
            print(f"⚠️  Performance: Not better than Champion ({f1_improvement:+.4f})")
            auto_promote = False
    else:
        print("✅ No existing Champion - ready for first deployment")
        auto_promote = True
else:
    print("❌ Validation: FAILED")
    print("    Review validation failures before promoting")
    auto_promote = False

print()
print(f"Decision: {'🚀 AUTO-PROMOTE' if auto_promote else '⚠️  MANUAL REVIEW REQUIRED'}")
print("="*60)

# COMMAND ----------

# DBTITLE 1,Promote to Champion
if auto_promote:
    print("\n🚀 Promoting Challenger to Champion...")
    
    # Archive old Champion
    if has_champion:
        print(f"1. Archiving old Champion v{champion_version.version}")
        client.set_registered_model_alias(
            name=model_name,
            alias="Previous_Champion",
            version=champion_version.version
        )
        client.delete_registered_model_alias(name=model_name, alias="Champion")
    
    # Promote Challenger
    print(f"2. Promoting Challenger v{challenger_version.version} to Champion")
    client.set_registered_model_alias(
        name=model_name,
        alias="Champion",
        version=challenger_version.version
    )
    
    # Remove Challenger alias
    client.delete_registered_model_alias(name=model_name, alias="Challenger")
    
    # Update metadata
    client.set_model_version_tag(
        name=model_name,
        version=challenger_version.version,
        key="promoted_to_champion",
        value=datetime.now().isoformat()
    )
    
    print("\n" + "="*60)
    print("✅ MODEL PROMOTION COMPLETE")
    print("="*60)
    print(f"New Champion: Version {challenger_version.version}")
    print(f"Status: Ready for production deployment")
    print("="*60)
    
    promotion_status = "promoted"
elif validation_passed is None:
    print("\n⚠️  Model NOT promoted - Manual review required")
    print("\nReasons:")
    print("  - No automated validation results available")
    print("\nNext Steps:")
    print("  1. Run notebook 04a (challenger_validation) to get validation results")
    print("  2. OR: Review model metrics manually")
    print("  3. If satisfied, uncomment 'validation_passed = True' above and re-run")
    
    promotion_status = "pending_manual_review"
else:
    print("\n⚠️  Model NOT promoted")
    print("\nReasons:")
    if not validation_passed:
        print("  - Failed validation tests")
    if has_champion and not auto_promote:
        print("  - Does not improve upon current Champion")
    print("\nNext Steps:")
    print("  1. Review validation failures in notebook 04a")
    print("  2. Retrain model with improved hyperparameters (notebook 02)")
    print("  3. OR: Manually approve if business case supports promotion")
    
    promotion_status = "not_promoted"

# COMMAND ----------

# DBTITLE 1,Model Registry Status
print("\n" + "="*60)
print("MODEL REGISTRY STATUS")
print("="*60)

all_versions = client.search_model_versions(f"name='{model_name}'")

for mv in all_versions:
    # Ensure aliases is always a list
    aliases = getattr(mv, 'aliases', None)
    if not isinstance(aliases, (list, tuple)):
        aliases = []
    alias_str = f" [{', '.join(aliases)}]" if aliases else ""
    print(f"Version {mv.version}{alias_str}")

# COMMAND ----------

# DBTITLE 1,Save Results
dbutils.jobs.taskValues.set(key="promotion_status", value=promotion_status)
print("\n✅ Approval workflow complete")