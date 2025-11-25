# Databricks notebook source
# MAGIC %md
# MAGIC # Model Approval and Promotion Workflow
# MAGIC
# MAGIC <img src="https://github.com/pravinva/mlops-advanced-drift-retrain/raw/main/diagrams/mlops-advanced-4b-approval.png" width="1200px">
# MAGIC
# MAGIC <!-- Tracking -->
# MAGIC <img width="1px" src="https://www.google-analytics.com/collect?v=1&gtm=GTM-NKQ8TT7&tid=UA-163989034-1&cid=555&aip=1&t=event&ec=field_demos&ea=display&dp=%2F42_field_demos%2Fml%2Fmlops_advanced%2F04b_approval&dt=MLOPS_ADVANCED">
# MAGIC
# MAGIC ## Automated Promotion from Challenger to Champion
# MAGIC
# MAGIC This notebook implements the approval workflow for promoting models to production.
# MAGIC
# MAGIC **Promotion Logic:**
# MAGIC 1. Check validation results from previous task
# MAGIC 2. Compare Challenger vs Champion performance
# MAGIC 3. Auto-promote if validation passed AND (no Champion OR improved F1)
# MAGIC 4. Archive old Champion as Previous_Champion
# MAGIC 5. Update model registry aliases

# COMMAND ----------

# DBTITLE 1,Setup and Configuration
dbutils.widgets.dropdown("reset_all_data", "false", ["true", "false"], "Reset all data")

# COMMAND ----------

# MAGIC %run ./_resources/00-setup $reset_all_data=$reset_all_data $catalog=mlops_advanced $db=churn_demo

# COMMAND ----------

# DBTITLE 1,Load Challenger Model
import mlflow
from mlflow.tracking import MlflowClient
from datetime import datetime

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

try:
    challenger_version = client.get_model_version_by_alias(model_name, "Challenger")
    print(f"Challenger Model Found:")
    print(f"  Model: {model_name}")
    print(f"  Version: {challenger_version.version}")
    print(f"  Run ID: {challenger_version.run_id}")
except Exception as e:
    raise Exception(f"No Challenger model found. Run notebooks 02 and 03 first. Error: {e}")

# COMMAND ----------

# DBTITLE 1,Load Validation Results
validation_passed = None

# Try to get validation results from previous task (if running as a job)
try:
    validation_passed = dbutils.jobs.taskValues.get(
        taskKey="challenger_validation",
        key="validation_passed",
        default=None
    )
    print(f"Validation Status (from job): {'PASSED' if validation_passed else 'FAILED'}")
except:
    print("Not running as part of a job, checking for manual validation...")

# If no validation results found
if validation_passed is None:
    print("\n" + "=" * 60)
    print("NO VALIDATION RESULTS FOUND")
    print("=" * 60)
    print("Options:")
    print("  1. Run notebook 04a (challenger_validation) first")
    print("  2. OR set validation_passed manually:")
    print("     validation_passed = True  # If you reviewed and approved")
    print("=" * 60)

# COMMAND ----------

# DBTITLE 1,Manual Override (Optional)
# MAGIC %md
# MAGIC ## Manual Approval Override
# MAGIC
# MAGIC If validation results aren't available but you want to proceed, uncomment:
# MAGIC ```python
# MAGIC validation_passed = True  # Set to True to approve, False to reject
# MAGIC ```

# COMMAND ----------

# Uncomment to manually override validation
# validation_passed = True

# COMMAND ----------

# DBTITLE 1,Get Challenger Metrics
run = mlflow.get_run(challenger_version.run_id)
metrics = run.data.metrics

print("\nChallenger Performance:")
print(f"  F1 Score:  {metrics.get('f1_score', 0):.4f}")
print(f"  Accuracy:  {metrics.get('accuracy', 0):.4f}")
print(f"  Precision: {metrics.get('precision', 0):.4f}")
print(f"  Recall:    {metrics.get('recall', 0):.4f}")
print(f"  ROC AUC:   {metrics.get('roc_auc', 0):.4f}")

# COMMAND ----------

# DBTITLE 1,Compare with Champion
has_champion = False
f1_improvement = 0

try:
    champion_version = client.get_model_version_by_alias(model_name, "Champion")
    champion_run = mlflow.get_run(champion_version.run_id)
    champion_metrics = champion_run.data.metrics

    print("\n" + "=" * 60)
    print("CHAMPION vs CHALLENGER COMPARISON")
    print("=" * 60)
    print(f"{'Metric':<12} {'Champion':<10} {'Challenger':<10} {'Delta':<10}")
    print("-" * 60)

    for metric in ['f1_score', 'accuracy', 'precision', 'recall', 'roc_auc']:
        champ_val = champion_metrics.get(metric, 0)
        chall_val = metrics.get(metric, 0)
        improvement = chall_val - champ_val
        print(f"{metric:<12} {champ_val:<10.4f} {chall_val:<10.4f} {improvement:+.4f}")

    has_champion = True
    f1_improvement = metrics.get('f1_score', 0) - champion_metrics.get('f1_score', 0)

except Exception as e:
    print("\nNo Champion found - this will be the first Champion")
    has_champion = False

# COMMAND ----------

# DBTITLE 1,Approval Decision
auto_promote = False

print("\n" + "=" * 60)
print("APPROVAL DECISION")
print("=" * 60)

# Check if we have validation results
if validation_passed is None:
    print("NO AUTOMATED VALIDATION RESULTS AVAILABLE")
    print("Manual review required before promotion")
    print("\nTo manually approve:")
    print("  1. Review model metrics above")
    print("  2. Uncomment 'validation_passed = True' in previous cell")
    print("  3. Re-run cells below")
    auto_promote = False

elif validation_passed:
    print("Validation: PASSED")

    if has_champion:
        if f1_improvement > 0:
            print(f"Performance: Better than Champion (+{f1_improvement:.4f} F1)")
            print("\nDecision: AUTO-PROMOTE")
            auto_promote = True
        elif f1_improvement > -0.02:
            print(f"Performance: Similar to Champion ({f1_improvement:+.4f} F1)")
            print("Note: Within acceptable 2% degradation threshold")
            print("\nDecision: AUTO-PROMOTE")
            auto_promote = True
        else:
            print(f"Performance: Worse than Champion ({f1_improvement:+.4f} F1)")
            print("\nDecision: MANUAL REVIEW REQUIRED")
            auto_promote = False
    else:
        print("No existing Champion - ready for first deployment")
        print("\nDecision: AUTO-PROMOTE")
        auto_promote = True

else:
    print("Validation: FAILED")
    print("Review validation failures before promoting")
    print("\nDecision: DO NOT PROMOTE")
    auto_promote = False

print("=" * 60)

# COMMAND ----------

# DBTITLE 1,Promote to Champion
if auto_promote:
    print("\n" + "=" * 60)
    print("PROMOTING CHALLENGER TO CHAMPION")
    print("=" * 60)

    # Archive old Champion
    if has_champion:
        print(f"\nStep 1: Archive current Champion")
        print(f"  Old Champion v{champion_version.version} -> Previous_Champion")

        client.set_registered_model_alias(
            name=model_name,
            alias="Previous_Champion",
            version=champion_version.version
        )

        client.delete_registered_model_alias(
            name=model_name,
            alias="Champion"
        )
        print("  Archived")

    # Promote Challenger
    print(f"\nStep 2: Promote Challenger to Champion")
    print(f"  Challenger v{challenger_version.version} -> Champion")

    client.set_registered_model_alias(
        name=model_name,
        alias="Champion",
        version=challenger_version.version
    )
    print("  Promoted")

    # Remove Challenger alias
    print(f"\nStep 3: Remove Challenger alias")
    client.delete_registered_model_alias(
        name=model_name,
        alias="Challenger"
    )
    print("  Removed")

    # Update metadata
    client.set_model_version_tag(
        name=model_name,
        version=challenger_version.version,
        key="promoted_to_champion",
        value=datetime.now().isoformat()
    )

    client.set_model_version_tag(
        name=model_name,
        version=challenger_version.version,
        key="promotion_user",
        value=user
    )

    print("\n" + "=" * 60)
    print("MODEL PROMOTION COMPLETE")
    print("=" * 60)
    print(f"New Champion: Version {challenger_version.version}")
    print(f"F1 Score: {metrics.get('f1_score', 0):.4f}")
    print(f"Status: Ready for production deployment")
    print("=" * 60)

    promotion_status = "promoted"

elif validation_passed is None:
    print("\n" + "=" * 60)
    print("MODEL NOT PROMOTED - MANUAL REVIEW REQUIRED")
    print("=" * 60)
    print("\nReasons:")
    print("  - No automated validation results available")
    print("\nNext Steps:")
    print("  1. Run notebook 04a (challenger_validation)")
    print("  2. OR review model metrics manually")
    print("  3. If satisfied, uncomment 'validation_passed = True' and re-run")
    print("=" * 60)

    promotion_status = "pending_manual_review"

else:
    print("\n" + "=" * 60)
    print("MODEL NOT PROMOTED")
    print("=" * 60)
    print("\nReasons:")
    if not validation_passed:
        print("  - Failed validation tests")
    if has_champion and not auto_promote:
        print(f"  - Performance degradation: {f1_improvement:+.4f} F1")
    print("\nNext Steps:")
    print("  1. Review validation failures in notebook 04a")
    print("  2. Retrain model with improved hyperparameters (notebook 02)")
    print("  3. OR manually approve if business case supports promotion")
    print("=" * 60)

    promotion_status = "not_promoted"

# COMMAND ----------

# DBTITLE 1,Model Registry Status
print("\n" + "=" * 60)
print("MODEL REGISTRY STATUS")
print("=" * 60)

all_versions = client.search_model_versions(f"name='{model_name}'")
all_versions = sorted(all_versions, key=lambda x: int(x.version), reverse=True)

print(f"\n{'Version':<10} {'Aliases':<30} {'Created':<20}")
print("-" * 60)

for mv in all_versions:
    aliases = getattr(mv, 'aliases', [])
    if not isinstance(aliases, (list, tuple)):
        aliases = []
    alias_str = ', '.join(aliases) if aliases else "(no alias)"
    created = datetime.fromtimestamp(mv.creation_timestamp / 1000).strftime('%Y-%m-%d %H:%M')
    print(f"{mv.version:<10} {alias_str:<30} {created:<20}")

print("=" * 60)

# COMMAND ----------

# DBTITLE 1,Save Results for Downstream Tasks
dbutils.jobs.taskValues.set(key="promotion_status", value=promotion_status)
dbutils.jobs.taskValues.set(key="champion_version", value=str(challenger_version.version) if auto_promote else "")

print(f"\nApproval workflow complete")
print(f"  promotion_status: {promotion_status}")
if auto_promote:
    print(f"  champion_version: {challenger_version.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC
# MAGIC Model approval complete. Proceed to:
# MAGIC - **If PROMOTED:** [05 - Batch Inference]($./05_batch_inference)
# MAGIC - **If NOT PROMOTED:** Review and retrain (Notebook 02)
