# Databricks notebook source
# MAGIC %md
# MAGIC # Automated Deployment Job
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/raw/main/images/product/mlops/advanced/mlops-advanced-9-deployment.png" width="1200px">
# MAGIC
# MAGIC <!-- Tracking -->
# MAGIC <img width="1px" src="https://www.google-analytics.com/collect?v=1&gtm=GTM-NKQ8TT7&tid=UA-163989034-1&cid=555&aip=1&t=event&ec=field_demos&ea=display&dp=%2F42_field_demos%2Fml%2Fmlops_advanced%2F09_deployment&dt=MLOPS_ADVANCED">
# MAGIC
# MAGIC ## Create Orchestrated Databricks Job for End-to-End Pipeline
# MAGIC
# MAGIC This notebook creates a Databricks Job that orchestrates the entire MLOps workflow.
# MAGIC
# MAGIC **Job Tasks:**
# MAGIC 1. Feature Engineering
# MAGIC 2. Model Training with HPO
# MAGIC 3. Model Registration
# MAGIC 4. Challenger Validation
# MAGIC 5. Challenger Approval
# MAGIC 6. Batch Inference
# MAGIC
# MAGIC **Scheduling Options:**
# MAGIC - Manual trigger for training pipeline
# MAGIC - Daily schedule for inference
# MAGIC - Triggered by drift alerts

# COMMAND ----------

# DBTITLE 1,Setup and Configuration
dbutils.widgets.dropdown("reset_all_data", "false", ["true", "false"], "Reset all data")

# COMMAND ----------

# MAGIC %run ./_resources/00-setup $reset_all_data=$reset_all_data $catalog=mlops_advanced $db=churn_demo

# COMMAND ----------

# DBTITLE 1,Install Databricks SDK
# MAGIC %pip install databricks-sdk --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Re-run Setup After Restart
# MAGIC %run ./_resources/00-setup $reset_all_data=$reset_all_data $catalog=mlops_advanced $db=churn_demo

# COMMAND ----------

# DBTITLE 1,Get Notebook Paths
from databricks.sdk import WorkspaceClient
import os

w = WorkspaceClient()

# Get current notebook path
notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
notebook_dir = "/".join(notebook_path.split("/")[:-1])

print(f"Notebook Directory: {notebook_dir}")

# Define notebook paths
notebooks = {
    "feature_engineering": f"{notebook_dir}/01_feature_engineering",
    "model_training": f"{notebook_dir}/02_model_training_hpo",
    "model_registration": f"{notebook_dir}/03_model_registration",
    "challenger_validation": f"{notebook_dir}/04a_challenger_validation",
    "challenger_approval": f"{notebook_dir}/04b_challenger_approval",
    "batch_inference": f"{notebook_dir}/05_batch_inference"
}

print("\nNotebook Paths:")
for name, path in notebooks.items():
    print(f"  {name}: {path}")

# COMMAND ----------

# DBTITLE 1,Define Job Configuration
from databricks.sdk.service.jobs import *

job_name = f"mlops_churn_pipeline_{username_prefix}"

print(f"Creating job: {job_name}")

# Cluster configuration
cluster_config = JobCluster(
    job_cluster_key="mlops_cluster",
    new_cluster=ClusterSpec(
        spark_version="14.3.x-cpu-ml-scala2.12",
        node_type_id="i3.xlarge",
        num_workers=2,
        data_security_mode=DataSecurityMode.USER_ISOLATION,
        spark_conf={
            "spark.databricks.delta.preview.enabled": "true"
        }
    )
)

# Task dependencies: feature → training → registration → validation → approval → inference
tasks = [
    # Task 1: Feature Engineering
    Task(
        task_key="feature_engineering",
        job_cluster_key="mlops_cluster",
        notebook_task=NotebookTask(
            notebook_path=notebooks["feature_engineering"],
            base_parameters={"reset_all_data": "false"}
        ),
        timeout_seconds=3600,
        max_retries=1
    ),

    # Task 2: Model Training
    Task(
        task_key="model_training",
        depends_on=[TaskDependency(task_key="feature_engineering")],
        job_cluster_key="mlops_cluster",
        notebook_task=NotebookTask(
            notebook_path=notebooks["model_training"],
            base_parameters={"reset_all_data": "false"}
        ),
        timeout_seconds=7200,
        max_retries=1,
        libraries=[
            Library(pypi=PyPiLibrary(package="optuna")),
            Library(pypi=PyPiLibrary(package="lightgbm")),
            Library(pypi=PyPiLibrary(package="scikit-learn"))
        ]
    ),

    # Task 3: Model Registration
    Task(
        task_key="model_registration",
        depends_on=[TaskDependency(task_key="model_training")],
        job_cluster_key="mlops_cluster",
        notebook_task=NotebookTask(
            notebook_path=notebooks["model_registration"],
            base_parameters={"reset_all_data": "false"}
        ),
        timeout_seconds=1800,
        max_retries=1,
        libraries=[
            Library(pypi=PyPiLibrary(package="lightgbm"))
        ]
    ),

    # Task 4: Challenger Validation
    Task(
        task_key="challenger_validation",
        depends_on=[TaskDependency(task_key="model_registration")],
        job_cluster_key="mlops_cluster",
        notebook_task=NotebookTask(
            notebook_path=notebooks["challenger_validation"],
            base_parameters={"reset_all_data": "false"}
        ),
        timeout_seconds=1800,
        max_retries=1,
        libraries=[
            Library(pypi=PyPiLibrary(package="lightgbm")),
            Library(pypi=PyPiLibrary(package="scikit-learn"))
        ]
    ),

    # Task 5: Challenger Approval
    Task(
        task_key="challenger_approval",
        depends_on=[TaskDependency(task_key="challenger_validation")],
        job_cluster_key="mlops_cluster",
        notebook_task=NotebookTask(
            notebook_path=notebooks["challenger_approval"],
            base_parameters={"reset_all_data": "false"}
        ),
        timeout_seconds=1200,
        max_retries=1
    ),

    # Task 6: Batch Inference
    Task(
        task_key="batch_inference",
        depends_on=[TaskDependency(task_key="challenger_approval")],
        job_cluster_key="mlops_cluster",
        notebook_task=NotebookTask(
            notebook_path=notebooks["batch_inference"],
            base_parameters={"reset_all_data": "false"}
        ),
        timeout_seconds=1800,
        max_retries=1,
        libraries=[
            Library(pypi=PyPiLibrary(package="lightgbm"))
        ]
    )
]

print(f"\nConfigured {len(tasks)} tasks with dependencies")

# COMMAND ----------

# DBTITLE 1,Create or Update Job
print("\nCreating/updating Databricks Job...")

# Check if job already exists
existing_jobs = w.jobs.list(name=job_name)
existing_job_id = None

for job in existing_jobs:
    if job.settings.name == job_name:
        existing_job_id = job.job_id
        break

if existing_job_id:
    print(f"Updating existing job: {job_name} (ID: {existing_job_id})")

    # Update job
    w.jobs.reset(
        job_id=existing_job_id,
        new_settings=JobSettings(
            name=job_name,
            tasks=tasks,
            job_clusters=[cluster_config],
            timeout_seconds=14400,  # 4 hours
            max_concurrent_runs=1,
            email_notifications=JobEmailNotifications(
                on_success=[user],
                on_failure=[user]
            ),
            format=Format.MULTI_TASK
        )
    )

    job_id = existing_job_id
    print(f"Job updated successfully!")

else:
    print(f"Creating new job: {job_name}")

    # Create job
    created_job = w.jobs.create(
        name=job_name,
        tasks=tasks,
        job_clusters=[cluster_config],
        timeout_seconds=14400,
        max_concurrent_runs=1,
        email_notifications=JobEmailNotifications(
            on_success=[user],
            on_failure=[user]
        ),
        format=Format.MULTI_TASK
    )

    job_id = created_job.job_id
    print(f"Job created successfully!")

# COMMAND ----------

# DBTITLE 1,Job Information
workspace_url = spark.conf.get("spark.databricks.workspaceUrl")
job_url = f"https://{workspace_url}/#job/{job_id}"

print("\n" + "=" * 70)
print("DATABRICKS JOB CREATED")
print("=" * 70)
print(f"Job Name: {job_name}")
print(f"Job ID: {job_id}")
print(f"Job URL: {job_url}")
print()
print("Job Configuration:")
print(f"  Tasks: {len(tasks)}")
print(f"  Cluster: ML Runtime 14.3 (2 workers)")
print(f"  Timeout: 4 hours")
print(f"  Max Concurrent Runs: 1")
print()
print("Task Dependencies:")
print("  1. Feature Engineering")
print("  2. Model Training (depends on 1)")
print("  3. Model Registration (depends on 2)")
print("  4. Challenger Validation (depends on 3)")
print("  5. Challenger Approval (depends on 4)")
print("  6. Batch Inference (depends on 5)")
print("=" * 70)

# COMMAND ----------

# DBTITLE 1,Add Job Schedule (Optional)
# MAGIC %md
# MAGIC ## Optional: Add Schedule
# MAGIC
# MAGIC Uncomment the cell below to add a schedule to the job.
# MAGIC
# MAGIC **Scheduling Options:**
# MAGIC - **Weekly retraining:** `0 0 0 ? * SUN` (Every Sunday at midnight)
# MAGIC - **Daily inference:** Create separate job for just batch_inference task
# MAGIC - **Monthly:** `0 0 0 1 * ?` (First day of month at midnight)

# COMMAND ----------

# # Add weekly schedule (Sundays at midnight)
# w.jobs.update(
#     job_id=job_id,
#     new_settings=JobSettings(
#         name=job_name,
#         tasks=tasks,
#         job_clusters=[cluster_config],
#         timeout_seconds=14400,
#         max_concurrent_runs=1,
#         email_notifications=JobEmailNotifications(
#             on_success=[user],
#             on_failure=[user]
#         ),
#         schedule=CronSchedule(
#             quartz_cron_expression="0 0 0 ? * SUN",
#             timezone_id="UTC",
#             pause_status=PauseStatus.UNPAUSED
#         ),
#         format=Format.MULTI_TASK
#     )
# )
#
# print("Weekly schedule added (Sundays at midnight UTC)")

# COMMAND ----------

# DBTITLE 1,Test Run Job (Optional)
# MAGIC %md
# MAGIC ## Optional: Trigger Test Run
# MAGIC
# MAGIC Uncomment to trigger a test run of the job.

# COMMAND ----------

# print("Triggering test run...")
#
# run = w.jobs.run_now(job_id=job_id)
#
# run_url = f"https://{workspace_url}/#job/{job_id}/run/{run.run_id}"
#
# print(f"Job run started!")
# print(f"Run ID: {run.run_id}")
# print(f"Run URL: {run_url}")
# print("\nMonitor progress in the Databricks UI")

# COMMAND ----------

# DBTITLE 1,Save Job Configuration
dbutils.jobs.taskValues.set(key="job_id", value=str(job_id))
dbutils.jobs.taskValues.set(key="job_name", value=job_name)
dbutils.jobs.taskValues.set(key="job_url", value=job_url)

print("\nJob configuration saved")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deployment Complete
# MAGIC
# MAGIC ### What Was Created:
# MAGIC - Databricks Job with 6 orchestrated tasks
# MAGIC - Task dependencies for proper execution order
# MAGIC - Email notifications on success/failure
# MAGIC - Shared cluster for all tasks
# MAGIC
# MAGIC ### Usage:
# MAGIC 1. **Manual Trigger:** Click "Run Now" in Databricks Jobs UI
# MAGIC 2. **Scheduled Run:** Add cron schedule (see optional cell above)
# MAGIC 3. **API Trigger:** Use Databricks REST API or SDK
# MAGIC
# MAGIC ### Monitoring:
# MAGIC - View job runs in [Databricks Jobs UI]({job_url})
# MAGIC - Email notifications configured for {user}
# MAGIC - Task-level logs and metrics available
# MAGIC
# MAGIC ### Next Steps:
# MAGIC - Run the job to test end-to-end workflow
# MAGIC - Add schedule for automated retraining
# MAGIC - Integrate with CI/CD pipelines
# MAGIC - Set up alerting for job failures
