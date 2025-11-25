# Databricks notebook source
# MAGIC %md
# MAGIC # Create Deployment Job

# COMMAND ----------

# MAGIC %run ./notebook-00-setup

# COMMAND ----------

# DBTITLE 1,Setup
# Define user variable
user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()

# Load configuration from setup notebook
catalog = "main"
db = "mlops_churn_demo"

print(f"User: {user}")
print(f"Catalog: {catalog}")
print(f"Database: {db}")

# COMMAND ----------

# MAGIC %pip install databricks-sdk --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Job Configuration
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import jobs, compute

w = WorkspaceClient()
workspace_url = spark.conf.get("spark.databricks.workspaceUrl")

job_name = f"mlops_churn_deployment_{user.split('@')[0]}"

# Build tasks as SDK objects
task_list = [
    jobs.Task(
        task_key="feature_engineering",
        notebook_task=jobs.NotebookTask(
            notebook_path=f"/Users/{user}/mlops-advanced/01_feature_engineering",
            source=jobs.Source.WORKSPACE
        ),
        job_cluster_key="ml_cluster",
        timeout_seconds=3600
    ),
    jobs.Task(
        task_key="model_training",
        depends_on=[jobs.TaskDependency(task_key="feature_engineering")],
        notebook_task=jobs.NotebookTask(
            notebook_path=f"/Users/{user}/mlops-advanced/02_model_training_hpo_optuna",
            source=jobs.Source.WORKSPACE
        ),
        job_cluster_key="ml_cluster",
        timeout_seconds=7200
    ),
    jobs.Task(
        task_key="model_registration",
        depends_on=[jobs.TaskDependency(task_key="model_training")],
        notebook_task=jobs.NotebookTask(
            notebook_path=f"/Users/{user}/mlops-advanced/03b_from_notebook_to_models_in_uc",
            source=jobs.Source.WORKSPACE
        ),
        job_cluster_key="ml_cluster",
        timeout_seconds=1800
    ),
    jobs.Task(
        task_key="model_validation",
        depends_on=[jobs.TaskDependency(task_key="model_registration")],
        notebook_task=jobs.NotebookTask(
            notebook_path=f"/Users/{user}/mlops-advanced/04a_challenger_validation",
            source=jobs.Source.WORKSPACE
        ),
        job_cluster_key="ml_cluster",
        timeout_seconds=1800
    )
]

# Build job clusters as SDK objects
cluster_list = [
    jobs.JobCluster(
        job_cluster_key="ml_cluster",
        new_cluster=compute.ClusterSpec(
            spark_version="14.3.x-cpu-ml-scala2.12",
            node_type_id="i3.xlarge",
            num_workers=2,
            data_security_mode=compute.DataSecurityMode.USER_ISOLATION
        )
    )
]

# Define the job_config dictionary with all required fields for job creation/update
job_config = {
    "name": job_name,
    "tasks": task_list,
    "job_clusters": cluster_list,
    "max_concurrent_runs": 1,
    "tags": {"RemoveAfter": "2024-12-31"}
}

print(f"Job configuration ready: {job_name}")

# COMMAND ----------

# DBTITLE 1,Create or Update Job
job_name = job_config["name"]
existing_jobs = w.jobs.list(name=job_name)

job_id = None
for job in existing_jobs:
    if job.settings.name == job_name:
        job_id = job.job_id
        break

if job_id:
    print(f"Updating job {job_id}...")
    w.jobs.reset(job_id=job_id, new_settings=jobs.JobSettings(**job_config))
    print(f"✅ Job updated")
else:
    print(f"Creating new job: {job_name}")
    created_job = w.jobs.create(**job_config)
    job_id = created_job.job_id
    print(f"✅ Job created with ID: {job_id}")

job_url = f"https://{workspace_url}/#job/{job_id}"
print(f"\nJob URL: {job_url}")
displayHTML(f'<a href="{job_url}" target="_blank">Open Deployment Job</a>')