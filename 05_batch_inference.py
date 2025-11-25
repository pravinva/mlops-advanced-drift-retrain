# Databricks notebook source
# MAGIC %md
# MAGIC # Batch Inference at Scale
# MAGIC
# MAGIC <img src="https://github.com/pravinva/mlops-advanced-drift-retrain/raw/main/diagrams/mlops-advanced-5-inference.png" width="1200px">
# MAGIC
# MAGIC <!-- Tracking -->
# MAGIC <img width="1px" src="https://www.google-analytics.com/collect?v=1&gtm=GTM-NKQ8TT7&tid=UA-163989034-1&cid=555&aip=1&t=event&ec=field_demos&ea=display&dp=%2F42_field_demos%2Fml%2Fmlops_advanced%2F05_inference&dt=MLOPS_ADVANCED">
# MAGIC
# MAGIC ## Distributed Batch Scoring with Business Impact Analysis
# MAGIC
# MAGIC This notebook demonstrates enterprise-grade batch inference using Spark UDFs for distributed scoring.
# MAGIC
# MAGIC **What we'll do:**
# MAGIC 1. Load Champion model from Unity Catalog
# MAGIC 2. Score all customers using distributed Spark UDFs
# MAGIC 3. Calculate risk categories and business metrics
# MAGIC 4. Generate high-risk customer lists
# MAGIC 5. Perform business impact analysis (ROI, value at risk)
# MAGIC 6. Save predictions for monitoring

# COMMAND ----------

# DBTITLE 1,Setup and Configuration
dbutils.widgets.dropdown("reset_all_data", "false", ["true", "false"], "Reset all data")

# COMMAND ----------

# MAGIC %run ./_resources/00-setup $reset_all_data=$reset_all_data $catalog=mlops_advanced $db=churn_demo

# COMMAND ----------

# DBTITLE 1,Install Required Libraries
# MAGIC %pip install lightgbm --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Re-run Setup After Restart
# MAGIC %run ./_resources/00-setup $reset_all_data=$reset_all_data $catalog=mlops_advanced $db=churn_demo

# COMMAND ----------

# DBTITLE 1,Load Champion Model
import mlflow
from mlflow.tracking import MlflowClient
from pyspark.sql import functions as F
from pyspark.sql.functions import struct, col
from datetime import datetime

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

try:
    champion_version = client.get_model_version_by_alias(model_name, "Champion")
    model_uri = f"models:/{model_name}@Champion"

    print(f"Champion Model Loaded:")
    print(f"  Model: {model_name}")
    print(f"  Version: {champion_version.version}")
    print(f"  URI: {model_uri}")

except Exception as e:
    raise Exception(f"No Champion model found. Run notebooks 02-04b first. Error: {e}")

# COMMAND ----------

# DBTITLE 1,Load Customer Data for Scoring
scoring_df = spark.table(features_table)

# Add metadata columns
scoring_df = scoring_df.withColumn("scoring_timestamp", F.lit(datetime.now()))
scoring_df = scoring_df.withColumn("model_version", F.lit(str(champion_version.version)))

customer_count = scoring_df.count()
print(f"Customers to score: {customer_count:,}")

# COMMAND ----------

# DBTITLE 1,Create Distributed Prediction UDF
# Create Spark UDF for distributed scoring across cluster
predict_udf = mlflow.pyfunc.spark_udf(
    spark,
    model_uri=model_uri,
    result_type="double"
)

# Get feature columns
feature_cols = [c for c in scoring_df.columns
                if c not in ['customerID', 'churn', 'scoring_timestamp', 'model_version']]

print(f"Scoring with {len(feature_cols)} features")
print("\nDistributed scoring will parallelize across Spark partitions")

# COMMAND ----------

# DBTITLE 1,Generate Predictions at Scale
print("Generating predictions...")

predictions_df = scoring_df.withColumn(
    "churn_probability",
    predict_udf(struct(*[col(c) for c in feature_cols]))
)

predictions_df = predictions_df.withColumn(
    "churn_prediction",
    F.when(col("churn_probability") >= 0.5, 1).otherwise(0)
)

predictions_df = predictions_df.withColumn(
    "risk_category",
    F.when(col("churn_probability") >= 0.7, "High Risk")
     .when(col("churn_probability") >= 0.4, "Medium Risk")
     .otherwise("Low Risk")
)

print("Predictions generated")

# Show top 20 highest risk customers
print("\nTop 20 Highest Risk Customers:")
display(
    predictions_df
    .select("customerID", "churn_probability", "churn_prediction", "risk_category", "tenure", "MonthlyCharges")
    .orderBy(col("churn_probability").desc())
    .limit(20)
)

# COMMAND ----------

# DBTITLE 1,Prediction Statistics
stats = predictions_df.select([
    F.count("*").alias("total_customers"),
    F.sum("churn_prediction").alias("predicted_churners"),
    F.avg("churn_probability").alias("avg_churn_probability"),
    F.min("churn_probability").alias("min_probability"),
    F.max("churn_probability").alias("max_probability")
]).collect()[0]

print("\n" + "=" * 60)
print("PREDICTION STATISTICS")
print("=" * 60)
print(f"Total Customers:        {stats.total_customers:,}")
print(f"Predicted Churners:     {stats.predicted_churners:,} ({stats.predicted_churners/stats.total_customers*100:.1f}%)")
print(f"Avg Churn Probability:  {stats.avg_churn_probability:.4f}")
print(f"Min Probability:        {stats.min_probability:.4f}")
print(f"Max Probability:        {stats.max_probability:.4f}")
print("=" * 60)

# Risk category breakdown
print("\nRisk Category Distribution:")
predictions_df.groupBy("risk_category").agg(
    F.count("*").alias("count"),
    F.avg("churn_probability").alias("avg_probability")
).orderBy(F.desc("avg_probability")).show()

# COMMAND ----------

# DBTITLE 1,Save Predictions Table
predictions_df.write \
    .mode("append") \
    .option("mergeSchema", "true") \
    .saveAsTable(predictions_table)

print(f"Predictions saved to {predictions_table}")
print(f"Total prediction records: {spark.table(predictions_table).count():,}")

# COMMAND ----------

# DBTITLE 1,Create High-Risk Customer List
high_risk_customers = predictions_df.filter(
    col("risk_category") == "High Risk"
).select(
    "customerID",
    "churn_probability",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "scoring_timestamp"
).orderBy(col("churn_probability").desc())

high_risk_table = f"{catalog}.{db}.high_risk_customers"
high_risk_customers.write.mode("overwrite").saveAsTable(high_risk_table)

high_risk_count = high_risk_customers.count()
print(f"High-risk list saved to {high_risk_table}")
print(f"Total high-risk customers: {high_risk_count:,}")

print("\nTop 20 Highest Risk Customers:")
display(high_risk_customers.limit(20))

# COMMAND ----------

# DBTITLE 1,Business Impact Analysis
# Business assumptions
avg_customer_value = 2000  # $2,000 average lifetime value
intervention_cost = 50     # $50 cost per retention intervention
success_rate = 0.30        # 30% retention success rate

predicted_churners = int(stats.predicted_churners)

# Calculate business metrics
value_at_risk = predicted_churners * avg_customer_value
intervention_budget = high_risk_count * intervention_cost
expected_customers_saved = int(high_risk_count * success_rate)
potential_savings = high_risk_count * avg_customer_value * success_rate
net_benefit = potential_savings - intervention_budget
roi = (net_benefit / intervention_budget * 100) if intervention_budget > 0 else 0

print("\n" + "=" * 60)
print("BUSINESS IMPACT ANALYSIS")
print("=" * 60)
print(f"\nChurn Risk:")
print(f"  Predicted Churners:     {predicted_churners:,}")
print(f"  High-Risk Customers:    {high_risk_count:,}")
print(f"  Total Value at Risk:    ${value_at_risk:,.2f}")

print(f"\nRetention Campaign:")
print(f"  Target Customers:       {high_risk_count:,}")
print(f"  Intervention Cost:      ${intervention_budget:,.2f}")
print(f"  Expected Success Rate:  {success_rate:.0%}")
print(f"  Expected Saves:         {expected_customers_saved:,} customers")

print(f"\nFinancial Impact:")
print(f"  Potential Value Saved:  ${potential_savings:,.2f}")
print(f"  Campaign Cost:          ${intervention_budget:,.2f}")
print(f"  Net Benefit:            ${net_benefit:,.2f}")
print(f"  ROI:                    {roi:.1f}%")
print("=" * 60)

# COMMAND ----------

# DBTITLE 1,Feature Analysis - Risk Comparison
high_risk_df = predictions_df.filter(col("risk_category") == "High Risk").toPandas()
low_risk_df = predictions_df.filter(col("risk_category") == "Low Risk").toPandas()

if len(high_risk_df) > 0 and len(low_risk_df) > 0:
    numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']

    print("\n" + "=" * 65)
    print("FEATURE COMPARISON: High Risk vs Low Risk")
    print("=" * 65)
    print(f"{'Feature':<20} {'High Risk':<15} {'Low Risk':<15} {'Difference':<15}")
    print("-" * 65)

    for feature in numeric_features:
        if feature in high_risk_df.columns:
            high_avg = high_risk_df[feature].mean()
            low_avg = low_risk_df[feature].mean()
            diff = high_avg - low_avg
            print(f"{feature:<20} {high_avg:<15.2f} {low_avg:<15.2f} {diff:<+15.2f}")

    print("=" * 65)

# COMMAND ----------

# DBTITLE 1,Generate Daily Report
report_date = datetime.now().strftime('%Y-%m-%d')

daily_report = f"""
{'='*60}
DAILY CHURN PREDICTION REPORT
{report_date}
{'='*60}

EXECUTIVE SUMMARY:
- Total Customers Scored:   {stats.total_customers:,}
- Predicted Churners:        {predicted_churners:,} ({predicted_churners/stats.total_customers*100:.1f}%)
- High-Risk Customers:       {high_risk_count:,}
- Total Value at Risk:       ${value_at_risk:,.2f}

RISK BREAKDOWN:
- High Risk:                 {predictions_df.filter(col('risk_category')=='High Risk').count():,} customers
- Medium Risk:               {predictions_df.filter(col('risk_category')=='Medium Risk').count():,} customers
- Low Risk:                  {predictions_df.filter(col('risk_category')=='Low Risk').count():,} customers

RETENTION CAMPAIGN FINANCIALS:
- Campaign Budget:           ${intervention_budget:,.2f}
- Expected Value Saved:      ${potential_savings:,.2f}
- Net Benefit:               ${net_benefit:,.2f}
- ROI:                       {roi:.1f}%

MODEL INFORMATION:
- Model:                     {model_name}
- Version:                   {champion_version.version}
- Scoring Date:              {report_date}

RECOMMENDED ACTIONS:
1. Contact top {min(high_risk_count, 100)} high-risk customers this week
2. Allocate ${intervention_budget:,.2f} for retention campaigns
3. Expected outcome: Save {expected_customers_saved:,} customers
4. Monitor customer feedback for common pain points

DATA ASSETS:
- Predictions Table:         {predictions_table}
- High-Risk List:            {high_risk_table}

{'='*60}
Generated by Databricks MLOps Batch Inference Pipeline
"""

print(daily_report)

# Save report to table
report_table = f"{catalog}.{db}.daily_churn_reports"
report_df = spark.createDataFrame([{
    'report_date': datetime.now(),
    'total_customers': int(stats.total_customers),
    'predicted_churners': predicted_churners,
    'high_risk_count': high_risk_count,
    'value_at_risk': float(value_at_risk),
    'potential_savings': float(potential_savings),
    'roi': float(roi),
    'model_version': str(champion_version.version),
    'report_text': daily_report
}])

report_df.write.mode("append").saveAsTable(report_table)
print(f"\nDaily report saved to {report_table}")

# COMMAND ----------

# DBTITLE 1,Log Inference Metrics to MLflow
with mlflow.start_run(run_name=f"batch_inference_{report_date}") as run:
    mlflow.log_param("model_name", model_name)
    mlflow.log_param("model_version", champion_version.version)
    mlflow.log_param("scoring_date", report_date)
    mlflow.log_param("customers_scored", stats.total_customers)

    mlflow.log_metric("predicted_churners", predicted_churners)
    mlflow.log_metric("high_risk_count", high_risk_count)
    mlflow.log_metric("avg_churn_probability", stats.avg_churn_probability)
    mlflow.log_metric("value_at_risk", value_at_risk)
    mlflow.log_metric("potential_savings", potential_savings)
    mlflow.log_metric("roi", roi)

print("Batch inference metrics logged to MLflow")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Scheduling Recommendations
# MAGIC
# MAGIC Configure this notebook as a scheduled Databricks Job:
# MAGIC
# MAGIC ### Schedule Configuration
# MAGIC - **Frequency:** Daily at 2 AM
# MAGIC - **Cluster:** ML Runtime 14.3+ with LightGBM
# MAGIC - **Timeout:** 60 minutes
# MAGIC - **Notifications:** Email to retention team on success/failure
# MAGIC
# MAGIC ### Automated Workflow
# MAGIC This pipeline will:
# MAGIC 1. Load latest Champion model from Unity Catalog
# MAGIC 2. Score all customers using distributed Spark
# MAGIC 3. Generate high-risk customer lists
# MAGIC 4. Calculate business impact metrics
# MAGIC 5. Update prediction tables for downstream dashboards
# MAGIC 6. Log metrics to MLflow for tracking
# MAGIC
# MAGIC ## Next Steps
# MAGIC
# MAGIC Batch scoring complete. Proceed to:
# MAGIC - **Next:** [06 - Model Serving]($./06_model_serving) (optional real-time endpoint)
# MAGIC - **Or:** [07 - Model Monitoring]($./07_model_monitoring) (production monitoring)
