# Databricks notebook source
# MAGIC %md
# MAGIC # Batch Inference Pipeline

# COMMAND ----------

# MAGIC %run ./notebook-00-setup

# COMMAND ----------

# MAGIC %pip install lightgbm

# COMMAND ----------

# DBTITLE 1,Load Champion Model
import mlflow
from mlflow.tracking import MlflowClient
from pyspark.sql import functions as F
from datetime import datetime

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

try:
    champion_version = client.get_model_version_by_alias(model_name, "Champion")
    model_uri = f"models:/{model_name}@Champion"
    print(f"Loading Champion: {model_name} v{champion_version.version}")
except Exception as e:
    raise Exception(f"No Champion model found: {e}")

# COMMAND ----------

# DBTITLE 1,Load Customer Data
scoring_df = spark.table(features_table)

scoring_df = scoring_df.withColumn("scoring_timestamp", F.lit(datetime.now()))
scoring_df = scoring_df.withColumn("model_version", F.lit(champion_version.version))

print(f"Customers to score: {scoring_df.count()}")

# COMMAND ----------

# DBTITLE 1,Batch Prediction with Spark UDF
from pyspark.sql.functions import struct, col

# Create Spark UDF for distributed scoring
predict_udf = mlflow.pyfunc.spark_udf(
    spark, 
    model_uri=model_uri,
    result_type="double"
)

feature_cols = [c for c in scoring_df.columns 
                if c not in ['customerID', 'churn', 'scoring_timestamp', 'model_version']]

print(f"Scoring with {len(feature_cols)} features")

# COMMAND ----------

# DBTITLE 1,Generate Predictions
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

print("✅ Predictions generated")

display(
    predictions_df.select(
        "customerID", 
        "churn_probability", 
        "churn_prediction", 
        "risk_category",
        "tenure",
        "MonthlyCharges"
    ).orderBy(col("churn_probability").desc())
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

print("\n" + "="*60)
print("PREDICTION STATISTICS")
print("="*60)
print(f"Total Customers: {stats.total_customers:,}")
print(f"Predicted Churners: {stats.predicted_churners:,} ({stats.predicted_churners/stats.total_customers*100:.1f}%)")
print(f"Avg Churn Probability: {stats.avg_churn_probability:.4f}")
print(f"Min Probability: {stats.min_probability:.4f}")
print(f"Max Probability: {stats.max_probability:.4f}")
print("="*60)

# Risk category breakdown
print("\nRisk Category Distribution:")
predictions_df.groupBy("risk_category").agg(
    F.count("*").alias("count"),
    F.avg("churn_probability").alias("avg_probability")
).orderBy(F.desc("avg_probability")).show()

# COMMAND ----------

# DBTITLE 1,Save Predictions
predictions_table = f"{catalog}.{db}.churn_predictions"

predictions_df.write \
    .mode("append") \
    .option("mergeSchema", "true") \
    .saveAsTable(predictions_table)

print(f"✅ Predictions saved to {predictions_table}")
print(f"Total prediction records: {spark.table(predictions_table).count()}")

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

print(f"✅ High-risk list saved to {high_risk_table}")
print(f"   Total high-risk customers: {high_risk_customers.count()}")

print("\nTop 20 Highest Risk Customers:")
display(high_risk_customers.limit(20))

# COMMAND ----------

# DBTITLE 1,Business Impact Analysis
# Calculate business metrics
avg_customer_value = 2000  # $2,000 average lifetime value
intervention_cost = 50     # $50 cost per intervention
success_rate = 0.30        # 30% retention success rate

predicted_churners = int(stats.predicted_churners)
high_risk_count = high_risk_customers.count()

value_at_risk = predicted_churners * avg_customer_value
intervention_budget = high_risk_count * intervention_cost
potential_savings = high_risk_count * avg_customer_value * success_rate
roi = (potential_savings - intervention_budget) / intervention_budget if intervention_budget > 0 else 0

print("\n" + "="*60)
print("BUSINESS IMPACT ANALYSIS")
print("="*60)
print(f"Predicted Churners: {predicted_churners:,}")
print(f"High-Risk Customers: {high_risk_count:,}")
print(f"Value at Risk: ${value_at_risk:,.2f}")
print()
print(f"Retention Campaign Metrics:")
print(f"  Target Customers: {high_risk_count:,}")
print(f"  Intervention Cost: ${intervention_budget:,.2f}")
print(f"  Expected Customers Saved: {int(high_risk_count * success_rate):,}")
print(f"  Potential Value Saved: ${potential_savings:,.2f}")
print(f"  Net Benefit: ${potential_savings - intervention_budget:,.2f}")
print(f"  ROI: {roi:.1%}")
print("="*60)

# COMMAND ----------

# DBTITLE 1,Feature Analysis - High vs Low Risk
# Compare features between risk groups
high_risk_df = predictions_df.filter(col("risk_category") == "High Risk").toPandas()
low_risk_df = predictions_df.filter(col("risk_category") == "Low Risk").toPandas()

if len(high_risk_df) > 0 and len(low_risk_df) > 0:
    numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']
    
    print("\nFeature Comparison: High Risk vs Low Risk")
    print(f"{'Feature':<20} {'High Risk':<15} {'Low Risk':<15} {'Difference':<15}")
    print("-" * 65)
    
    for feature in numeric_features:
        if feature in high_risk_df.columns:
            high_avg = high_risk_df[feature].mean()
            low_avg = low_risk_df[feature].mean()
            diff = high_avg - low_avg
            print(f"{feature:<20} {high_avg:<15.2f} {low_avg:<15.2f} {diff:<+15.2f}")

# COMMAND ----------

# DBTITLE 1,Daily Report
report_date = datetime.now().strftime('%Y-%m-%d')

daily_report = f"""
DAILY CHURN PREDICTION REPORT
{report_date}
{'='*60}

EXECUTIVE SUMMARY:
- Total Customers Scored: {stats.total_customers:,}
- Predicted Churners: {predicted_churners:,} ({predicted_churners/stats.total_customers*100:.1f}%)
- High-Risk Customers: {high_risk_count:,}
- Value at Risk: ${value_at_risk:,.2f}

RISK BREAKDOWN:
- High Risk: {predictions_df.filter(col('risk_category')=='High Risk').count():,} customers
- Medium Risk: {predictions_df.filter(col('risk_category')=='Medium Risk').count():,} customers  
- Low Risk: {predictions_df.filter(col('risk_category')=='Low Risk').count():,} customers

MODEL INFORMATION:
- Model: {model_name}
- Version: {champion_version.version}

RECOMMENDED ACTIONS:
1. Contact top {min(high_risk_count, 100)} high-risk customers this week
2. Budget ${high_risk_count * intervention_cost:,.2f} for retention campaigns
3. Expected value saved: ${potential_savings:,.2f}
4. Review customer feedback for common issues

TABLES UPDATED:
- Predictions: {predictions_table}
- High-Risk List: {high_risk_table}

{'='*60}
Generated by MLOps Batch Inference Pipeline
"""

print(daily_report)

# Save report
report_table = f"{catalog}.{db}.daily_churn_reports"
report_df = spark.createDataFrame([{
    'report_date': datetime.now(),
    'total_customers': int(stats.total_customers),
    'predicted_churners': predicted_churners,
    'high_risk_count': high_risk_count,
    'value_at_risk': float(value_at_risk),
    'model_version': str(champion_version.version),
    'report_text': daily_report
}])

report_df.write.mode("append").saveAsTable(report_table)
print(f"\n✅ Daily report saved to {report_table}")

# COMMAND ----------

# DBTITLE 1,Log Batch Inference Metrics
end_time = datetime.now()

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

print("✅ Batch inference metrics logged to MLflow")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Scheduling Recommendations
# MAGIC
# MAGIC Schedule this notebook to run daily:
# MAGIC - **Frequency:** Daily at 2 AM
# MAGIC - **Cluster:** ML Runtime cluster with lightgbm installed
# MAGIC - **Notifications:** Email to retention team
# MAGIC - **Timeout:** 60 minutes
# MAGIC
# MAGIC This will:
# MAGIC 1. Score all current customers
# MAGIC 2. Generate high-risk lists
# MAGIC 3. Update dashboards
# MAGIC 4. Calculate business metrics