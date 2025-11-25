# MLOps Advanced: Drift Detection and Automated Retraining

End-to-end MLOps pipeline demonstrating production-grade ML workflows with drift detection, model monitoring, and automated retraining capabilities. Built on Databricks with Unity Catalog integration.

## Overview

This repository contains a complete MLOps implementation for customer churn prediction, covering the entire machine learning lifecycle from data preparation through production monitoring and automated retraining. The implementation demonstrates best practices for enterprise ML systems including model governance, quality gates, drift detection, and automated remediation.

## Architecture

The pipeline follows the Databricks MLOps lifecycle:

1. Data Preparation & Feature Engineering
2. Model Training with Hyperparameter Optimization
3. Model Registration to Unity Catalog
4. Automated Validation & Quality Gates
5. Promotion Workflow (Challenger to Champion)
6. Batch Inference at Scale
7. Real-time Model Serving
8. Production Monitoring with Drift Detection
9. Automated Retraining Triggers

## Notebooks

### notebook-00-setup.py

**Purpose:** Configuration and initialization for the complete MLOps pipeline

**Key Functions:**
- Sets up Unity Catalog namespace (catalog and schema)
- Defines global configuration variables for model name, experiment path, and table names
- Loads sample telco customer churn dataset from IBM repository
- Creates bronze layer table with raw customer data
- Enables Change Data Feed for table-level change tracking
- Configures MLflow experiment tracking and Unity Catalog registry integration

**Outputs:**
- Catalog: `art_mlops`
- Schema: `mlops_churn_demo`
- Bronze table: `art_mlops.mlops_churn_demo.bronze_customers`
- Model name: `art_mlops.mlops_churn_demo.churn_model`
- Experiment path: `/Users/{user}/mlops_churn_experiments`

**Usage:** This notebook is executed via `%run ./notebook-00-setup` in all subsequent notebooks to inherit configuration.

---

### notebook-01-features.py

**Purpose:** Feature engineering pipeline to transform raw customer data into ML-ready features

**Key Operations:**
- Loads raw data from bronze table
- Cleans TotalCharges column (handles missing values as "0")
- Casts numeric columns to appropriate types (int, double)
- Performs one-hot encoding for 15 categorical columns (gender, Partner, Dependents, PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod)
- Creates binary churn label from categorical Churn column
- Cleans column names (removes spaces, parentheses, hyphens)
- Handles missing values with zero-fill strategy

**Outputs:**
- Feature table: `art_mlops.mlops_churn_demo.churn_features`
- Training table: `art_mlops.mlops_churn_demo.churn_features_train` (80% split)
- Test table: `art_mlops.mlops_churn_demo.churn_features_test` (20% split)
- Change Data Feed enabled for feature drift tracking

**Features Generated:**
- Numeric: tenure, MonthlyCharges, TotalCharges, SeniorCitizen
- One-hot encoded: 50+ binary columns from categorical variables
- Target: churn (binary: 0 or 1)

---

### notebook-02-training.py

**Purpose:** Model training with Optuna-based hyperparameter optimization

**Key Operations:**
- Loads training and test data from feature tables
- Runs Optuna study with 50 trials to optimize hyperparameters
- Uses LightGBM as the base model with binary classification objective
- Optimizes for F1 score (balanced metric for imbalanced churn dataset)
- Tracks all trials in MLflow with nested runs
- Trains final model with best hyperparameters
- Logs model artifact with signature and input examples

**Hyperparameters Optimized:**
- num_leaves: [20, 150]
- learning_rate: [0.01, 0.3]
- feature_fraction: [0.5, 1.0]
- bagging_fraction: [0.5, 1.0]
- bagging_freq: [1, 7]
- min_child_samples: [5, 100]
- max_depth: [3, 12]
- lambda_l1: [0.0, 10.0]
- lambda_l2: [0.0, 10.0]

**Metrics Logged:**
- F1 Score
- Accuracy
- Precision
- Recall
- ROC AUC

**Outputs:**
- MLflow run with final model artifact
- Run ID saved to task values for downstream notebooks
- Model ready for registration to Unity Catalog

---

### notebook-03a-deployment.py

**Purpose:** Create automated deployment job orchestrating the full ML pipeline

**Key Operations:**
- Defines multi-task Databricks Job using Databricks SDK
- Configures job cluster with ML runtime (14.3.x-cpu-ml-scala2.12)
- Sets up task dependencies for proper execution order
- Creates or updates existing job (idempotent operation)

**Job Tasks:**
1. Feature Engineering (runs notebook-01-features.py)
2. Model Training (runs 02_model_training_hpo_optuna, depends on #1)
3. Model Registration (runs 03b_from_notebook_to_models_in_uc, depends on #2)
4. Model Validation (runs 04a_challenger_validation, depends on #3)

**Cluster Configuration:**
- Node type: i3.xlarge
- Workers: 2
- Data security mode: USER_ISOLATION
- Max concurrent runs: 1

**Outputs:**
- Job ID and URL for monitoring
- Automated workflow for model updates

---

### notebook-03b-registration.py

**Purpose:** Register trained model to Unity Catalog with governance metadata

**Key Operations:**
- Searches MLflow experiment for best model run (tagged as 'final_champion_model')
- Registers model to Unity Catalog with three-level namespace
- Sets comprehensive model version tags (model_type, use_case, training_date, f1_score)
- Adds detailed model description with performance metrics
- Assigns "Challenger" alias for staging before production
- Adds data lineage tags (feature_table, training_table)
- Tests model loading from Unity Catalog to verify accessibility

**Model Metadata:**
- Name: `art_mlops.mlops_churn_demo.churn_model`
- Type: LightGBM Classifier
- Alias: Challenger (initial registration)
- Tags: Lineage, performance metrics, training metadata

**Outputs:**
- Model version number in Unity Catalog
- Model version saved to task values
- Model accessible via `models:/{model_name}@Challenger`

---

### notebook-04a-validation.py

**Purpose:** Comprehensive automated validation of Challenger model before production promotion

**Test Suite:**

**Test 1: Performance Metrics**
- Validates accuracy >= 70%
- Validates precision >= 55%
- Validates recall >= 50%
- Validates F1 score >= 55%
- Validates ROC AUC >= 75%
- Quality gates calibrated for imbalanced churn datasets

**Test 2: Confusion Matrix Analysis**
- Calculates false negative rate (FNR)
- Validates FNR <= 50% (acceptable miss rate for business case)
- Calculates false positive rate (FPR)
- Ensures business cost of false alarms is manageable

**Test 3: Class-wise Performance**
- Validates minority class (Churn) F1 score >= 50%
- Critical for imbalanced datasets where churn class is underrepresented
- Ensures model doesn't simply predict majority class

**Test 4: Prediction Distribution**
- Validates minority class prediction rate >= 10%
- Prevents degenerate models that predict only one class
- Ensures balanced prediction distribution

**Test 5: Inference Latency**
- Measures single-record prediction latency
- Validates average latency <= 50ms
- Ensures production performance requirements

**Test 6: Champion Comparison**
- Compares Challenger F1 score with current Champion
- Allows up to -2% degradation (prevents regression)
- Skipped if no Champion exists (first deployment)

**Test 7: Robustness Testing**
- Introduces 10% random missing values
- Validates model handles missing data gracefully
- Checks prediction validity (range [0,1], reasonable mean, variance)
- Tests dtype handling for mixed int/float features

**Outputs:**
- Boolean validation result saved to task values
- Detailed test report with pass/fail status
- Recommendations for manual review if tests fail

**Quality Gate Philosophy:**
- Realistic thresholds for production churn models
- Balance between model performance and business requirements
- Allow minor degradation to adapt to data distribution shifts

---

### notebook-04b-approval.py

**Purpose:** Automated approval workflow to promote Challenger to Champion

**Decision Logic:**

1. **Validation Check:**
   - Retrieves validation results from previous task or MLflow
   - If no validation results found, requires manual review
   - Supports manual override for edge cases

2. **Champion Comparison:**
   - Loads current Champion metrics (if exists)
   - Compares F1 scores between Challenger and Champion
   - Calculates improvement across all metrics

3. **Promotion Criteria:**
   - Auto-promote if: validation passed AND (F1 improved OR no Champion exists)
   - Manual review if: validation passed but F1 degraded
   - Block promotion if: validation failed

**Promotion Actions:**
1. Archives old Champion as "Previous_Champion" alias
2. Removes "Champion" alias from old model
3. Assigns "Champion" alias to Challenger model
4. Removes "Challenger" alias
5. Adds promotion timestamp tag
6. Updates model version metadata

**Outputs:**
- Promotion status: "promoted" | "pending_manual_review" | "not_promoted"
- Model registry with updated aliases
- Audit trail via model version tags

**Rollback Support:**
- Previous Champion retained with alias
- One-click rollback by reassigning "Champion" to previous version

---

### notebook-05-batch-inference.py

**Purpose:** Production batch scoring pipeline for weekly churn predictions

**Batch Inference Strategy:**
- No real-time endpoint (cost-efficient for weekly use case)
- Distributed scoring using Spark UDFs
- Loads Champion model dynamically via alias
- Scales to millions of customers (2.4M+ demonstrated)

**Key Operations:**
1. Loads Champion model from Unity Catalog
2. Loads all customers from feature table
3. Creates Spark UDF from MLflow model for distributed scoring
4. Generates predictions with churn probability and risk category
5. Calculates business impact metrics
6. Creates high-risk customer list for retention campaigns

**Risk Categories:**
- High Risk: churn_probability >= 0.7
- Medium Risk: churn_probability >= 0.4
- Low Risk: churn_probability < 0.4

**Business Metrics:**
- Value at risk (predicted churners x avg customer lifetime value)
- Intervention budget (high-risk customers x cost per intervention)
- Expected customers saved (high-risk count x retention success rate)
- ROI calculation for retention campaigns

**Outputs:**
- Predictions table: `art_mlops.mlops_churn_demo.churn_predictions`
- High-risk customer list: `art_mlops.mlops_churn_demo.high_risk_customers`
- Daily report table: `art_mlops.mlops_churn_demo.daily_churn_reports`
- MLflow run with batch inference metrics

**Performance:**
- Typical batch time: 12 minutes for 2.4M customers
- Alternative (non-distributed): 16+ hours
- Cost: Pay only for 12 minutes vs 24/7 endpoint cost

**Scheduling Recommendation:**
- Frequency: Daily at 2 AM
- Timeout: 60 minutes
- Notifications: Email to retention team

---

### notebook-06-serving.py

**Purpose:** Deploy real-time REST API endpoint for on-demand predictions

**Endpoint Configuration:**
- Model: Champion model from Unity Catalog
- Workload size: Small (scales to zero for cost efficiency)
- Scale to zero: Enabled (cost optimization)
- Auto-capture: Enabled for inference table generation

**Key Operations:**
1. Retrieves Champion model metadata
2. Creates or updates serving endpoint using Databricks SDK
3. Configures auto-capture to log requests/responses
4. Waits for endpoint readiness (5-15 minutes)
5. Tests endpoint with sample prediction
6. Measures latency (P50, P95, P99)

**Endpoint Features:**
- REST API with JSON input/output
- Bearer token authentication
- Auto-scaling based on request volume
- Inference table for monitoring
- Environment variables for model version tracking

**Performance Metrics:**
- Single prediction latency: Measured across 20 requests
- Average latency target: < 100ms
- P95 latency target: < 200ms

**API Integration Example:**
```python
import requests

response = requests.post(
    ENDPOINT_URL,
    headers={"Authorization": f"Bearer {token}"},
    json={"dataframe_records": [customer_data]}
)

churn_probability = response.json()['predictions'][0]
```

**Outputs:**
- Endpoint name and URL
- Inference table: `{catalog}.{db}.churn_endpoint_*`
- MLflow run with deployment metadata
- Task values with endpoint info for downstream notebooks

**Use Cases:**
- Real-time churn assessment during customer calls
- Integration with CRM systems
- A/B testing with web applications
- API-driven decision making

---

### notebook-07-monitoring.py

**Purpose:** Comprehensive production model monitoring with drift detection and performance tracking

**Monitoring Strategy:**

**1. Inference Table Creation**
- Creates custom inference table with predictions, ground truth, and timestamps
- Generates 7 days of historical prediction data
- Includes both probability predictions and binary classifications
- Validates data quality (binary predictions for confusion matrix)

**2. Lakehouse Monitor Setup**
- Uses InferenceLog profile type for classification metrics
- Configures prediction_col and label_col for performance tracking
- Sets daily granularity for time-series analysis
- Schedules automatic refresh every 6 hours

**3. Performance Metrics Tracked**
- Accuracy, Precision, Recall, F1 Score
- ROC AUC score
- Log Loss
- Confusion Matrix (TP, FP, TN, FN)
- False Positive Rate, False Negative Rate
- Per-class performance (minority/majority)

**4. Drift Detection**
- Feature distribution changes over time
- Statistical tests: KS-test, Chi-squared test
- Wasserstein distance for distribution shift magnitude
- Consecutive drift (compare adjacent time windows)
- Baseline drift (compare to initial deployment)

**5. Drift Simulation**
- Creates synthetic drift data over 14 days
- MonthlyCharges: +30% progressive increase
- Tenure: -25% progressive decrease
- Contract shifts: More month-to-month contracts
- Validates drift detection capabilities

**Key Tables:**
- Inference table: `{catalog}.{db}.churn_inference_with_labels`
- Profile metrics: `{catalog}.{db}.churn_inference_with_labels_profile_metrics`
- Drift metrics: `{catalog}.{db}.churn_inference_with_labels_drift_metrics`

**Visualizations:**
- Performance trends over time (line charts)
- Feature drift analysis (distribution changes)
- Prediction drift (churn rate changes)
- Confusion matrix evolution
- Correlation analysis (features vs predictions)

**Alerting Criteria:**
- F1 score degradation > 5%
- Significant drift (p-value < 0.05)
- Prediction distribution shift > 10%
- False negative rate spike

**Dashboard:**
- Auto-generated Databricks SQL dashboard
- Accessible via Catalog Explorer Quality tab
- Custom SQL queries for metric exploration

**Outputs:**
- Monitor info with table references and dashboard ID
- Refresh status and metrics
- Drift detection reports
- Visual trend analysis

---

### notebook-08-retrain.py

**Purpose:** Automated model retraining triggered by drift or performance degradation

**Retraining Workflow:**

**Step 1: Assess Current Model Performance**
- Retrieves Champion model metadata
- Queries latest performance metrics from monitoring profile table
- Compares current metrics with baseline (initial deployment)
- Determines if retraining is needed (5% degradation threshold)

**Step 2: Prepare Fresh Training Data**
- Loads recent inference data with ground truth labels
- Combines 70% recent observations + 30% historical data
- Ensures data freshness while maintaining stability
- Creates new train/test splits (80/20)
- Validates churn rate and data quality

**Step 3: Load Fresh Data**
- Converts Spark DataFrames to Pandas for LightGBM
- Separates features from labels
- Validates feature count and churn rate consistency

**Step 4: Retrieve Champion Hyperparameters**
- Loads hyperparameters from current Champion model
- Uses proven parameters as starting point (no re-optimization)
- Faster retraining vs full hyperparameter search
- Maintains model architecture consistency

**Step 5: Retrain Model**
- Trains new LightGBM model with fresh data
- Uses early stopping (30 rounds)
- Logs comprehensive metadata to MLflow
- Records retrain reason, date, and data sources

**Step 6: Compare with Current Champion**
- Evaluates new model on test set
- Compares all metrics with Champion
- Calculates improvement/degradation
- Determines registration eligibility

**Registration Criteria:**
- Register if F1 improvement > 0 (any improvement)
- Register if F1 degradation < 2% (acceptable similarity)
- Block registration if F1 degradation >= 2%

**Step 7: Register as New Challenger**
- Registers model to Unity Catalog if criteria met
- Adds detailed retrain metadata and comparison
- Removes old Challenger alias
- Assigns new Challenger alias
- Adds data lineage tags

**Step 8: Model Registry Status**
- Lists recent model versions with aliases
- Shows F1 scores for comparison
- Displays lineage and promotion history

**Step 9: Next Steps**
- Provides actionable recommendations
- Links to validation and approval notebooks
- Saves metadata to task values for automation

**Retraining Triggers:**
1. Performance degradation (F1 drop > 5%)
2. Feature drift detection (PSI > 0.25)
3. Prediction drift (distribution shift)
4. Time-based policy (monthly fallback)
5. Manual trigger for model updates

**Automation Options:**
- Uncomment auto-trigger cell for hands-free workflow
- Schedule notebook to run weekly/monthly
- Trigger from drift alerts
- Integrate with Databricks Workflows

**Outputs:**
- New model version in Unity Catalog (if registered)
- Challenger alias assignment
- Task values with retrain metadata
- MLflow run with training lineage

**Monitoring After Retraining:**
- Continue tracking in monitoring pipeline
- Compare new Challenger performance in production
- Run A/B test between Champion and Challenger
- Promote to Champion after validation

---

## Data Flow

```
Raw Data (Bronze)
    |
    v
Feature Engineering (notebook-01)
    |
    v
Training Data (Train/Test Split)
    |
    v
Model Training (notebook-02)
    |
    v
Model Registration (notebook-03b)
    |
    v
Challenger Validation (notebook-04a)
    |
    v
Approval & Promotion (notebook-04b)
    |
    +----> Batch Inference (notebook-05)
    |
    +----> Real-time Serving (notebook-06)
    |
    v
Production Monitoring (notebook-07)
    |
    v
Drift Detection
    |
    v
Automated Retraining (notebook-08)
```

## Model Lifecycle

### Model Aliases

- **Challenger:** Newly registered model pending validation
- **Champion:** Production model actively serving predictions
- **Previous_Champion:** Last Champion before promotion (rollback support)

### Promotion Flow

1. Model trained and registered with Challenger alias
2. Automated validation tests (7 quality gates)
3. Comparison with current Champion
4. Approval workflow (auto or manual)
5. Promotion to Champion alias
6. Production deployment (batch/real-time)

## Key Features

### Production-Grade Governance

- Unity Catalog three-level namespace for model organization
- Model aliases for lifecycle management (Challenger/Champion)
- Comprehensive model metadata and lineage tracking
- Audit trail via model version tags and MLflow tracking

### Automated Quality Gates

- 7-stage validation suite before production promotion
- Realistic thresholds for imbalanced classification
- Performance, latency, robustness, and comparison tests
- Automated approval or manual review based on results

### Drift Detection

- Statistical tests for feature drift (KS-test, Chi-squared)
- Prediction distribution monitoring
- Performance degradation tracking
- Confusion matrix evolution over time

### Automated Retraining

- Triggered by drift or performance degradation
- Uses recent data with historical context
- Leverages proven hyperparameters for speed
- Automatic Challenger registration if improved

### Monitoring & Observability

- Lakehouse Monitoring for comprehensive metrics
- F1, Precision, Recall, ROC AUC tracking
- Confusion matrix with FPR/FNR analysis
- Feature drift and prediction drift visualization
- Automated dashboard generation

### Scalability

- Distributed batch inference using Spark UDFs
- Scales to millions of predictions (2.4M+ tested)
- Cost-efficient batch strategy vs 24/7 endpoints
- Real-time endpoint with auto-scaling and scale-to-zero

## Prerequisites

- Databricks workspace with Unity Catalog enabled
- ML Runtime 14.3.x or higher
- Required libraries: lightgbm, optuna, scikit-learn, databricks-sdk
- Appropriate permissions for catalog and schema creation
- API access for Databricks SDK operations

## Setup Instructions

1. **Create Catalog and Schema:**
   ```python
   # Run notebook-00-setup.py
   # This creates the catalog, schema, and loads sample data
   ```

2. **Run Feature Engineering:**
   ```python
   # Run notebook-01-features.py
   # Generates ML-ready features and train/test splits
   ```

3. **Train Initial Model:**
   ```python
   # Run notebook-02-training.py
   # Performs hyperparameter optimization and trains model
   ```

4. **Register Model:**
   ```python
   # Run notebook-03b-registration.py
   # Registers model to Unity Catalog with Challenger alias
   ```

5. **Validate and Promote:**
   ```python
   # Run notebook-04a-validation.py
   # Runs 7 automated validation tests

   # Run notebook-04b-approval.py
   # Promotes to Champion if tests pass
   ```

6. **Deploy for Inference:**
   ```python
   # Option A: Batch Inference
   # Run notebook-05-batch-inference.py
   # Generates predictions for all customers

   # Option B: Real-time Serving
   # Run notebook-06-serving.py
   # Deploys REST API endpoint
   ```

7. **Enable Monitoring:**
   ```python
   # Run notebook-07-monitoring.py
   # Sets up Lakehouse Monitoring with drift detection
   ```

8. **Configure Automated Retraining:**
   ```python
   # Run notebook-08-retrain.py manually or schedule
   # Monitors drift and retrains when needed
   ```

## Deployment Job

For automated execution, use notebook-03a-deployment.py to create a Databricks Job that orchestrates the pipeline:

```python
# Job tasks in order:
1. Feature Engineering
2. Model Training (depends on #1)
3. Model Registration (depends on #2)
4. Model Validation (depends on #3)

# Schedule: Weekly or triggered by data updates
```

## Configuration

Update `notebook-00-setup.py` with your environment-specific settings:

```python
catalog = "art_mlops"  # Your Unity Catalog name
db = "mlops_churn_demo"  # Your schema name
user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()

model_name = f"{catalog}.{db}.churn_model"
experiment_path = f"/Users/{user}/mlops_churn_experiments"
```

## Monitoring and Alerts

### Performance Degradation Alert

Create SQL alert in Databricks SQL:

```sql
SELECT
    window.end as check_time,
    f1_score.weighted as f1_score,
    accuracy_score
FROM {profile_table}
WHERE column_name = ':table'
  AND window.end >= current_date() - 1
  AND f1_score.weighted < 0.55
ORDER BY window.end DESC
LIMIT 1
```

**Trigger:** Query returns results (F1 < 0.55)
**Frequency:** Daily
**Action:** Notify ML team, consider retraining

### Drift Detection Alert

```sql
SELECT
    window.start,
    column_name,
    ks_test.pvalue as drift_pvalue,
    wasserstein_distance
FROM {drift_table}
WHERE drift_type = 'CONSECUTIVE'
  AND ks_test.pvalue < 0.05
  AND window.start >= current_date() - 1
ORDER BY wasserstein_distance DESC
```

**Trigger:** Query returns results (significant drift detected)
**Frequency:** Daily
**Action:** Review drift causes, trigger retraining

## Cost Optimization

- **Batch vs Real-time:** Use batch inference for periodic scoring (12 min/week vs 24/7 endpoint)
- **Scale to Zero:** Enable for real-time endpoints with sporadic traffic
- **Job Clusters:** Use ephemeral clusters for training jobs
- **Monitoring Schedule:** Adjust refresh frequency based on data velocity

## Best Practices

1. **Always run validation before promotion** - Use notebook-04a to catch issues
2. **Monitor production performance** - Set up alerts for degradation
3. **Retain Previous_Champion** - Enable quick rollback if needed
4. **Document retrain reasons** - Use model version tags and descriptions
5. **Test with production data distribution** - Use recent inference data for retraining
6. **Schedule regular retraining** - Monthly fallback even without drift
7. **Use realistic quality gates** - Calibrate thresholds for your business context
8. **Track business metrics** - ROI, customer retention, intervention costs

## Troubleshooting

### Validation Fails

- Review quality gate thresholds in notebook-04a
- Check test data distribution vs training data
- Validate feature engineering consistency
- Consider manual approval if business justifies deployment

### Drift Not Detected

- Verify monitoring refresh completed successfully
- Check inference table has multiple days of data
- Ensure ground truth labels are available
- Review drift threshold settings (p-value < 0.05)

### Retraining Produces Worse Model

- Check fresh data quality and churn rate
- Verify feature engineering didn't change
- Try full hyperparameter optimization vs using Champion params
- Investigate data leakage or temporal issues

### Endpoint Latency High

- Check workload size (Small/Medium/Large)
- Verify feature computation time
- Consider caching for frequently accessed features
- Profile prediction code for bottlenecks

## Future Enhancements

- A/B testing framework for Challenger vs Champion comparison
- Automated hyperparameter re-optimization during retraining
- Multi-model ensemble for improved predictions
- Feature importance tracking over time
- Bias and fairness monitoring
- Integration with external data sources
- Custom drift detection algorithms
- Automated rollback on production errors

## References

- [Databricks MLflow Documentation](https://docs.databricks.com/mlflow/)
- [Databricks Lakehouse Monitoring](https://docs.databricks.com/lakehouse-monitoring/)
- [Unity Catalog Model Registry](https://docs.databricks.com/machine-learning/manage-model-lifecycle/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [Optuna Hyperparameter Optimization](https://optuna.org/)

## License

This repository is for educational and demonstration purposes.

## Author

Pravin Varma

## Dataset

IBM Telco Customer Churn Dataset - Used for demonstration purposes only.
