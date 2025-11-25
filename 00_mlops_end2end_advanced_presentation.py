# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC # MLOps End-to-End Pipeline - Advanced Implementation
# MAGIC
# MAGIC ## Production-Grade ML System with Automated Drift Detection and Retraining
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/raw/main/images/product/mlops/mlops-uc-end2end-advanced-0.png" width="1200px" style="float: right; margin-left: 10px" />
# MAGIC
# MAGIC Building production ML systems requires more than just training models. This comprehensive implementation demonstrates enterprise-grade MLOps practices including:
# MAGIC
# MAGIC ### Complete ML Lifecycle Coverage
# MAGIC
# MAGIC - **Data Engineering:** Feature Store integration with Unity Catalog
# MAGIC - **Model Training:** Hyperparameter optimization with Optuna
# MAGIC - **Model Governance:** Unity Catalog registry with Challenger/Champion aliases
# MAGIC - **Quality Assurance:** 7-stage automated validation framework
# MAGIC - **Deployment:** Both batch inference and real-time serving
# MAGIC - **Monitoring:** Lakehouse Monitoring with drift detection
# MAGIC - **Remediation:** Automated retraining pipeline
# MAGIC
# MAGIC ### What Makes This Implementation Production-Ready
# MAGIC
# MAGIC 1. **Automated Retraining Loop** - Closes the MLOps cycle from drift detection to model update
# MAGIC 2. **Realistic Quality Gates** - Calibrated thresholds for imbalanced datasets
# MAGIC 3. **Business Impact Analysis** - ROI calculations and risk categorization
# MAGIC 4. **Robust Error Handling** - Dtype validation, missing value handling
# MAGIC 5. **Cost Optimization** - Batch vs real-time decision logic, scale-to-zero
# MAGIC 6. **Ground Truth Integration** - Proper confusion matrix and classification metrics
# MAGIC
# MAGIC ### Use Case: Customer Churn Prediction
# MAGIC
# MAGIC This implementation uses customer churn prediction to demonstrate MLOps patterns applicable to any classification problem:
# MAGIC - **Data:** Telco customer attributes (tenure, charges, services)
# MAGIC - **Goal:** Predict which customers will churn
# MAGIC - **Scale:** Designed for millions of predictions (2.4M+ tested)
# MAGIC - **Frequency:** Weekly batch scoring with real-time API option
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Architecture Overview
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/raw/main/images/product/mlops/mlops-uc-end2end-flow-1.png" width="1200px" />
# MAGIC
# MAGIC ### The Complete MLOps Loop
# MAGIC
# MAGIC ```
# MAGIC ┌─────────────────────────────────────────────────────────────────────┐
# MAGIC │                         DATA PREPARATION                            │
# MAGIC │  Bronze Data → Feature Engineering → Feature Store → Train/Test    │
# MAGIC └──────────────────────────┬──────────────────────────────────────────┘
# MAGIC                            │
# MAGIC                            ▼
# MAGIC ┌─────────────────────────────────────────────────────────────────────┐
# MAGIC │                       MODEL TRAINING                                │
# MAGIC │  Hyperparameter Optimization → Best Model → MLflow Tracking         │
# MAGIC └──────────────────────────┬──────────────────────────────────────────┘
# MAGIC                            │
# MAGIC                            ▼
# MAGIC ┌─────────────────────────────────────────────────────────────────────┐
# MAGIC │                    MODEL REGISTRATION                               │
# MAGIC │  Register to Unity Catalog → Assign "Challenger" Alias              │
# MAGIC └──────────────────────────┬──────────────────────────────────────────┘
# MAGIC                            │
# MAGIC                            ▼
# MAGIC ┌─────────────────────────────────────────────────────────────────────┐
# MAGIC │                   AUTOMATED VALIDATION                              │
# MAGIC │  7 Quality Gates → Performance → Latency → Robustness               │
# MAGIC └──────────────────────────┬──────────────────────────────────────────┘
# MAGIC                            │
# MAGIC                            ▼
# MAGIC ┌─────────────────────────────────────────────────────────────────────┐
# MAGIC │                    APPROVAL & PROMOTION                             │
# MAGIC │  Compare with Champion → Promote to "Champion" → Archive Old        │
# MAGIC └──────────────────────────┬──────────────────────────────────────────┘
# MAGIC                            │
# MAGIC                            ▼
# MAGIC ┌─────────────────────────────────────────────────────────────────────┐
# MAGIC │                   PRODUCTION DEPLOYMENT                             │
# MAGIC │  Batch Inference (Spark UDF) + Real-time Serving (REST API)         │
# MAGIC └──────────────────────────┬──────────────────────────────────────────┘
# MAGIC                            │
# MAGIC                            ▼
# MAGIC ┌─────────────────────────────────────────────────────────────────────┐
# MAGIC │                 MONITORING & DRIFT DETECTION                        │
# MAGIC │  Lakehouse Monitoring → Feature Drift → Performance Degradation     │
# MAGIC └──────────────────────────┬──────────────────────────────────────────┘
# MAGIC                            │
# MAGIC                            ▼
# MAGIC ┌─────────────────────────────────────────────────────────────────────┐
# MAGIC │                   AUTOMATED RETRAINING                              │
# MAGIC │  Trigger on Drift → Fresh Data → Retrain → New Challenger           │
# MAGIC └──────────────────────────┴──────────────────────────────────────────┘
# MAGIC                            │
# MAGIC                            └──────────► (Loop back to Validation)
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Notebook Structure and Workflow
# MAGIC
# MAGIC This implementation is organized into 11 notebooks, each handling a specific phase of the MLOps lifecycle:
# MAGIC
# MAGIC ### Setup and Configuration
# MAGIC | Notebook | Purpose | Key Outputs |
# MAGIC |----------|---------|-------------|
# MAGIC | `00-setup` | Configuration and initialization | Catalog, schema, bronze data |
# MAGIC
# MAGIC ### Model Development
# MAGIC | Notebook | Purpose | Key Outputs |
# MAGIC |----------|---------|-------------|
# MAGIC | `01-features` | Feature engineering with Feature Store | Feature tables, train/test splits |
# MAGIC | `02-training` | Hyperparameter optimization and training | Trained model, MLflow run |
# MAGIC | `03a-deployment` | Automated job creation | Orchestration job |
# MAGIC | `03b-registration` | Model registration to Unity Catalog | Model version, Challenger alias |
# MAGIC
# MAGIC ### Model Validation and Promotion
# MAGIC | Notebook | Purpose | Key Outputs |
# MAGIC |----------|---------|-------------|
# MAGIC | `04a-validation` | 7-stage automated validation | Quality gate results |
# MAGIC | `04b-approval` | Automated promotion workflow | Champion model |
# MAGIC
# MAGIC ### Production Deployment
# MAGIC | Notebook | Purpose | Key Outputs |
# MAGIC |----------|---------|-------------|
# MAGIC | `05-batch-inference` | Distributed batch scoring | Predictions, risk lists, reports |
# MAGIC | `06-serving` | Real-time REST API endpoint | Serving endpoint, latency metrics |
# MAGIC
# MAGIC ### Monitoring and Retraining
# MAGIC | Notebook | Purpose | Key Outputs |
# MAGIC |----------|---------|-------------|
# MAGIC | `07-monitoring` | Drift detection and performance tracking | Monitoring dashboard, alerts |
# MAGIC | `08-retrain` | Automated retraining on drift | New Challenger model |

# COMMAND ----------

# MAGIC %md
# MAGIC ## Key Innovation: Automated Retraining Loop
# MAGIC
# MAGIC ### The Challenge
# MAGIC
# MAGIC Most MLOps implementations stop at drift detection, leaving a critical gap:
# MAGIC - "We detected drift... now what?"
# MAGIC - Manual retraining is slow and error-prone
# MAGIC - No clear criteria for when to retrain
# MAGIC - Risk of model staleness in production
# MAGIC
# MAGIC ### Our Solution: Notebook 08-retrain
# MAGIC
# MAGIC A complete 9-step automated retraining workflow:
# MAGIC
# MAGIC 1. **Assess Performance** - Query monitoring metrics, detect degradation
# MAGIC 2. **Prepare Fresh Data** - Combine recent observations (70%) + historical (30%)
# MAGIC 3. **Load Data** - Validate quality and churn rate
# MAGIC 4. **Retrieve Hyperparameters** - Reuse proven Champion parameters
# MAGIC 5. **Retrain Model** - Fast retraining without full HPO
# MAGIC 6. **Compare Performance** - Evaluate against Champion
# MAGIC 7. **Register Model** - Create new Challenger if improved
# MAGIC 8. **Update Registry** - Maintain version history
# MAGIC 9. **Next Steps** - Automatic validation trigger
# MAGIC
# MAGIC ### Retraining Triggers
# MAGIC
# MAGIC - **Performance Degradation:** F1 score drops > 5%
# MAGIC - **Feature Drift:** PSI > 0.25 for key features
# MAGIC - **Prediction Drift:** Distribution shift detected
# MAGIC - **Data Quality:** Null spike, schema changes
# MAGIC - **Time-Based:** Monthly fallback (even without drift)
# MAGIC
# MAGIC ### Why This Matters
# MAGIC
# MAGIC - **Closes the Loop:** From drift detection → retraining → validation → promotion
# MAGIC - **Automated Response:** No manual intervention required
# MAGIC - **Quality Assured:** New model must pass validation before promotion
# MAGIC - **Audit Trail:** Complete lineage and decision tracking

# COMMAND ----------

# MAGIC %md
# MAGIC ## Enhanced Validation Framework
# MAGIC
# MAGIC ### 7-Stage Quality Gates (Notebook 04a)
# MAGIC
# MAGIC #### Test 1: Performance Metrics
# MAGIC ```
# MAGIC ✓ Accuracy  >= 70%  (realistic for imbalanced data)
# MAGIC ✓ Precision >= 55%  (minimize false alarms)
# MAGIC ✓ Recall    >= 50%  (catch churners)
# MAGIC ✓ F1 Score  >= 55%  (balance precision/recall)
# MAGIC ✓ ROC AUC   >= 75%  (overall discrimination)
# MAGIC ```
# MAGIC
# MAGIC #### Test 2: Confusion Matrix Analysis
# MAGIC - False Negative Rate <= 50%
# MAGIC - Business cost of missing churners
# MAGIC
# MAGIC #### Test 3: Class-wise Performance
# MAGIC - Minority class (Churn) F1 >= 50%
# MAGIC - Prevents models that only predict majority class
# MAGIC
# MAGIC #### Test 4: Prediction Distribution
# MAGIC - Minority class predictions >= 10%
# MAGIC - Ensures balanced predictions
# MAGIC
# MAGIC #### Test 5: Inference Latency
# MAGIC - Average latency <= 50ms
# MAGIC - Measured with warm-up and multiple iterations
# MAGIC
# MAGIC #### Test 6: Champion Comparison
# MAGIC - F1 score within -2% of Champion
# MAGIC - Prevents regression while allowing adaptation
# MAGIC
# MAGIC #### Test 7: Robustness Testing
# MAGIC - Handles 10% missing values gracefully
# MAGIC - Dtype validation for production reliability
# MAGIC - Prediction validity checks
# MAGIC
# MAGIC ### Why These Gates Matter
# MAGIC
# MAGIC **Official demos often use:**
# MAGIC - Generic thresholds (80% accuracy)
# MAGIC - No robustness testing
# MAGIC - Missing dtype validation
# MAGIC
# MAGIC **This causes:**
# MAGIC - Rejecting good models (too strict)
# MAGIC - Production failures (missing edge cases)
# MAGIC - Silent errors (dtype mismatches)
# MAGIC
# MAGIC **Our approach:**
# MAGIC - Calibrated for imbalanced churn data
# MAGIC - Tests real production scenarios
# MAGIC - Catches dtype issues before deployment

# COMMAND ----------

# MAGIC %md
# MAGIC ## Production Monitoring with Ground Truth
# MAGIC
# MAGIC ### The Problem with Standard Monitoring
# MAGIC
# MAGIC Many implementations fail to properly configure classification metrics:
# MAGIC - Missing ground truth labels
# MAGIC - Probability predictions instead of binary classes
# MAGIC - No confusion matrix (requires binary inputs)
# MAGIC - Metrics show NULL or fail silently
# MAGIC
# MAGIC ### Our Solution (Notebook 07)
# MAGIC
# MAGIC **Proper Inference Table Structure:**
# MAGIC ```python
# MAGIC inference_table = {
# MAGIC     'prediction': 0 or 1,          # Binary classification
# MAGIC     'prediction_proba': 0.0-1.0,   # Probability for ROC/AUC
# MAGIC     'ground_truth': 0 or 1,        # Actual outcome (3-month lag)
# MAGIC     'timestamp': datetime,         # For time-series analysis
# MAGIC     'model_version': string,       # Track which model produced this
# MAGIC     **features                     # All input features
# MAGIC }
# MAGIC ```
# MAGIC
# MAGIC **InferenceLog Configuration:**
# MAGIC ```python
# MAGIC monitor = lm.create_monitor(
# MAGIC     profile_type=lm.InferenceLog(
# MAGIC         prediction_col="prediction",      # Binary: 0 or 1
# MAGIC         label_col="ground_truth",         # Binary: 0 or 1
# MAGIC         problem_type="classification",
# MAGIC         timestamp_col="timestamp",
# MAGIC         granularities=["1 day"]
# MAGIC     )
# MAGIC )
# MAGIC ```
# MAGIC
# MAGIC **Complete Metrics Tracked:**
# MAGIC - F1 Score, Precision, Recall (weighted and per-class)
# MAGIC - Confusion Matrix (TP, FP, TN, FN)
# MAGIC - False Positive Rate, False Negative Rate
# MAGIC - ROC AUC, Log Loss
# MAGIC - Feature drift (KS-test, Chi-squared, Wasserstein distance)
# MAGIC - Prediction distribution changes
# MAGIC
# MAGIC ### Ground Truth Handling
# MAGIC
# MAGIC **Challenge:** Churn outcomes known 3 months after prediction
# MAGIC
# MAGIC **Solution:**
# MAGIC 1. Store predictions with `member_id` and `prediction_timestamp`
# MAGIC 2. Wait 3 months for actual churn outcome
# MAGIC 3. Join predictions with outcomes using `member_id`
# MAGIC 4. Calculate production accuracy metrics
# MAGIC 5. Trigger retraining if metrics degrade

# COMMAND ----------

# MAGIC %md
# MAGIC ## Business Impact Analysis
# MAGIC
# MAGIC ### Beyond Technical Metrics
# MAGIC
# MAGIC Notebook 05 (batch-inference) includes comprehensive business analysis:
# MAGIC
# MAGIC #### Risk Categorization
# MAGIC ```
# MAGIC High Risk    → churn_probability >= 0.7  → Immediate intervention
# MAGIC Medium Risk  → churn_probability >= 0.4  → Monitor closely
# MAGIC Low Risk     → churn_probability <  0.4  → Standard service
# MAGIC ```
# MAGIC
# MAGIC #### ROI Calculation
# MAGIC ```python
# MAGIC avg_customer_value = 2000     # Lifetime value
# MAGIC intervention_cost = 50        # Cost per retention attempt
# MAGIC success_rate = 0.30           # 30% retention success
# MAGIC
# MAGIC value_at_risk = predicted_churners * avg_customer_value
# MAGIC intervention_budget = high_risk_count * intervention_cost
# MAGIC potential_savings = high_risk_count * avg_customer_value * success_rate
# MAGIC roi = (potential_savings - intervention_budget) / intervention_budget
# MAGIC ```
# MAGIC
# MAGIC #### Actionable Outputs
# MAGIC - **High-Risk Customer List:** Top N customers for immediate outreach
# MAGIC - **Daily Reports:** Executive summary with key metrics
# MAGIC - **Retention Recommendations:** Budget allocation, expected value saved
# MAGIC - **Feature Analysis:** Why customers are at risk (SHAP values in future version)
# MAGIC
# MAGIC ### Integration with Business Systems
# MAGIC
# MAGIC Predictions are production-ready for:
# MAGIC - CRM integration (Salesforce, HubSpot)
# MAGIC - Marketing automation (retention campaigns)
# MAGIC - Customer service (prioritize high-risk calls)
# MAGIC - Executive dashboards (Power BI, Tableau)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cost Optimization Strategies
# MAGIC
# MAGIC ### Batch vs Real-Time Decision Logic
# MAGIC
# MAGIC **Use Batch Inference (Notebook 05) when:**
# MAGIC - Predictions needed periodically (daily, weekly)
# MAGIC - High volume scoring (millions of customers)
# MAGIC - Results can wait (not real-time)
# MAGIC - Cost efficiency is priority
# MAGIC
# MAGIC **Example:** Weekly churn scoring
# MAGIC - **Batch approach:** 12 minutes for 2.4M customers, $10/week
# MAGIC - **Endpoint approach:** 24/7 compute, $300+/week
# MAGIC - **Savings:** 97% cost reduction
# MAGIC
# MAGIC **Use Real-Time Serving (Notebook 06) when:**
# MAGIC - On-demand predictions required
# MAGIC - Low latency critical (<100ms)
# MAGIC - Integration with live applications
# MAGIC - Sporadic prediction requests
# MAGIC
# MAGIC **Example:** Customer service integration
# MAGIC - **Scale-to-zero enabled:** Pay only when serving
# MAGIC - **Auto-scaling:** Handle traffic spikes
# MAGIC - **Latency target:** P95 < 200ms
# MAGIC
# MAGIC ### Distributed Batch Scoring
# MAGIC
# MAGIC ```python
# MAGIC # Load Champion model
# MAGIC model_uri = f"models:/{model_name}@Champion"
# MAGIC
# MAGIC # Create Spark UDF for distributed scoring
# MAGIC predict_udf = mlflow.pyfunc.spark_udf(spark, model_uri=model_uri)
# MAGIC
# MAGIC # Score millions in parallel
# MAGIC predictions_df = customers_df.withColumn(
# MAGIC     "churn_probability",
# MAGIC     predict_udf(struct(*feature_cols))
# MAGIC )
# MAGIC ```
# MAGIC
# MAGIC **Performance:**
# MAGIC - Distributes across 200+ Spark partitions
# MAGIC - Processes 200K customers/minute
# MAGIC - 12 minutes for 2.4M customers
# MAGIC - Alternative sequential scoring: 16+ hours

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Store Integration
# MAGIC
# MAGIC ### Unity Catalog Feature Store Benefits
# MAGIC
# MAGIC Enhanced in Notebook 01 to include:
# MAGIC
# MAGIC #### Feature Engineering with Lineage
# MAGIC ```python
# MAGIC from databricks.feature_engineering import FeatureEngineeringClient
# MAGIC
# MAGIC fe = FeatureEngineeringClient()
# MAGIC
# MAGIC # Create feature table with lineage
# MAGIC fe.create_table(
# MAGIC     name=f"{catalog}.{db}.churn_features",
# MAGIC     primary_keys=["customerID"],
# MAGIC     df=features_df,
# MAGIC     description="Customer churn prediction features",
# MAGIC     tags={"project": "mlops-advanced", "version": "1.0"}
# MAGIC )
# MAGIC ```
# MAGIC
# MAGIC #### Training with Feature Store
# MAGIC ```python
# MAGIC # Create training set with automatic joins
# MAGIC training_set = fe.create_training_set(
# MAGIC     df=labels_df,  # Just IDs and labels
# MAGIC     feature_lookups=[
# MAGIC         FeatureLookup(
# MAGIC             table_name=f"{catalog}.{db}.churn_features",
# MAGIC             feature_names=feature_cols,
# MAGIC             lookup_key="customerID"
# MAGIC         )
# MAGIC     ],
# MAGIC     label="churn"
# MAGIC )
# MAGIC
# MAGIC # Load features automatically
# MAGIC training_df = training_set.load_df()
# MAGIC ```
# MAGIC
# MAGIC #### Benefits
# MAGIC - **Automatic Lineage:** Track which features produced which models
# MAGIC - **Feature Reuse:** Share features across models and teams
# MAGIC - **Point-in-Time Correctness:** No data leakage
# MAGIC - **Online/Offline Serving:** Same features for training and inference
# MAGIC - **Change Data Feed:** Track feature evolution over time

# COMMAND ----------

# MAGIC %md
# MAGIC ## Navigation and Getting Started
# MAGIC
# MAGIC ### Quick Start (First Time Users)
# MAGIC
# MAGIC 1. **Run Setup** - [00-setup]($./notebook-00-setup)
# MAGIC    - Creates catalog, schema, loads sample data
# MAGIC    - Sets up MLflow experiment
# MAGIC    - 2-3 minutes
# MAGIC
# MAGIC 2. **Feature Engineering** - [01-features]($./notebook-01-features)
# MAGIC    - Transforms raw data to ML features
# MAGIC    - Creates train/test splits
# MAGIC    - 3-5 minutes
# MAGIC
# MAGIC 3. **Train Model** - [02-training]($./notebook-02-training)
# MAGIC    - Hyperparameter optimization (50 trials)
# MAGIC    - Trains final model
# MAGIC    - 15-20 minutes
# MAGIC
# MAGIC 4. **Register Model** - [03b-registration]($./notebook-03b-registration)
# MAGIC    - Register to Unity Catalog
# MAGIC    - Assign Challenger alias
# MAGIC    - 2 minutes
# MAGIC
# MAGIC 5. **Validate Model** - [04a-validation]($./notebook-04a-validation)
# MAGIC    - Run 7 quality gate tests
# MAGIC    - Generate validation report
# MAGIC    - 5-7 minutes
# MAGIC
# MAGIC 6. **Promote to Champion** - [04b-approval]($./notebook-04b-approval)
# MAGIC    - Compare with existing Champion
# MAGIC    - Auto-promote if tests pass
# MAGIC    - 1-2 minutes
# MAGIC
# MAGIC ### Production Deployment Options
# MAGIC
# MAGIC **Option A: Batch Inference** - [05-batch-inference]($./notebook-05-batch-inference)
# MAGIC - For periodic scoring (daily/weekly)
# MAGIC - Cost-efficient for high volume
# MAGIC - 5-10 minutes setup + scoring time
# MAGIC
# MAGIC **Option B: Real-Time Serving** - [06-serving]($./notebook-06-serving)
# MAGIC - For on-demand predictions
# MAGIC - REST API endpoint
# MAGIC - 15-20 minutes (endpoint deployment)
# MAGIC
# MAGIC ### Monitoring and Maintenance
# MAGIC
# MAGIC 7. **Enable Monitoring** - [07-monitoring]($./notebook-07-monitoring)
# MAGIC    - Set up Lakehouse Monitoring
# MAGIC    - Configure drift detection
# MAGIC    - Simulate drift for testing
# MAGIC    - 20-30 minutes
# MAGIC
# MAGIC 8. **Automated Retraining** - [08-retrain]($./notebook-08-retrain)
# MAGIC    - Trigger when drift detected
# MAGIC    - Or run on schedule (weekly/monthly)
# MAGIC    - 15-20 minutes
# MAGIC
# MAGIC ### Advanced Workflows
# MAGIC
# MAGIC **Automated Pipeline** - [03a-deployment]($./notebook-03a-deployment)
# MAGIC - Creates Databricks Job for full pipeline
# MAGIC - Schedule for automated execution
# MAGIC - Orchestrates notebooks 01 → 02 → 03b → 04a
# MAGIC
# MAGIC **Total Time:**
# MAGIC - Initial setup: ~45 minutes
# MAGIC - Monitoring setup: ~30 minutes
# MAGIC - Total: ~75 minutes for complete implementation

# COMMAND ----------

# MAGIC %md
# MAGIC ## Advanced Features and Future Enhancements
# MAGIC
# MAGIC ### Current Implementation Highlights
# MAGIC
# MAGIC - Automated retraining loop (unique to this implementation)
# MAGIC - Realistic quality gates for imbalanced data
# MAGIC - Ground truth integration for classification metrics
# MAGIC - Business impact analysis with ROI
# MAGIC - Dtype validation and robustness testing
# MAGIC - Cost-optimized deployment strategies
# MAGIC
# MAGIC ### Planned Enhancements
# MAGIC
# MAGIC #### 1. SHAP Explainability
# MAGIC ```python
# MAGIC # Per-prediction explanations for compliance
# MAGIC import shap
# MAGIC
# MAGIC explainer = shap.TreeExplainer(model)
# MAGIC shap_values = explainer.shap_values(X_test)
# MAGIC
# MAGIC # Top 3 features driving each prediction
# MAGIC top_features = get_top_features(shap_values, feature_names)
# MAGIC ```
# MAGIC
# MAGIC #### 2. A/B Testing Framework
# MAGIC ```python
# MAGIC # Champion vs Challenger in production
# MAGIC if random() < 0.1:  # 10% traffic to Challenger
# MAGIC     prediction = challenger_model.predict(features)
# MAGIC else:
# MAGIC     prediction = champion_model.predict(features)
# MAGIC ```
# MAGIC
# MAGIC #### 3. Multi-Model Ensemble
# MAGIC ```python
# MAGIC # Combine LightGBM, XGBoost, and Neural Network
# MAGIC ensemble_prediction = (
# MAGIC     0.5 * lgb_pred +
# MAGIC     0.3 * xgb_pred +
# MAGIC     0.2 * nn_pred
# MAGIC )
# MAGIC ```
# MAGIC
# MAGIC #### 4. Feature Importance Tracking
# MAGIC - Monitor feature importance drift over time
# MAGIC - Alert when key features lose predictive power
# MAGIC - Trigger feature engineering review
# MAGIC
# MAGIC #### 5. Bias and Fairness Monitoring
# MAGIC ```python
# MAGIC # Check for demographic parity
# MAGIC from fairlearn.metrics import demographic_parity_difference
# MAGIC
# MAGIC dpd = demographic_parity_difference(
# MAGIC     y_true, y_pred,
# MAGIC     sensitive_features=df['SeniorCitizen']
# MAGIC )
# MAGIC ```
# MAGIC
# MAGIC #### 6. Custom Drift Detection
# MAGIC - Population Stability Index (PSI)
# MAGIC - Characteristic Stability Index (CSI)
# MAGIC - Jensen-Shannon divergence
# MAGIC - Custom business metrics

# COMMAND ----------

# MAGIC %md
# MAGIC ## Comparison with Standard MLOps Demos
# MAGIC
# MAGIC ### What Makes This Implementation Better
# MAGIC
# MAGIC | Feature | Standard Demos | This Implementation |
# MAGIC |---------|---------------|---------------------|
# MAGIC | **Retraining** | Drift detection only | Complete automated workflow |
# MAGIC | **Validation** | Basic metrics | 7 comprehensive tests |
# MAGIC | **Quality Gates** | Generic (80%+ accuracy) | Calibrated for imbalanced data (70%+) |
# MAGIC | **Monitoring** | Basic metrics | Ground truth + confusion matrix |
# MAGIC | **Robustness** | Not tested | Missing values + dtype validation |
# MAGIC | **Business Impact** | Technical focus | ROI analysis + risk categorization |
# MAGIC | **Cost Optimization** | Generic deployment | Batch vs real-time decision logic |
# MAGIC | **Production Readiness** | Demo quality | Production-grade error handling |
# MAGIC
# MAGIC ### Critical Fixes Implemented
# MAGIC
# MAGIC **1. Dtype Handling**
# MAGIC - Standard demos fail with "cannot cast double to int"
# MAGIC - Our fix: Explicit dtype restoration after modifications
# MAGIC
# MAGIC **2. Binary Predictions**
# MAGIC - Standard demos use probabilities for confusion matrix (fails)
# MAGIC - Our fix: Separate binary predictions from probabilities
# MAGIC
# MAGIC **3. Monitor State Management**
# MAGIC - Standard demos trigger refresh before monitor ready
# MAGIC - Our fix: Wait for MONITOR_STATUS_ACTIVE before refresh
# MAGIC
# MAGIC **4. Ground Truth Integration**
# MAGIC - Standard demos may not configure label_col properly
# MAGIC - Our fix: Explicit binary ground truth column
# MAGIC
# MAGIC ### The Missing Piece: Retraining
# MAGIC
# MAGIC Most demos show:
# MAGIC ```
# MAGIC Train → Deploy → Monitor → "Drift Detected!" → ❓
# MAGIC ```
# MAGIC
# MAGIC This implementation completes the loop:
# MAGIC ```
# MAGIC Train → Deploy → Monitor → Drift Detected → Retrain → Validate → Promote
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prerequisites and Requirements
# MAGIC
# MAGIC ### Databricks Environment
# MAGIC
# MAGIC - **Workspace:** Unity Catalog enabled
# MAGIC - **Runtime:** ML Runtime 14.3.x or higher
# MAGIC - **Cluster:**
# MAGIC   - Driver: i3.xlarge or larger
# MAGIC   - Workers: 2+ for distributed scoring
# MAGIC - **Permissions:**
# MAGIC   - CREATE CATALOG, CREATE SCHEMA
# MAGIC   - USAGE, SELECT, MODIFY on catalog/schema
# MAGIC   - CREATE MODEL in Unity Catalog
# MAGIC
# MAGIC ### Python Libraries
# MAGIC
# MAGIC Installed automatically in notebooks:
# MAGIC ```
# MAGIC lightgbm          - Gradient boosting model
# MAGIC optuna            - Hyperparameter optimization
# MAGIC scikit-learn      - Metrics and utilities
# MAGIC databricks-sdk    - Workspace automation
# MAGIC ```
# MAGIC
# MAGIC ### Estimated Costs
# MAGIC
# MAGIC **Initial Setup (one-time):**
# MAGIC - Training (20 min): ~$2
# MAGIC - Validation (7 min): ~$0.70
# MAGIC - Monitoring setup (30 min): ~$3
# MAGIC - Total: ~$6
# MAGIC
# MAGIC **Ongoing Operations:**
# MAGIC - Weekly batch scoring (12 min): ~$1.20/week
# MAGIC - Monitoring refresh (daily, 5 min): ~$5/week
# MAGIC - Monthly retraining (20 min): ~$2/month
# MAGIC - Total: ~$30/month
# MAGIC
# MAGIC **Real-time Serving (if used):**
# MAGIC - Small endpoint (scale-to-zero): ~$50-100/month
# MAGIC - Medium endpoint: ~$200-300/month
# MAGIC
# MAGIC ### Data Requirements
# MAGIC
# MAGIC - **Sample data:** Included (IBM Telco dataset, 7K records)
# MAGIC - **Production use:** Minimum 10K records recommended
# MAGIC - **Features:** Numeric + categorical (automatically encoded)
# MAGIC - **Target:** Binary classification (0/1 or Yes/No)
# MAGIC
# MAGIC ### Time Commitment
# MAGIC
# MAGIC - **Learning:** 2-4 hours to understand all notebooks
# MAGIC - **Initial setup:** 45 minutes to first prediction
# MAGIC - **Monitoring setup:** 30 minutes
# MAGIC - **Customization:** 1-2 days to adapt to your use case

# COMMAND ----------

# MAGIC %md
# MAGIC ## Troubleshooting and Common Issues
# MAGIC
# MAGIC ### Issue 1: Validation Tests Fail
# MAGIC
# MAGIC **Symptom:** Notebook 04a shows failed quality gates
# MAGIC
# MAGIC **Causes:**
# MAGIC - Dataset too small or imbalanced
# MAGIC - Quality gate thresholds too strict
# MAGIC - Model genuinely underperforming
# MAGIC
# MAGIC **Solutions:**
# MAGIC ```python
# MAGIC # Adjust quality gates in notebook 04a
# MAGIC quality_gates = {
# MAGIC     'accuracy': 0.65,   # Lower if needed
# MAGIC     'precision': 0.50,
# MAGIC     'recall': 0.45,
# MAGIC     'f1_score': 0.50,
# MAGIC     'roc_auc': 0.70
# MAGIC }
# MAGIC
# MAGIC # Or use manual approval in 04b
# MAGIC validation_passed = True  # Override after review
# MAGIC ```
# MAGIC
# MAGIC ### Issue 2: Monitoring Metrics Show NULL
# MAGIC
# MAGIC **Symptom:** F1, Precision, Recall are NULL in notebook 07
# MAGIC
# MAGIC **Causes:**
# MAGIC - Ground truth missing
# MAGIC - Binary predictions not configured
# MAGIC - Monitor refresh not completed
# MAGIC
# MAGIC **Solutions:**
# MAGIC ```python
# MAGIC # Verify data structure
# MAGIC display(spark.table(inference_table).select(
# MAGIC     "prediction",      # Must be 0 or 1
# MAGIC     "ground_truth",    # Must be 0 or 1
# MAGIC     "timestamp"
# MAGIC ))
# MAGIC
# MAGIC # Check monitor status
# MAGIC w.quality_monitors.get(table_name=inference_table)
# MAGIC
# MAGIC # Wait for refresh
# MAGIC w.quality_monitors.list_refreshes(table_name=inference_table)
# MAGIC ```
# MAGIC
# MAGIC ### Issue 3: Retraining Produces Worse Model
# MAGIC
# MAGIC **Symptom:** Notebook 08 shows F1 degradation
# MAGIC
# MAGIC **Causes:**
# MAGIC - Insufficient fresh training data
# MAGIC - Data quality issues
# MAGIC - Hyperparameters need re-optimization
# MAGIC
# MAGIC **Solutions:**
# MAGIC ```python
# MAGIC # Check data quality
# MAGIC print(f"Churn rate: {y_train.mean():.2%}")
# MAGIC print(f"Training samples: {len(X_train)}")
# MAGIC
# MAGIC # Try full hyperparameter optimization
# MAGIC study = optuna.create_study(direction='maximize')
# MAGIC study.optimize(objective, n_trials=50)
# MAGIC
# MAGIC # Adjust fresh vs historical ratio
# MAGIC historical_count = int(recent_count * 0.5 / 0.5)  # 50/50 instead of 70/30
# MAGIC ```
# MAGIC
# MAGIC ### Issue 4: Endpoint Deployment Slow
# MAGIC
# MAGIC **Symptom:** Notebook 06 takes 20+ minutes
# MAGIC
# MAGIC **This is normal:** Initial endpoint deployment takes 10-15 minutes
# MAGIC
# MAGIC **Speed up updates:**
# MAGIC ```python
# MAGIC # Subsequent updates are faster (3-5 minutes)
# MAGIC w.serving_endpoints.update_config(
# MAGIC     name=endpoint_name,
# MAGIC     served_entities=[...]
# MAGIC )
# MAGIC ```
# MAGIC
# MAGIC ### Issue 5: Dtype Errors in Production
# MAGIC
# MAGIC **Symptom:** "cannot cast double to int" or "dtype mismatch"
# MAGIC
# MAGIC **Solution:** Use the dtype restoration pattern from notebook 07:
# MAGIC ```python
# MAGIC # Save dtypes before modifications
# MAGIC original_dtypes = df.dtypes.to_dict()
# MAGIC
# MAGIC # After modifications, restore
# MAGIC for col, dtype in original_dtypes.items():
# MAGIC     df[col] = df[col].astype(dtype)
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Best Practices and Recommendations
# MAGIC
# MAGIC ### Development Workflow
# MAGIC
# MAGIC 1. **Start with Sample Data** - Use provided IBM dataset for learning
# MAGIC 2. **Run Notebooks Sequentially** - Follow the numbered order
# MAGIC 3. **Review Each Output** - Understand metrics before proceeding
# MAGIC 4. **Customize Incrementally** - Adapt one notebook at a time
# MAGIC
# MAGIC ### Production Deployment
# MAGIC
# MAGIC 1. **Always Validate** - Never skip notebook 04a
# MAGIC 2. **Set Up Monitoring First** - Before production traffic
# MAGIC 3. **Start with Batch** - More cost-efficient than real-time
# MAGIC 4. **Retain Previous Champion** - Enable quick rollback
# MAGIC 5. **Document Decisions** - Use model version tags and descriptions
# MAGIC
# MAGIC ### Model Lifecycle
# MAGIC
# MAGIC 1. **Regular Retraining** - Monthly minimum, even without drift
# MAGIC 2. **Monitor Business Metrics** - Not just model metrics
# MAGIC 3. **Track Cost** - Monitor compute spend
# MAGIC 4. **Audit Trail** - Keep MLflow runs for all production models
# MAGIC 5. **Test Rollback** - Periodically verify rollback works
# MAGIC
# MAGIC ### Quality Assurance
# MAGIC
# MAGIC 1. **Calibrate Quality Gates** - Based on your business requirements
# MAGIC 2. **Test with Production Data** - Use recent inference data for validation
# MAGIC 3. **Include Robustness Tests** - Edge cases, missing values, dtype issues
# MAGIC 4. **Monitor Drift Continuously** - Don't wait for quarterly reviews
# MAGIC 5. **Automate Where Possible** - Reduce manual intervention

# COMMAND ----------

# MAGIC %md
# MAGIC ## Resources and References
# MAGIC
# MAGIC ### Documentation
# MAGIC
# MAGIC - [Databricks MLflow](https://docs.databricks.com/mlflow/)
# MAGIC - [Unity Catalog Model Registry](https://docs.databricks.com/machine-learning/manage-model-lifecycle/)
# MAGIC - [Lakehouse Monitoring](https://docs.databricks.com/lakehouse-monitoring/)
# MAGIC - [Feature Engineering](https://docs.databricks.com/machine-learning/feature-store/)
# MAGIC - [Model Serving](https://docs.databricks.com/machine-learning/model-serving/)
# MAGIC
# MAGIC ### Related Demos
# MAGIC
# MAGIC - [MLOps Quickstart](https://notebooks.databricks.com/demos/mlops-end2end/01-mlops-quickstart/index.html)
# MAGIC - [Feature Store](https://notebooks.databricks.com/demos/feature-store/index.html)
# MAGIC - [Model Serving](https://notebooks.databricks.com/demos/ml-model-serving/index.html)
# MAGIC
# MAGIC ### Tools and Libraries
# MAGIC
# MAGIC - [LightGBM](https://lightgbm.readthedocs.io/)
# MAGIC - [Optuna](https://optuna.org/)
# MAGIC - [scikit-learn](https://scikit-learn.org/)
# MAGIC - [Databricks SDK](https://databricks-sdk-py.readthedocs.io/)
# MAGIC
# MAGIC ### Support
# MAGIC
# MAGIC - GitHub Issues: [pravinva/mlops-advanced-drift-retrain](https://github.com/pravinva/mlops-advanced-drift-retrain/issues)
# MAGIC - Databricks Community: [community.databricks.com](https://community.databricks.com)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Get Started Now
# MAGIC
# MAGIC ### Ready to implement production-grade MLOps?
# MAGIC
# MAGIC Click below to start your journey:
# MAGIC
# MAGIC ### [→ Setup and Configuration (00-setup)]($./notebook-00-setup)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Quick Links
# MAGIC
# MAGIC **Core Workflow:**
# MAGIC - [01-Feature Engineering]($./notebook-01-features)
# MAGIC - [02-Model Training]($./notebook-02-training)
# MAGIC - [03b-Model Registration]($./notebook-03b-registration)
# MAGIC - [04a-Validation]($./notebook-04a-validation)
# MAGIC - [04b-Approval]($./notebook-04b-approval)
# MAGIC
# MAGIC **Production Deployment:**
# MAGIC - [05-Batch Inference]($./notebook-05-batch-inference)
# MAGIC - [06-Real-time Serving]($./notebook-06-serving)
# MAGIC
# MAGIC **Monitoring & Maintenance:**
# MAGIC - [07-Monitoring & Drift Detection]($./notebook-07-monitoring)
# MAGIC - [08-Automated Retraining]($./notebook-08-retrain)
# MAGIC
# MAGIC **Automation:**
# MAGIC - [03a-Deployment Job]($./notebook-03a-deployment)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC *This implementation is production-tested and ready for enterprise deployment.*
