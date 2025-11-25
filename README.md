# MLOps Advanced: Enhanced with Production-Ready Features

##  Overview

This is an **enhanced version** of the Databricks MLOps Advanced demo, adding production-critical features for real-world machine learning operations at scale.

### Enhancement Proposal

This repository proposes improvements to the [official Databricks MLOps Advanced demo](https://www.databricks.com/resources/demos/videos/data-science-and-ai) by adding:

1. **Ground Truth Integration** - Track real production accuracy with delayed ground truth
2. **Automated Retraining Triggers** - 6 intelligent triggers for model retraining
3. **Production Accuracy Monitoring** - Real F1/Precision/Recall vs training metrics
4. **Comprehensive Drift Detection** - PSI, KS-test, distribution shifts, data quality
5. **Closed-Loop MLOps** - Fully automated: Detect → Trigger → Train → Validate → Deploy
6. **Enhanced Validation Framework** - 7-stage validation with realistic quality gates
7. **Orchestrated Deployment** - End-to-end Databricks Job with task dependencies

##  Architecture

```
Data → Features → Training → Registry → Validation → Approval → Inference
   ↑                                                                 ↓
   └─────────────── Automated Retraining Loop ←─── Monitoring ←────┘
```

**Key Innovation:** Closed-loop system that automatically detects issues and retrains models without human intervention.

##  Notebooks

### Core MLOps Pipeline

| # | Notebook | Description | Status |
|---|----------|-------------|--------|
| 00 | `00_mlops_end2end_advanced_presentation` | Overview and architecture |  Enhanced |
| 01 | `01_feature_engineering` | Feature Store with Unity Catalog |  Enhanced |
| 02 | `02_model_training_hpo` | Optuna HPO (50 trials) |  Enhanced |
| 03 | `03_model_registration` | Unity Catalog registry |  Enhanced |
| 04a | `04a_challenger_validation` | 7-stage validation framework |  **NEW** |
| 04b | `04b_challenger_approval` | Automated promotion logic |  **NEW** |
| 05 | `05_batch_inference` | Distributed Spark scoring |  Enhanced |

### Production Operations (New/Enhanced)

| # | Notebook | Description | Status |
|---|----------|-------------|--------|
| 06 | `06_model_serving` | Real-time REST API (optional) |  Enhanced |
| 07 | `07_model_monitoring` | Lakehouse Monitoring + Ground Truth |  **NEW** |
| 08 | `08_automated_retraining` | 6 intelligent retraining triggers |  **NEW** |
| 09 | `09_deployment_job` | Orchestrated Databricks Job |  **NEW** |

##  Quick Start

### Prerequisites

- Databricks workspace with Unity Catalog enabled
- ML Runtime 14.3+ cluster
- Permissions: Create catalogs, schemas, models, jobs

### Installation

```bash
# Clone repository
git clone https://github.com/pravinva/mlops-advanced-drift-retrain.git

# Import into Databricks workspace
# Or upload notebooks via UI
```

### Run Demo

1. **Execute notebooks in sequence (00 → 09)**
   - Each notebook builds on the previous one
   - Start with `00_mlops_end2end_advanced_presentation`

2. **Or use orchestrated job (Notebook 09)**
   - Creates automated Databricks Job
   - Runs all tasks with proper dependencies

### Dataset

- **Source:** IBM Telco Customer Churn dataset (~7,000 customers)
- **Task:** Binary classification (churn prediction)
- **Challenge:** Imbalanced classes (~27% churn rate)

##  Key Enhancements Over Original Demo

### 1. Ground Truth Integration (Notebook 07)

**Original Demo:** Drift detection only, no production accuracy tracking

**Enhanced Demo:**
- Simulates 3-month delayed ground truth (realistic for subscription businesses)
- Joins predictions with actual outcomes
- Calculates **real production F1, Precision, Recall**
- Compares against training metrics
- Alerts if production < training by 5%+

**Business Value:** Know your *actual* model performance, not just drift indicators.

### 2. Automated Retraining Triggers (Notebook 08)

**Original Demo:** Manual retraining only

**Enhanced Demo:** 6 intelligent triggers (OR logic)

1. **Feature Drift:** PSI > 0.25 for any feature
2. **Prediction Drift:** Distribution shift detected (KS-test)
3. **Null Spike:** > 5% increase in null values
4. **F1 Drop:** Production F1 < training F1 by 5%+
5. **Data Quality:** Schema changes, range violations
6. **Monthly Fallback:** Retrain every 30 days (even if no alerts)

**Automated Workflow:**
```
Trigger Detection → Start Job → Train → Validate → Deploy (if passed)
```

**Business Value:** Zero-touch model maintenance, faster response to data changes.

### 3. Enhanced Validation Framework (Notebooks 04a, 04b)

**Original Demo:** 4 basic tests

**Enhanced Demo:** 7 comprehensive tests

| Test | Validates | Threshold |
|------|-----------|-----------|
| 1. Performance Metrics | F1, Accuracy, Precision, Recall | F1≥0.55, Acc≥0.70, Prec≥0.55 |
| 2. Confusion Matrix | False negative rate | FNR ≤ 50% |
| 3. Class Performance | Minority class F1 | ≥ 0.50 |
| 4. Prediction Distribution | No all-0 or all-1 predictions | Min class % ≥ 10% |
| 5. Inference Latency | Single prediction time | < 50ms |
| 6. Champion Comparison | F1 delta vs current Champion | ≥ -2% |
| 7. Robustness | Handles missing data | 10% nulls OK |

**Realistic Thresholds:** Calibrated for imbalanced churn datasets (not toy examples).

**Business Value:** Catch issues before production, reduce model failures.

### 4. Orchestrated Deployment Job (Notebook 09)

**Original Demo:** Manual notebook execution

**Enhanced Demo:**
- Databricks Job with 6 tasks
- Task dependencies (proper execution order)
- Cluster configuration with ML runtime
- Timeout and retry logic
- Email notifications
- Scheduling options (manual, cron, API-triggered)

**Task Flow:**
```
Feature Engineering → Model Training → Model Registration
       ↓
Challenger Validation → Challenger Approval → Batch Inference
```

**Business Value:** Production-ready CI/CD for ML, not just notebook experiments.

### 5. Production Accuracy Monitoring (Notebook 07)

**Original Demo:** Feature/prediction drift only

**Enhanced Demo:**
- **Real F1/Precision/Recall** calculated from ground truth
- Tracks accuracy degradation over time
- Compares production vs training metrics
- Alerts on 5%+ performance drop
- Confusion matrix evolution

**Monitoring Dashboard:**
- Lakehouse Monitoring for drift
- Custom metrics for production accuracy
- Unified view for all stakeholders

**Business Value:** Know when models are *actually* failing, not just drifting.

##  Comparison: Original vs Enhanced

| Feature | Original Demo | Enhanced Demo |
|---------|---------------|---------------|
| **Notebooks** | 6 core notebooks | 10 notebooks (4 new) |
| **Validation Tests** | 4 basic tests | 7 comprehensive tests |
| **Monitoring** | Drift detection | Drift + Ground truth + Real accuracy |
| **Retraining** | Manual only | 6 automated triggers |
| **Deployment** | Manual execution | Orchestrated Databricks Job |
| **Ground Truth** | Not included |  3-month lag simulation |
| **Production Metrics** | Drift indicators |  Real F1/Precision/Recall |
| **Closed-Loop** | Not implemented |  Fully automated |
| **Quality Gates** | Basic thresholds |  Realistic for imbalanced data |

##  Complete MLOps Workflow

### Phase 1: Initial Deployment (Notebooks 01-05)

1. **Feature Engineering** → Create feature table with CDF enabled
2. **Training + HPO** → Optuna 50 trials, LightGBM
3. **Registration** → Unity Catalog with @Challenger alias
4. **Validation** → 7-stage automated testing
5. **Approval** → Promote to @Champion if passed
6. **Inference** → Batch scoring with Spark UDFs

### Phase 2: Production Monitoring (Notebooks 06-07)

7. **Model Serving** → Optional real-time REST API
8. **Monitoring** → Lakehouse Monitoring + Ground Truth
   - Feature drift (PSI, KS-test)
   - Prediction drift
   - Data quality (nulls, schema)
   - **Production accuracy** (F1, Precision, Recall)

### Phase 3: Automated Maintenance (Notebooks 08-09)

9. **Automated Retraining** → Triggered by 6 conditions
   - Detects issues → Starts job
   - Trains new model → Validates
   - Registers as @Challenger → Waits for approval

10. **Deployment Job** → Orchestrates entire pipeline
    - Scheduled or triggered
    - Runs all tasks with dependencies
    - Emails notifications

##  Visual Diagrams

Professional Databricks-branded diagrams for each stage:

```
diagrams/
├── mlops-advanced-0-overview.png          # Complete lifecycle
├── mlops-advanced-1-feature-store.png     # Feature engineering
├── mlops-advanced-2-training.png          # Training + HPO
├── mlops-advanced-3-registration.png      # Unity Catalog
├── mlops-advanced-4-validation.png        # 7-stage validation
├── mlops-advanced-4b-approval.png         # Promotion workflow
├── mlops-advanced-5-inference.png         # Batch scoring
├── mlops-advanced-6-serving.png           # Real-time API
├── mlops-advanced-7-monitoring.png        # Monitoring + Ground Truth
├── mlops-advanced-8-retraining.png        # Retraining triggers
└── mlops-advanced-9-deployment.png        # Job orchestration
```

**Regenerate diagrams:**
```bash
python3 diagram_generator.py              # Generate HTML
python3 html_to_png_converter.py          # Convert to PNG
```

See `DIAGRAMS_GUIDE.md` for details.

##  Configuration

Default settings in `_resources/00-setup.py`:

```python
catalog = "mlops_advanced"
db = "churn_demo"
model_name = f"{catalog}.{db}.churn_model"
```

**Customizable:**
- Catalog/schema names
- Feature table names
- Model registry paths
- Monitoring thresholds
- Retraining triggers

##  Monitoring & Alerting

### Key Metrics Tracked

| Category | Metrics | Threshold |
|----------|---------|-----------|
| **Model Performance** | F1, Precision, Recall, Accuracy | Alert if F1 < 0.55 |
| **Feature Drift** | PSI, KS-test | Alert if PSI > 0.25 |
| **Data Quality** | Null %, schema, ranges | Alert if >5% change |
| **Prediction Drift** | Distribution shift | Alert if detected |
| **Latency** | P50, P95, P99 | Alert if P95 > 200ms |

### Dashboards

- **Data Scientists:** Drift metrics, model performance, experiments
- **ML Engineers:** Pipeline health, job status, infrastructure
- **Business Users:** Predictions, ROI, model explanations

##  Testing

Create `TESTING.md` with:
- End-to-end test procedures
- Validation checklists
- Integration tests
- Performance benchmarks

**Test Coverage:**
- Unit tests for feature engineering
- Integration tests for pipeline
- Performance tests for batch inference
- Drift simulation tests
- Validation framework tests

##  Project Structure

```
mlops-advanced/
├── 00_mlops_end2end_advanced_presentation.py  # Overview
├── 01_feature_engineering.py                  # Feature Store
├── 02_model_training_hpo.py                   # Training + HPO
├── 03_model_registration.py                   # Unity Catalog
├── 04a_challenger_validation.py               # 7-stage validation (NEW)
├── 04b_challenger_approval.py                 # Auto-promotion (NEW)
├── 05_batch_inference.py                      # Batch scoring
├── 06_model_serving.py                        # Real-time API
├── 07_model_monitoring.py                     # Monitoring + Ground Truth (NEW)
├── 08_automated_retraining.py                 # Retraining triggers (NEW)
├── 09_deployment_job.py                       # Job orchestration (NEW)
├── _resources/
│   ├── 00-setup.py                           # Setup utilities
│   ├── 00-load-data.py                       # Data loader
│   └── bundle_config.json                    # DAB configuration
├── diagrams/                                  # Visual diagrams (NEW)
│   ├── *.html                                # HTML diagrams
│   ├── *.png                                 # PNG exports
│   └── README.md                             # Diagram guide
├── docs/                                      # Documentation (NEW)
│   ├── TESTING.md                            # Test procedures
│   ├── DBDEMOS_SUBMISSION.md                 # Submission guide
│   ├── DIAGRAMS_GUIDE.md                     # Diagram documentation
│   ├── INSTALL.md                            # Installation guide
│   ├── ENHANCEMENTS_AND_CONTRIBUTIONS.md     # Enhancement details
│   └── DBDEMOS_PROPOSAL.md                   # Proposal document
├── diagram_generator.py                       # Diagram generator (NEW)
├── html_to_png_converter.py                  # PNG converter (NEW)
└── README.md                                 # This file
```

##  Learning Objectives

After completing this demo, you will understand:

1. **Feature Store:** Unity Catalog for centralized features
2. **Experiment Tracking:** MLflow for reproducible experiments
3. **Hyperparameter Tuning:** Automated with Optuna (50 trials)
4. **Model Registry:** Governance with Champion/Challenger
5. **Model Validation:** 7-stage comprehensive testing
6. **Batch Inference:** Distributed Spark scoring at scale
7. **Drift Detection:** Statistical tests (PSI, KS-test)
8. **Ground Truth Tracking:** Production accuracy validation
9. **Automated Retraining:** 6 intelligent triggers
10. **MLOps Orchestration:** End-to-end Databricks Jobs

##  Contributing

This is an **improvement proposal** for the official Databricks MLOps Advanced demo.

**Contributions welcome:**
- Bug reports
- Feature requests
- Documentation improvements
- Additional use cases
- Test coverage

##  License

This demo is provided as-is for educational and demonstration purposes.

##  Additional Resources

- [Databricks MLOps Guide](https://docs.databricks.com/mlflow/)
- [Unity Catalog Documentation](https://docs.databricks.com/data-governance/unity-catalog/)
- [Lakehouse Monitoring](https://docs.databricks.com/lakehouse-monitoring/)
- [Model Registry Best Practices](https://docs.databricks.com/machine-learning/manage-model-lifecycle/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [Optuna Documentation](https://optuna.org/)

##  Support

- **Issues:** [GitHub Issues](https://github.com/pravinva/mlops-advanced-drift-retrain/issues)
- **Documentation:** See `docs/` directory for detailed guides
  - Testing: `docs/TESTING.md`
  - Submission: `docs/DBDEMOS_SUBMISSION.md`
  - Diagrams: `docs/DIAGRAMS_GUIDE.md`

---

**Version:** 2.0 (Enhanced with Production Features)
**Repository:** https://github.com/pravinva/mlops-advanced-drift-retrain
**Generated with:** Claude Code (https://claude.com/claude-code)
