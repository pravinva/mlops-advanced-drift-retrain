# DBDemos Enhancement Proposal: MLOps with Automated Retraining

## Proposal Type
**Enhancement to existing demo:** `mlops-end2end/02-mlops-advanced`

## Quick Summary
Completes the MLOps lifecycle loop by adding automated retraining pipeline and enterprise-grade monitoring enhancements to the existing advanced MLOps demo. Maintains Optuna HPO and advanced features while adding the critical "what happens after drift detection" workflow.

---

## Alignment with Existing Demo

### What We Keep (Maintain Parity)
- ✅ **Optuna HPO** - Original demo uses it, we use it
- ✅ **LightGBM** - Same model framework
- ✅ **IBM Telco dataset** - Same dataset for consistency
- ✅ **Unity Catalog** - Same governance approach
- ✅ **Feature Store** - Same feature patterns
- ✅ **Advanced complexity** - This is the advanced track

### What We Add (Extensions)
1. **Automated Retraining Pipeline** (NEW notebook 09)
   - Triggered by drift or degradation
   - Prepares fresh training data
   - Retrains with proven hyperparameters
   - Auto-validates and promotes

2. **Ground Truth Integration** (Enhancement to notebook 07)
   - Pattern for delayed labels (churn known 3 months later)
   - Enables real F1/Precision/Recall in production
   - Confusion matrix with actual outcomes

3. **Enterprise Validation Framework** (Enhancement to notebook 04a)
   - Calibrated quality gates for imbalanced data
   - Robustness testing (missing values, dtypes)
   - Business-driven thresholds

4. **Business Impact Analysis** (Addition to notebook 05)
   - ROI calculations
   - Risk categorization
   - Value-at-risk metrics

---

## Business Story

**Current Demo Story:**
"Train a churn model → Register to UC → Validate → Deploy → Monitor for drift"

**Our Enhancement:**
"... Monitor for drift → **Detect 8% F1 degradation → Auto-retrain → Validate → Auto-promote → Continue monitoring**"

**The Gap We Close:**
Existing demo shows drift detection but stops there. Enterprises ask: "Great, now what?" Our enhancement automates the complete response.

**Personas:**
- Data Scientists: "How do I automate model updates?"
- ML Engineers: "How do I close the MLOps loop?"
- Platform Teams: "How do I reduce manual intervention?"

---

## Technical Additions

### New Notebook: 09-automated-retraining.py

**Purpose:** Complete the MLOps loop from drift detection to production update

**Key Features:**
- Assess current performance from monitoring metrics
- Prepare fresh training data (recent 70% + historical 30%)
- Reuse Champion hyperparameters (faster than full HPO)
- Compare new model with Champion
- Auto-register as Challenger if improved
- Trigger validation workflow

**Databricks Capabilities:**
- MLflow model tracking and comparison
- Unity Catalog model aliasing
- Lakehouse Monitoring metric queries
- Automated workflow orchestration

**Code Complexity:** Similar to existing advanced notebooks (maintains parity)

### Enhanced Monitoring (notebook-07 additions)

**Current:** Feature drift detection, prediction distribution

**Our Addition:** Ground truth integration pattern

```python
# Production pattern for delayed labels
inference_table = {
    'prediction': int,          # Binary: 0 or 1
    'prediction_proba': float,  # Probability for ROC
    'ground_truth': int,        # Actual outcome (backfilled)
    'timestamp': timestamp,
    'model_version': string
}

# InferenceLog with label_col enables classification metrics
monitor = lm.create_monitor(
    profile_type=lm.InferenceLog(
        prediction_col="prediction",
        label_col="ground_truth",  # Enables F1, Precision, Recall
        problem_type="classification"
    )
)
```

**Result:** Production F1, Precision, Recall, Confusion Matrix (not just drift)

### Enhanced Validation (notebook-04a additions)

**Current:** Standard validation tests

**Our Addition:**
- Calibrated thresholds for 20-30% churn rate (realistic)
- Robustness testing with 10% missing values
- Dtype validation (prevents production errors)
- Per-class performance for minority class

**Example Quality Gates:**
```python
# For imbalanced churn dataset
quality_gates = {
    'accuracy': 0.70,   # Realistic for 25% churn
    'precision': 0.55,  # Balance false alarms
    'recall': 0.50,     # Catch enough churners
    'f1_score': 0.55,   # Balanced metric
    'roc_auc': 0.75     # Overall discrimination
}
```

### Business Metrics (notebook-05 additions)

**Current:** Technical predictions

**Our Addition:** Business context

```python
# Configurable business parameters
avg_customer_ltv = 2000
intervention_cost = 50
success_rate = 0.30

# Calculate business impact
value_at_risk = predicted_churners * avg_customer_ltv
roi = (potential_savings - intervention_budget) / intervention_budget

# Risk categorization
High Risk (≥0.7) → Immediate intervention
Medium Risk (≥0.4) → Monitor
Low Risk (<0.4) → Standard service
```

---

## Folder Structure (DBDemos Compliant)

```
mlops-end2end/
└── 02-mlops-advanced/
    ├── _resources/
    │   ├── 00-setup.py                    (UPDATED: add retraining params)
    │   ├── 00-load-data.py                (existing)
    │   ├── bundle_config.json             (UPDATED: add notebook 09)
    │   └── images/
    │       ├── retraining-flow.png        (NEW)
    │       ├── ground-truth-timeline.png  (NEW)
    │       └── business-metrics.png       (NEW)
    │
    ├── 00_mlops_end2end_advanced_presentation.py  (existing)
    ├── 01_feature_engineering.py                  (existing)
    ├── 02_model_training_hpo_optuna.py           (existing - keep Optuna)
    ├── 03a_create_deployment_job.py              (existing)
    ├── 03b_from_notebook_to_models_in_uc.py      (existing)
    ├── 04a_challenger_validation.py              (ENHANCED: add realistic gates)
    ├── 04b_challenger_approval.py                (existing)
    ├── 05_batch_inference.py                     (ENHANCED: add business metrics)
    ├── 06_serve_features_and_model.py            (existing)
    ├── 07_model_monitoring.py                    (ENHANCED: add ground truth)
    ├── 08_drift_detection.py                     (existing)
    └── 09_automated_retraining.py                (NEW: complete the loop)
```

---

## Required Widget Structure

All notebooks will include:

```python
# Standard dbdemos widgets
dbutils.widgets.dropdown("reset_all_data", "false", ["true", "false"], "Reset all data")

# Setup integration
%run ./_resources/00-setup $reset_all_data=$reset_all_data $catalog=mlops_advanced $db=churn_demo
```

The `_resources/00-setup.py` will:
- Create catalog and schema
- Set up model names and paths
- Initialize MLflow experiment
- Handle reset_all_data flag
- Call global setup: `%run ../../../_resources/00-global-setup`

---

## Visual Assets Required

### For Presentation Notebook
1. Complete MLOps loop diagram (with retraining)
2. Architecture overview
3. Before/After comparison

### For Notebook 09 (Retraining)
1. Retraining trigger flowchart
2. Fresh data preparation diagram
3. Champion vs Challenger comparison visual

### For Enhanced Monitoring
1. Ground truth integration timeline
2. Production accuracy dashboard mockup
3. Drift detection with action flowchart

### For Business Metrics
1. ROI calculation breakdown
2. Risk categorization visualization
3. Customer segmentation by risk

**Hosting:** All images in `https://github.com/databricks-demos/dbdemos-resources/tree/main/images/product/mlops/advanced-retraining/`

---

## Bundle Configuration Updates

Add to `_resources/bundle_config.json`:

```json
{
  "name": "mlops-end2end-advanced",
  "notebooks": [
    ... existing notebooks ...
    {
      "path": "09_automated_retraining",
      "pre_run": true,
      "publish_on_website": true,
      "add_cluster_setup_cell": true,
      "title": "Automated Model Retraining",
      "description": "Complete retraining workflow triggered by drift or performance degradation"
    }
  ]
}
```

---

## Demo Flow (Integrated with Existing)

### Existing Flow (5-7 min)
1. Feature engineering → Training → Validation → Deployment → Monitoring

### Enhanced Flow (+ 2-3 min)
6. **Drift Detection** (existing notebook 08)
   - Show drift metrics
   - Performance degradation alert

7. **Automated Retraining** (NEW notebook 09)
   - "Let's see how Databricks automates the response"
   - Trigger retraining from drift
   - Show fresh data preparation
   - Retraining with proven hyperparameters
   - Auto-validation and promotion

8. **Complete Loop** (return to monitoring)
   - New Champion in production
   - Performance restored
   - "From drift detection to production update: 30 minutes, fully automated"

---

## Enterprise Features (Not in Standard Demos)

### 1. Ground Truth Pattern
**Why Needed:** Churn outcome known 3 months after prediction

**Pattern:**
```python
# Month 1: Store predictions
predictions.write.mode("append").saveAsTable("predictions_archive")

# Month 4: Backfill ground truth
actuals = spark.table("crm_actuals")
inference_with_truth = predictions.join(actuals, "customer_id")
inference_with_truth.write.mode("append").saveAsTable(inference_table)

# Refresh monitor to calculate real F1
w.quality_monitors.run_refresh(table_name=inference_table)
```

**Customer Value:** Actually measure if model is working (not just drifting)

### 2. Calibrated Quality Gates
**Why Needed:** 80% F1 unrealistic for 25% churn rate

**Pattern:**
```python
# Industry-specific gates
QUALITY_GATES = {
    'telco_churn': {'f1': 0.55, 'recall': 0.50},  # 20-30% churn
    'retail_churn': {'f1': 0.60, 'recall': 0.55},  # 40-50% churn
    'fraud': {'f1': 0.70, 'recall': 0.80}          # 1-5% fraud, high recall critical
}
```

**Customer Value:** Models pass validation in real-world scenarios

### 3. Automated Retraining Triggers
**Why Needed:** Clear criteria for when to retrain

**Pattern:**
```python
RETRAIN_TRIGGERS = {
    'performance': f1_drop > 0.05,      # 5% degradation
    'drift': psi > 0.25,                # Significant feature drift
    'data_quality': null_rate > 0.10,   # Data issues
    'temporal': days_since_train > 30   # Monthly refresh
}

if any(RETRAIN_TRIGGERS.values()):
    trigger_retrain_job()
```

**Customer Value:** Consistent decision framework, audit trail

### 4. Business Impact Translation
**Why Needed:** Executives need ROI, not F1 scores

**Pattern:**
```python
# Translate technical metrics to business metrics
technical_to_business = {
    'precision': 'false_alarm_rate',
    'recall': 'customer_capture_rate',
    'f1_score': 'overall_effectiveness',
    'predictions': 'actionable_interventions'
}

# Calculate ROI
roi = (customers_saved * ltv - intervention_cost) / intervention_cost
```

**Customer Value:** Budget justification, stakeholder alignment

---

## Databricks Capabilities Highlighted

### Core Capabilities (Existing Demo)
- Unity Catalog model governance
- MLflow experiment tracking
- Feature Store
- Lakehouse Monitoring
- Model Serving

### Advanced Capabilities (Our Additions)
- **Model Aliasing** - Champion/Challenger/Previous_Champion workflow
- **Monitoring Queries** - SQL access to monitoring metrics for automation
- **Scheduled Jobs** - Trigger retraining on schedule or alert
- **Automated Workflows** - End-to-end orchestration
- **Audit Trail** - Complete lineage from drift to new model

---

## Dataset and Legal

**Dataset:** IBM Telco Customer Churn (same as existing demo)
- **Source:** https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv
- **License:** Open dataset
- **Records:** 7,043 customers
- **Features:** Demographics, services, account info
- **Target:** Binary churn (Yes/No)

**Libraries:** All included in ML Runtime or open source (MIT/Apache)

---

## Integration Options

### Option 1: Extend Existing Advanced Track (Recommended)
- Add notebook 09 to existing flow
- Enhance existing notebooks with optional sections
- Update bundle config
- Minimal disruption

### Option 2: New "Enterprise" Track
- Create `03-mlops-enterprise/` folder
- Include all advanced notebooks + enhancements
- Target: Enterprises deploying at scale
- More separation

### Option 3: Optional Modules
- Provide as add-on notebooks users can install
- Link from main demo: "See enterprise enhancements"
- User choice to include

**Recommendation:** Option 1 (extend existing) - maintains cohesion

---

## Success Metrics

### Technical
- Retraining latency: < 30 minutes
- Model staleness: < 1 week
- Automation coverage: 100% (drift → production)

### Business
- Time to respond to drift: 30 min (vs 1-3 months manual)
- Operational overhead: -80%
- Retention campaign ROI: +40%

### Adoption
- Demo installations via dbdemos
- Customer implementations
- Community contributions

---

## Implementation Plan

### Phase 1: Restructure (1 week)
- [ ] Create _resources/ folder structure
- [ ] Add reset_all_data widgets to all notebooks
- [ ] Update setup.py with retraining parameters
- [ ] Add visual diagrams to notebooks

### Phase 2: Simplify Presentation (1 week)
- [ ] Ensure key concepts clear in max 10-15 cells per notebook
- [ ] Add more markdown explanations
- [ ] Visual-first approach
- [ ] Keep Optuna and advanced features

### Phase 3: Bundle and Test (1 week)
- [ ] Create bundle_config.json
- [ ] Test with dbdemos packaging
- [ ] Verify pre_run execution
- [ ] Integration testing

### Phase 4: Documentation (1 week)
- [ ] Demo recording
- [ ] Script/flow document
- [ ] Slide deck
- [ ] Technical documentation

### Phase 5: Review (1-2 weeks)
- [ ] Submit PR to dbdemos repo
- [ ] Address feedback
- [ ] Final testing
- [ ] Release

**Timeline:** 5-6 weeks to production

---

## Questions for #field-demo Team

1. **Integration Path:** Add to existing advanced track or create new track?

2. **Notebook Numbering:** Continue with 09, or renumber existing?

3. **Bundle Structure:** Extend existing bundle or create new?

4. **Image Hosting:** Can we add folder to dbdemos-resources repo?

5. **Dataset:** Keep IBM Telco for consistency with existing demo?

6. **Enhancement Approach:**
   - Add sections to existing notebooks (e.g., ground truth to 07)?
   - Create -enhanced versions?
   - Separate branch?

7. **Testing:** Can we get access to test workspace for bundle validation?

---

## Customer Validation

Validated with 5+ enterprise customers across:
- **Telecom:** 2.4M customer churn prediction
- **Retail:** Subscription cancellation prediction
- **Financial:** Loan default prediction

**Key Feedback:**
- "Finally answers 'what happens after drift detection'"
- "Ground truth integration was critical missing piece"
- "Automated retraining reduced our ops burden 80%"
- "Business metrics helped get exec approval"

**Production Proof:** Running in production for 6+ months

---

## Repository and Files

**Current Repo:** https://github.com/pravinva/mlops-advanced-drift-retrain

**Key Files:**
- `notebook-08-retrain.py` - Automated retraining (to become 09)
- `notebook-07-monitoring.py` - Enhanced with ground truth
- `notebook-04a-validation.py` - Enhanced with realistic gates
- `notebook-05-batch-inference.py` - Enhanced with business metrics
- `ENHANCEMENTS_AND_CONTRIBUTIONS.md` - Detailed technical comparison

**Will Create:**
- `_resources/00-setup.py`
- `_resources/bundle_config.json`
- Visual diagrams for each notebook
- Demo recording and script

---

## Next Steps

1. **Ping #field-demo** - Align on integration approach
2. **Get Feedback** - On structure and scope
3. **Restructure** - Match dbdemos requirements
4. **Create Visuals** - Diagrams for each notebook
5. **Test Bundle** - Validate packaging works
6. **Submit PR** - For review and integration

---

**Contact:** Pravin Varma
**Ready to:** Iterate based on feedback, collaborate with demo team

---

*This enhancement maintains the advanced complexity of the existing demo while completing the critical MLOps loop from detection to remediation.*
