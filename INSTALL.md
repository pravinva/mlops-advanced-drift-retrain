# MLOps Advanced - Installation Guide

## Installation via dbdemos (Coming Soon)

This implementation will be available via dbdemos package:

```python
%pip install dbdemos
import dbdemos
dbdemos.install('mlops-advanced-retraining')
```

## Manual Installation (Current)

### Prerequisites

1. **Databricks Workspace**
   - Unity Catalog enabled
   - ML Runtime 14.3.x or higher
   - Minimum 2 workers recommended

2. **Permissions Required**
   - CREATE CATALOG, CREATE SCHEMA
   - USAGE, SELECT, MODIFY on catalog/schema
   - CREATE MODEL in Unity Catalog

3. **Estimated Time**
   - Setup: 5 minutes
   - Initial run: 45 minutes
   - Complete implementation: 75 minutes

### Step-by-Step Installation

#### 1. Clone or Import Notebooks

**Option A: Git Clone**
```bash
git clone https://github.com/pravinva/mlops-advanced-drift-retrain.git
cd mlops-advanced-drift-retrain
```

**Option B: Databricks Repos**
1. Navigate to Repos in Databricks workspace
2. Click "Add Repo"
3. Enter URL: `https://github.com/pravinva/mlops-advanced-drift-retrain.git`
4. Click "Create Repo"

#### 2. Configure Catalog and Schema

Edit `notebook-00-setup.py`:

```python
catalog = "your_catalog_name"  # Change to your catalog
db = "your_schema_name"        # Change to your schema
```

#### 3. Create ML Cluster

Create a cluster with these specifications:
- **Runtime:** 14.3.x-cpu-ml-scala2.12 or higher
- **Workers:** 2 minimum (4 recommended for faster training)
- **Node Type:** i3.xlarge or equivalent
- **Data Security Mode:** USER_ISOLATION

#### 4. Run Initial Setup

1. Open `notebook-00-setup.py`
2. Attach to your ML cluster
3. Run all cells
4. Verify output shows:
   - Catalog created
   - Schema created
   - Bronze table with ~7K records

**Expected Duration:** 2-3 minutes

#### 5. Run Feature Engineering

1. Open `notebook-01-features-enhanced.py` (or `notebook-01-features.py`)
2. Run all cells
3. Verify output shows:
   - Feature table created
   - Train/test splits created
   - Change Data Feed enabled

**Expected Duration:** 3-5 minutes

#### 6. Train Initial Model

1. Open `notebook-02-training.py`
2. Run all cells (includes 50 Optuna trials)
3. Verify output shows:
   - Best hyperparameters found
   - Final model metrics (F1, accuracy, etc.)
   - MLflow run ID

**Expected Duration:** 15-20 minutes

#### 7. Register Model

1. Open `notebook-03b-registration.py`
2. Run all cells
3. Verify output shows:
   - Model registered to Unity Catalog
   - Challenger alias assigned
   - Model loads successfully

**Expected Duration:** 2 minutes

#### 8. Validate Model

1. Open `notebook-04a-validation.py`
2. Run all cells
3. Review validation results (7 tests)
4. Note: Some tests may fail on small dataset (expected)

**Expected Duration:** 5-7 minutes

#### 9. Promote to Champion

1. Open `notebook-04b-approval.py`
2. Set `validation_passed = True` if manual approval needed
3. Run all cells
4. Verify Champion alias assigned

**Expected Duration:** 1-2 minutes

### Production Deployment Options

#### Option A: Batch Inference (Recommended)

1. Open `notebook-05-batch-inference.py`
2. Run all cells
3. Review:
   - Predictions table
   - High-risk customer list
   - Business impact analysis

**Expected Duration:** 5-10 minutes

#### Option B: Real-Time Serving

1. Open `notebook-06-serving.py`
2. Run all cells
3. Wait for endpoint deployment (10-15 minutes)
4. Test with sample prediction

**Expected Duration:** 15-20 minutes

### Monitoring and Maintenance

#### Enable Monitoring

1. Open `notebook-07-monitoring.py`
2. Run all cells
3. Wait for monitor refresh (10-15 minutes)
4. View metrics and drift detection

**Expected Duration:** 30 minutes total

#### Configure Automated Retraining

1. Open `notebook-08-retrain.py`
2. Review retraining triggers
3. Run manually or schedule

**Expected Duration:** 15-20 minutes per run

### Automated Orchestration

#### Create Deployment Job

1. Open `notebook-03a-deployment.py`
2. Run all cells
3. Note the Job ID and URL
4. Schedule as needed

**Job Components:**
- Feature Engineering
- Model Training
- Model Registration
- Model Validation

## Configuration Options

### Catalog and Schema

```python
# In notebook-00-setup.py
catalog = "production_ml"     # Your catalog
db = "churn_prediction_v2"   # Your schema
```

### Model Name

```python
# In notebook-00-setup.py
model_name = f"{catalog}.{db}.churn_xgboost"  # Custom model name
```

### Quality Gates

```python
# In notebook-04a-validation.py
quality_gates = {
    'accuracy': 0.70,   # Adjust based on your requirements
    'precision': 0.55,
    'recall': 0.50,
    'f1_score': 0.55,
    'roc_auc': 0.75
}
```

### Retraining Triggers

```python
# In notebook-08-retrain.py
degradation_threshold = 0.05  # 5% F1 drop triggers retrain
```

## Verification Checklist

After installation, verify:

- [ ] Catalog and schema created
- [ ] Bronze data loaded (~7K records)
- [ ] Feature table with 50+ columns
- [ ] Train/test splits created
- [ ] Model trained and registered
- [ ] Champion alias assigned
- [ ] Validation tests run (some may fail on small data)
- [ ] Batch predictions generated OR endpoint deployed
- [ ] Monitoring configured (optional but recommended)

## Common Issues

### Issue 1: Permission Errors

```
Error: User does not have CREATE CATALOG permission
```

**Solution:** Ask workspace admin to grant permissions or use existing catalog.

### Issue 2: Cluster Type Mismatch

```
Error: ML Runtime required
```

**Solution:** Create cluster with ML Runtime 14.3.x or higher.

### Issue 3: Unity Catalog Not Enabled

```
Error: Unity Catalog not found
```

**Solution:** Contact Databricks admin to enable Unity Catalog for workspace.

### Issue 4: Validation Tests Fail

```
Warning: SOME TESTS FAILED
```

**Solution:** This is expected with small dataset. Adjust quality gates or use manual approval.

## Cost Estimation

### One-Time Setup
- Training: ~$2
- Validation: ~$0.70
- Monitoring: ~$3
- **Total:** ~$6

### Monthly Operations
- Weekly batch (12 min): ~$5/month
- Daily monitoring (5 min): ~$20/month
- Monthly retrain: ~$2/month
- **Total:** ~$30/month

### Optional: Real-Time Endpoint
- Small (scale-to-zero): ~$50-100/month
- Medium: ~$200-300/month

## Support

- **GitHub Issues:** https://github.com/pravinva/mlops-advanced-drift-retrain/issues
- **Documentation:** See README.md
- **Databricks Community:** https://community.databricks.com

## Next Steps

After installation:

1. **Learn:** Review [00_mlops_end2end_advanced_presentation]($./00_mlops_end2end_advanced_presentation)
2. **Customize:** Adapt notebooks to your use case
3. **Deploy:** Choose batch or real-time serving
4. **Monitor:** Set up alerts for drift and degradation
5. **Automate:** Schedule retraining pipeline

## Uninstall

To remove the demo:

```sql
-- Drop all tables
DROP SCHEMA IF EXISTS <your_catalog>.<your_schema> CASCADE;

-- Drop catalog (if created by demo)
DROP CATALOG IF EXISTS <your_catalog> CASCADE;
```

```python
# Delete model versions from Unity Catalog
from mlflow.tracking import MlflowClient
client = MlflowClient()
client.delete_registered_model("<model_name>")
```

## License

This implementation is for educational and demonstration purposes.
