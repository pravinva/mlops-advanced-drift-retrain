# DBDemos Submission Guide

## Overview

This document provides instructions for submitting the MLOps Advanced Enhanced Demo to the Databricks Demo marketplace (DBDemos).

## Prerequisites

Before submission:
- [ ] All notebooks tested end-to-end
- [ ] Diagrams generated and hosted
- [ ] Documentation complete (README, TESTING.md)
- [ ] Code quality review completed
- [ ] No customer-specific references (ART, etc.)
- [ ] All dependencies documented

## Submission Package Structure

### Required Files

```
mlops-advanced/
├── 00_mlops_end2end_advanced_presentation.py  # Required: Overview notebook
├── 01-09_*.py                                 # Required: All workflow notebooks
├── _resources/                                # Required: Setup and utilities
│   ├── 00-setup.py
│   ├── 00-load-data.py
│   └── bundle_config.json
├── diagrams/                                  # Required: Visual assets
│   └── *.png
├── README.md                                  # Required: Documentation
├── TESTING.md                                # Recommended: Test guide
├── DIAGRAMS_GUIDE.md                         # Recommended: Diagram docs
└── conf/                                     # Optional: DBX configuration
    └── deployment.yml
```

### File Naming Conventions

**DBDemos follows strict naming:**
- Notebooks: `##_descriptive_name.py` (e.g., `01_feature_engineering.py`)
- Overview notebook: `00_*_presentation.py`
- Setup resources: `_resources/` folder
- No spaces in filenames
- Use underscores for readability

### Metadata Requirements

Each notebook should include:

```python
# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook Title
# MAGIC
# MAGIC <img src="DIAGRAM_URL" width="1200px">
# MAGIC
# MAGIC <!-- Tracking -->
# MAGIC <img width="1px" src="TRACKING_URL">
# MAGIC
# MAGIC ## Description
# MAGIC Brief description of what this notebook does
```

## Preparing for Submission

### 1. Clean Up Code

```bash
# Remove debug code
grep -r "print(" *.py | grep -v "# MAGIC"

# Remove commented code
grep -r "^#[^M]" *.py

# Check for hardcoded values
grep -r "pravinva\|pravin.varma" *.py
```

### 2. Validate Diagrams

Ensure all diagram URLs are accessible:

```bash
# Check diagram URLs in notebooks
grep -r "img src" *.py | grep -v "google-analytics"

# Test URLs (should return 200)
curl -I https://github.com/pravinva/mlops-advanced-drift-retrain/raw/main/diagrams/mlops-advanced-1-feature-store.png
```

### 3. Update Configuration

**_resources/00-setup.py:**
- Remove hardcoded usernames
- Use dynamic username: `dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()`
- Ensure catalog/schema names are configurable

```python
# Good
user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
catalog = "mlops_advanced"

# Bad
user = "pravin.varma@databricks.com"
catalog = "art_mlops"
```

### 4. Test Clean Installation

**Test on fresh workspace:**
1. Clone repo to new Databricks workspace
2. Run notebooks in sequence (00 → 09)
3. Verify all tables, models, jobs created
4. Check diagrams display correctly
5. Validate monitoring dashboards

### 5. Documentation Review

**README.md should include:**
- [ ] Clear overview of enhancements
- [ ] Architecture diagram
- [ ] Notebook descriptions
- [ ] Quick start guide
- [ ] Configuration instructions
- [ ] Prerequisites
- [ ] Learning objectives
- [ ] Support information

**TESTING.md should include:**
- [ ] Test categories
- [ ] Test procedures
- [ ] Expected results
- [ ] Troubleshooting guide

## DBDemos-Specific Requirements

### 1. Bundle Configuration

**_resources/bundle_config.json:**

```json
{
  "name": "mlops-advanced",
  "category": "ML",
  "title": "MLOps Advanced: Production-Ready ML with Automated Retraining",
  "description": "End-to-end MLOps pipeline with ground truth tracking, automated retraining, and production monitoring",
  "bundle": true,
  "tags": [
    {"mlops": "MLOps"},
    {"ml": "Machine Learning"},
    {"unity_catalog": "Unity Catalog"},
    {"monitoring": "Monitoring"},
    {"drift": "Drift Detection"},
    {"retraining": "Automated Retraining"}
  ],
  "notebooks": [
    {
      "path": "00_mlops_end2end_advanced_presentation",
      "pre_run": false,
      "publish_on_website": true,
      "add_cluster_setup_cell": false,
      "title": "MLOps Advanced Overview",
      "description": "Complete overview of the MLOps lifecycle with automated retraining"
    },
    {
      "path": "01_feature_engineering",
      "pre_run": false,
      "publish_on_website": true,
      "add_cluster_setup_cell": true,
      "title": "Feature Engineering",
      "description": "Build feature store with Unity Catalog and enable CDF"
    },
    {
      "path": "02_model_training_hpo",
      "pre_run": false,
      "publish_on_website": true,
      "add_cluster_setup_cell": true,
      "title": "Model Training with HPO",
      "description": "Train model with Optuna hyperparameter optimization (50 trials)"
    },
    {
      "path": "03_model_registration",
      "pre_run": false,
      "publish_on_website": true,
      "add_cluster_setup_cell": false,
      "title": "Model Registration",
      "description": "Register model to Unity Catalog with Champion/Challenger pattern"
    },
    {
      "path": "04a_challenger_validation",
      "pre_run": false,
      "publish_on_website": true,
      "add_cluster_setup_cell": false,
      "title": "Challenger Validation",
      "description": "7-stage automated validation framework"
    },
    {
      "path": "04b_challenger_approval",
      "pre_run": false,
      "publish_on_website": true,
      "add_cluster_setup_cell": false,
      "title": "Challenger Approval",
      "description": "Automated promotion logic with quality gates"
    },
    {
      "path": "05_batch_inference",
      "pre_run": false,
      "publish_on_website": true,
      "add_cluster_setup_cell": false,
      "title": "Batch Inference",
      "description": "Distributed Spark scoring at scale"
    },
    {
      "path": "06_model_serving",
      "pre_run": false,
      "publish_on_website": true,
      "add_cluster_setup_cell": false,
      "title": "Model Serving (Optional)",
      "description": "Deploy real-time REST API with auto-scaling"
    },
    {
      "path": "07_model_monitoring",
      "pre_run": false,
      "publish_on_website": true,
      "add_cluster_setup_cell": false,
      "title": "Production Monitoring + Ground Truth",
      "description": "Lakehouse Monitoring with real production accuracy tracking"
    },
    {
      "path": "08_automated_retraining",
      "pre_run": false,
      "publish_on_website": true,
      "add_cluster_setup_cell": false,
      "title": "Automated Retraining",
      "description": "6 intelligent retraining triggers with automated workflow"
    },
    {
      "path": "09_deployment_job",
      "pre_run": false,
      "publish_on_website": true,
      "add_cluster_setup_cell": false,
      "title": "Deployment Job Orchestration",
      "description": "End-to-end Databricks Job with task dependencies"
    }
  ],
  "cluster": {
    "spark_version": "14.3.x-cpu-ml-scala2.12",
    "node_type_id": "i3.xlarge",
    "num_workers": 2,
    "spark_conf": {
      "spark.databricks.delta.preview.enabled": "true"
    }
  },
  "init_job": {
    "settings": {
      "name": "mlops_advanced_init",
      "email_notifications": {},
      "timeout_seconds": 7200,
      "max_concurrent_runs": 1,
      "tasks": [
        {
          "task_key": "init_data",
          "notebook_task": {
            "notebook_path": "{{DEMO_FOLDER}}/_resources/00-load-data",
            "base_parameters": {}
          },
          "job_cluster_key": "Shared_job_cluster"
        }
      ],
      "job_clusters": [
        {
          "job_cluster_key": "Shared_job_cluster",
          "new_cluster": {
            "spark_version": "14.3.x-cpu-ml-scala2.12",
            "node_type_id": "i3.xlarge",
            "num_workers": 2
          }
        }
      ]
    }
  },
  "pipelines": [],
  "dashboards": []
}
```

### 2. Tracking Pixels

Add Google Analytics tracking to each notebook:

```python
# MAGIC <!-- Tracking -->
# MAGIC <img width="1px" src="https://www.google-analytics.com/collect?v=1&gtm=GTM-NKQ8TT7&tid=UA-163989034-1&cid=555&aip=1&t=event&ec=field_demos&ea=display&dp=%2F42_field_demos%2Fml%2Fmlops_advanced%2F##&dt=MLOPS_ADVANCED">
```

Replace `##` with notebook number (01, 02, etc.)

### 3. Cluster Setup Cells

For notebooks that require specific libraries, add setup cells:

```python
# DBTITLE 1,Install Required Libraries
# MAGIC %pip install optuna lightgbm scikit-learn --quiet
# MAGIC dbutils.library.restartPython()
```

### 4. Widget Configuration

Use widgets for configuration:

```python
dbutils.widgets.dropdown("reset_all_data", "false", ["true", "false"], "Reset all data")
dbutils.widgets.text("catalog", "mlops_advanced", "Catalog Name")
dbutils.widgets.text("db", "churn_demo", "Schema Name")
```

## Diagram Hosting

### Option 1: Upload to dbdemos-resources Repo

If accepted as official demo:
1. Fork: https://github.com/databricks-demos/dbdemos-resources
2. Create PR with diagrams:
   ```
   images/product/mlops/advanced/mlops-advanced-*.png
   ```
3. Update notebook URLs after merge

### Option 2: Use Your GitHub Repo (Current)

Diagrams currently hosted at:
```
https://github.com/pravinva/mlops-advanced-drift-retrain/raw/main/diagrams/mlops-advanced-*.png
```

**Pros:**
- Immediate availability
- Full control

**Cons:**
- Not official Databricks hosting
- Requires public repo

## Submission Process

### 1. Internal Review (Databricks SAs)

**Submit to Databricks Solutions Architects:**
- Email: field-eng@databricks.com
- Subject: "MLOps Advanced Demo Enhancement Proposal"
- Include:
  - GitHub repo link
  - README.md
  - Enhancement summary (key improvements over original)
  - Test results

**Review Checklist:**
- Technical accuracy
- Code quality
- Documentation completeness
- Databricks best practices
- Customer value proposition

### 2. Demo Marketplace Submission

**After SA approval:**
1. Fork: https://github.com/databricks-demos/dbdemos
2. Add demo to `demos/` folder:
   ```
   demos/ml/mlops-advanced-enhanced/
   ```
3. Create PR with:
   - All notebook files
   - `_resources/` folder
   - `bundle_config.json`
   - README.md

**PR Description Template:**
```markdown
# MLOps Advanced: Enhanced with Production Features

## Overview
Enhancement proposal for the existing MLOps Advanced demo, adding:
- Ground truth integration
- Automated retraining (6 triggers)
- Enhanced validation (7 stages)
- Production monitoring
- Deployment job orchestration

## Testing
-  Tested on DBR 14.3 ML
-  Unity Catalog enabled
-  All notebooks run successfully
-  End-to-end workflow validated

## Enhancements Over Original
- 4 new notebooks (07, 08, 09, 04b)
- 11 professional diagrams
- Comprehensive documentation
- Production-ready features

## Links
- Demo repo: https://github.com/pravinva/mlops-advanced-drift-retrain
- Documentation: README.md, TESTING.md, DIAGRAMS_GUIDE.md
```

### 3. Community Feedback

**After PR submission:**
- Monitor PR comments
- Address feedback promptly
- Update code/docs as needed
- Participate in review discussions

## Post-Submission

### Maintenance

If accepted:
- Monitor GitHub issues
- Address bug reports
- Update for new Databricks features
- Keep diagrams current
- Update dependencies

### Promotion

**Share demo:**
- Databricks Community forums
- LinkedIn posts
- Blog posts
- Conference presentations
- Customer workshops

## Quality Checklist

### Code Quality

- [ ] No hardcoded credentials
- [ ] No customer-specific references
- [ ] Consistent coding style
- [ ] Meaningful variable names
- [ ] Adequate comments
- [ ] Error handling implemented
- [ ] No debug print statements

### Documentation Quality

- [ ] README complete and accurate
- [ ] All notebooks have descriptions
- [ ] Diagrams display correctly
- [ ] Links are valid
- [ ] Learning objectives clear
- [ ] Prerequisites listed
- [ ] Configuration documented

### Testing Quality

- [ ] All notebooks run successfully
- [ ] End-to-end workflow tested
- [ ] Validation tests pass
- [ ] Performance acceptable
- [ ] No resource leaks
- [ ] Clean workspace after execution

### User Experience

- [ ] Clear instructions
- [ ] Logical notebook order
- [ ] Helpful error messages
- [ ] Progress indicators
- [ ] Output formatting clean
- [ ] Widgets properly labeled

## Troubleshooting Submission

### Common Issues

**Issue:** Notebooks fail on clean workspace
- **Fix:** Test on fresh workspace with no existing data
- **Verify:** `reset_all_data=true` works correctly

**Issue:** Diagrams don't display
- **Fix:** Check URL accessibility (must be public)
- **Verify:** `curl -I <diagram_url>` returns 200

**Issue:** Unity Catalog permissions
- **Fix:** Document required permissions in README
- **Verify:** Test with limited-permission user

**Issue:** Cluster configuration
- **Fix:** Specify exact runtime version and node types
- **Verify:** Works on standard cluster configs

## Contact

**For submission questions:**
- Databricks SAs: field-eng@databricks.com
- DBDemos team: Via GitHub issues on dbdemos repo

**For technical issues:**
- GitHub Issues: https://github.com/pravinva/mlops-advanced-drift-retrain/issues

---

## Next Steps

1. Complete pre-submission checklist
2. Test on clean workspace
3. Submit for internal SA review
4. Address feedback
5. Submit PR to dbdemos repo
6. Monitor and maintain

---

**Version:** 1.0
**Last Updated:** 2024-11-25
**Status:** Ready for submission after SA review
