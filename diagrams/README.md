# MLOps Advanced - Diagram Files

This directory contains 9 HTML/SVG diagrams for the MLOps Advanced demo.

## Diagrams Overview

- **diagram_0_overview.html** - Complete MLOps Lifecycle with Automated Retraining
- **diagram_1_features.html** - Feature Engineering with Unity Catalog
- **diagram_2_training.html** - Model Training with Hyperparameter Optimization
- **diagram_3_registration.html** - Model Registration to Unity Catalog
- **diagram_4_validation.html** - 7-Stage Automated Validation Framework
- **diagram_5_promotion.html** - Automated Promotion Workflow
- **diagram_6_batch_inference.html** - Batch Inference at 2.4M Scale
- **diagram_7_monitoring.html** - Production Monitoring with Ground Truth
- **diagram_8_retraining.html** - Automated Retraining & Deployment Job

## Converting HTML to PNG (High Quality)

### Option 1: Browser Screenshot (Manual)
1. Open any `.html` file in Chrome/Firefox
2. Press F12 to open DevTools
3. Press Ctrl+Shift+P (Cmd+Shift+P on Mac) to open Command Palette
4. Type "screenshot" and select "Capture full size screenshot"
5. Save as PNG with the same filename (e.g., `diagram_0_overview.png`)

### Option 2: Automated Conversion (Python)
Use the provided script to convert all HTML files to PNG automatically:

```bash
# Install playwright (one-time setup)
pip install playwright
python -m playwright install chromium

# Run conversion script
python html_to_png_converter.py
```

### Option 3: Online Tools
- Upload HTML file to: https://html2canvas.hertzen.com/
- Or use: https://www.screenshotmachine.com/

## Design Specifications

- **Brand Colors:**
  - Navy: #1B3139
  - Cyan: #00A8E1
  - Lava: #FF3621
  - Gray Light: #F5F5F5
  - Gray Medium: #E0E0E0
  - Gray Dark: #666666

- **Typography:**
  - Font: Barlow (Google Fonts)
  - Title: 700 weight, 28px
  - Label: 500 weight, 16px
  - Text: 400 weight, 14px

- **Dimensions:**
  - Width: 1200px (standard)
  - Height: Varies by diagram (400-800px)

## Usage in Notebooks

Once converted to PNG, reference them in Databricks notebooks using:

```python
# MAGIC %md
# MAGIC <img src="https://github.com/your-repo/raw/main/diagrams/diagram_0_overview.png" width="1200px">
```

Or for local testing:
```python
# MAGIC %md
# MAGIC <img src="./diagrams/diagram_0_overview.png" width="1200px">
```

## Maintenance

To regenerate all diagrams:
```bash
cd /Users/pravin.varma/Documents/Demo/mlops-advanced
python3 diagram_generator.py
```

This will update all HTML files in the `diagrams/` directory.
