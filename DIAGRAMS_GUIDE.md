# MLOps Advanced Diagrams - Complete Guide

## Overview

Created 9 professional Databricks-branded diagrams for the MLOps Advanced demo using HTML/CSS approach. This provides superior quality compared to Python matplotlib/plotly methods.

## What Was Created

### 1. Diagram Generator (`diagram_generator.py`)
Python script that generates 9 standalone HTML/SVG diagrams:

- **Diagram 0**: Complete MLOps Lifecycle Overview with Automated Retraining Loop
- **Diagram 1**: Feature Engineering with Unity Catalog
- **Diagram 2**: Model Training with Hyperparameter Optimization (Optuna + LightGBM)
- **Diagram 3**: Model Registration to Unity Catalog (Champion/Challenger pattern)
- **Diagram 4**: 7-Stage Automated Validation Framework
- **Diagram 5**: Automated Promotion Workflow (Challenger → Champion)
- **Diagram 6**: Batch Inference at 2.4M Scale (Distributed Spark)
- **Diagram 7**: Production Monitoring with Ground Truth Integration
- **Diagram 8**: Automated Retraining Triggers & Deployment Job

### 2. HTML Diagram Files (`diagrams/*.html`)
9 self-contained HTML files with:
- Databricks brand colors (Navy, Cyan, Lava)
- Barlow font family (Google Fonts)
- SVG vector graphics (scalable, high quality)
- 1200px width (standard banner size)
- Responsive layout with Flexbox

### 3. HTML to PNG Converter (`html_to_png_converter.py`)
Automated script to convert all HTML diagrams to PNG images using Playwright.

### 4. Documentation
- `diagrams/README.md` - Usage instructions
- This guide - Complete overview

## Why HTML/CSS vs Python matplotlib?

### HTML/CSS Advantages:
- ✅ **Instant preview**: Just refresh browser to see changes
- ✅ **No regeneration**: Edit CSS, see results immediately
- ✅ **Vector output**: SVG scales perfectly at any resolution
- ✅ **Brand consistency**: CSS variables for colors, fonts
- ✅ **10x faster iteration**: No Python dependencies, no rendering wait
- ✅ **Professional quality**: Browser rendering engine = publication-ready

### Python matplotlib/plotly Disadvantages:
- ❌ Trial and error positioning (manual coordinates)
- ❌ Raster output (PNG doesn't scale)
- ❌ Slow iteration (regenerate on every change)
- ❌ Complex code (~50 lines for simple diagram)

## Usage Guide

### Regenerating HTML Diagrams

```bash
cd /Users/pravin.varma/Documents/Demo/mlops-advanced
python3 diagram_generator.py
```

This creates/updates all 9 HTML files in `diagrams/` directory.

### Converting HTML to PNG

**Option 1: Automated (Recommended)**
```bash
# One-time setup
pip install playwright
python -m playwright install chromium

# Convert all diagrams
python3 html_to_png_converter.py
```

**Option 2: Manual (Browser)**
1. Open `diagrams/diagram_0_overview.html` in Chrome/Firefox
2. Press `F12` to open DevTools
3. Press `Ctrl+Shift+P` (Mac: `Cmd+Shift+P`)
4. Type "screenshot" → Select "Capture full size screenshot"
5. Save as `diagram_0_overview.png`
6. Repeat for all 9 diagrams

**Option 3: Command Line (macOS)**
```bash
# Using built-in webkit2png or wkhtmltoimage
for file in diagrams/*.html; do
    base=$(basename "$file" .html)
    /Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome \
        --headless --screenshot="diagrams/${base}.png" \
        --window-size=1200,800 "$file"
done
```

### Using Diagrams in Notebooks

Once converted to PNG, reference them in Databricks notebooks:

```python
# MAGIC %md
# MAGIC <img src="https://github.com/your-repo/raw/main/diagrams/diagram_0_overview.png" width="1200px">
```

Or for local/dbdemos submission:
```python
# MAGIC %md
# MAGIC <img src="./diagrams/diagram_0_overview.png" width="1200px">
```

## Design Specifications

### Databricks Brand Colors
```css
Navy:        #1B3139  (primary text, boxes)
Cyan:        #00A8E1  (accent, arrows, highlights)
Lava:        #FF3621  (important highlights, retraining loop)
White:       #FFFFFF  (box backgrounds)
Gray Light:  #F5F5F5  (feature boxes)
Gray Medium: #E0E0E0  (borders)
Gray Dark:   #666666  (secondary text)
```

### Typography
```css
Font Family: 'Barlow', sans-serif
Title:       700 weight, 28px
Label:       500 weight, 16px
Text:        400 weight, 14px
Highlight:   700 weight, 14px (cyan color)
```

### Layout
- **Width**: 1200px (standard banner)
- **Height**: Varies (400-800px depending on content)
- **Format**: SVG embedded in HTML
- **Positioning**: Manual coordinates (simple boxes + arrows)

## Diagram Content Summary

### Diagram 0: Complete Lifecycle (800px height)
Shows end-to-end MLOps workflow with automated retraining loop:
- Data → Features → Training → Registry → Validation → Approval
- Batch Inference → Monitoring → Drift Detection
- Automated retraining loop (highlighted in lava red)
- Key callouts: Ground truth, Real accuracy, Closed-loop MLOps

### Diagram 1: Feature Engineering (400px)
Bronze → Feature Engineering → Unity Catalog Feature Store:
- IBM Telco 7K customers
- Clean TotalCharges, one-hot encoding, binary labels
- 80/20 split
- Change Data Feed (CDF) enabled for drift tracking

### Diagram 2: Model Training (450px)
Training Data → Optuna HPO → MLflow Tracking:
- 50 trials (not simplified, enterprise-grade)
- LightGBM binary classification
- F1 score maximization
- Early stopping, 300 rounds
- Signature + input example

### Diagram 3: Model Registration (400px)
MLflow Run → Unity Catalog Registry → Aliases:
- @Challenger (new model)
- @Champion (production)
- @Previous (rollback)
- Full lineage, tags, descriptions

### Diagram 4: 7-Stage Validation (550px)
Comprehensive validation framework:
1. Performance metrics (F1≥0.55, Acc≥0.70, Prec≥0.55)
2. Confusion matrix (FNR ≤50%)
3. Class performance (minority F1≥0.50)
4. Prediction distribution (no all-0 or all-1)
5. Inference latency (<50ms)
6. Champion comparison (F1 delta ≥-2%)
7. Robustness (handles 10% missing values)

### Diagram 5: Automated Promotion (500px)
Decision logic → Promotion workflow → Production ready:
- IF validation PASSED AND (no Champion OR F1 improved)
- Archive: Champion → Previous_Champion
- Promote: Challenger → Champion
- Zero-downtime, rollback capability

### Diagram 6: Batch Inference (500px)
Champion → Load with Spark UDF → Distributed scoring:
- 2.4M customers in 200+ partitions
- 12 minutes end-to-end (vs 16+ hours sequential)
- Output: Delta table with predictions
- Zero-code updates via @Champion alias

### Diagram 7: Production Monitoring (650px)
Predictions → Lakehouse Monitoring → Ground Truth:
- PSI, KS-test, null tracking, freshness checks
- 3-month lag ground truth integration
- Real F1/Precision/Recall calculation
- Alert if production metrics < training by 5%+
- Unified dashboard for all stakeholders

### Diagram 8: Automated Retraining (700px)
6 triggers → Databricks Job → 6-task workflow:
- Triggers: Feature drift, prediction drift, null spike, F1 drop, data quality, monthly fallback
- Orchestrated: Feature engineering → Training → Registration → Validation → Approval → Inference
- Quality gates: All 7 tests PASS + F1 improvement
- Result: New @Champion in production

## Maintenance

### Editing Diagrams
1. Open `diagram_generator.py`
2. Find the relevant `generate_diagram_X()` function
3. Edit boxes, arrows, labels using helper functions:
   - `box(x, y, width, height, text, class)` - Create boxes
   - `arrow(x1, y1, x2, y2, highlight=False)` - Create arrows
   - `title(text, x, y)` - Create titles
   - `label(text, x, y, highlight=False)` - Create labels
4. Run `python3 diagram_generator.py` to regenerate
5. Open HTML in browser to preview
6. Iterate until satisfied
7. Convert to PNG

### Adding New Diagrams
1. Create new function `generate_diagram_9()` in `diagram_generator.py`
2. Follow existing pattern:
   ```python
   def generate_diagram_9():
       """Diagram 9: Description"""
       svg = generate_svg_header(width, height)
       svg += title("Diagram Title")
       # Add boxes, arrows, labels
       svg += generate_svg_footer()
       return svg
   ```
3. Add to `diagrams` dict in `generate_all_diagrams()`
4. Regenerate and convert

## File Structure

```
mlops-advanced/
├── diagram_generator.py           # Main generator script
├── html_to_png_converter.py       # PNG conversion script
├── DIAGRAMS_GUIDE.md             # This file
└── diagrams/
    ├── README.md                 # Quick reference
    ├── diagram_0_overview.html   # Lifecycle overview
    ├── diagram_1_features.html   # Feature engineering
    ├── diagram_2_training.html   # Model training
    ├── diagram_3_registration.html  # Model registry
    ├── diagram_4_validation.html # Validation framework
    ├── diagram_5_promotion.html  # Promotion workflow
    ├── diagram_6_batch_inference.html  # Batch scoring
    ├── diagram_7_monitoring.html # Monitoring + ground truth
    └── diagram_8_retraining.html # Automated retraining
```

## Next Steps

1. **Convert HTML to PNG**:
   ```bash
   python3 html_to_png_converter.py
   ```

2. **Update Notebook Image URLs**: Replace placeholder URLs in notebooks 01-09 with actual diagram paths

3. **Test in Databricks**: Upload diagrams and verify they display correctly in notebooks

4. **Submit to DBDemos**: Include diagrams in submission package

## Support

For issues or modifications:
- Edit `diagram_generator.py` and regenerate
- Open HTML files in browser to preview before converting
- Use browser DevTools to inspect SVG elements and adjust coordinates
