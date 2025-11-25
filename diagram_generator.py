#!/usr/bin/env python3
"""
MLOps Diagram Generator - Databricks Brand Compliant
Creates HTML/SVG architectural diagrams for the MLOps Advanced demo
"""

import os
from pathlib import Path

# Databricks Brand Colors
COLORS = {
    'navy': '#1B3139',
    'cyan': '#00A8E1',
    'lava': '#FF3621',
    'white': '#FFFFFF',
    'gray_light': '#F5F5F5',
    'gray_medium': '#E0E0E0',
    'gray_dark': '#666666'
}

def generate_svg_header(width=1200, height=600):
    """Generate SVG header with Databricks branding"""
    return f'''<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Barlow:wght@400;500;700&amp;display=swap');

      .db-title {{
        font-family: 'Barlow', sans-serif;
        font-weight: 700;
        font-size: 28px;
        fill: {COLORS['navy']};
      }}

      .db-label {{
        font-family: 'Barlow', sans-serif;
        font-weight: 500;
        font-size: 16px;
        fill: {COLORS['navy']};
      }}

      .db-text {{
        font-family: 'Barlow', sans-serif;
        font-weight: 400;
        font-size: 14px;
        fill: {COLORS['gray_dark']};
      }}

      .db-highlight {{
        font-family: 'Barlow', sans-serif;
        font-weight: 700;
        font-size: 14px;
        fill: {COLORS['cyan']};
      }}

      .db-box {{
        fill: {COLORS['white']};
        stroke: {COLORS['cyan']};
        stroke-width: 2;
        rx: 8;
      }}

      .db-box-feature {{
        fill: {COLORS['gray_light']};
        stroke: {COLORS['navy']};
        stroke-width: 2;
        rx: 8;
      }}

      .db-arrow {{
        stroke: {COLORS['cyan']};
        stroke-width: 3;
        fill: none;
        marker-end: url(#arrowhead);
      }}

      .db-arrow-highlight {{
        stroke: {COLORS['lava']};
        stroke-width: 3;
        fill: none;
        marker-end: url(#arrowhead-lava);
      }}
    </style>

    <marker id="arrowhead" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
      <polygon points="0 0, 10 3, 0 6" fill="{COLORS['cyan']}" />
    </marker>

    <marker id="arrowhead-lava" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
      <polygon points="0 0, 10 3, 0 6" fill="{COLORS['lava']}" />
    </marker>
  </defs>

  <rect width="{width}" height="{height}" fill="{COLORS['white']}"/>
'''

def generate_svg_footer():
    """Generate SVG footer"""
    return '</svg>'

def box(x, y, width, height, text, box_class="db-box"):
    """Generate a rounded box with text"""
    lines = text.split('\n')
    text_y = y + height/2 - (len(lines) - 1) * 10

    box_svg = f'  <rect x="{x}" y="{y}" width="{width}" height="{height}" class="{box_class}"/>\n'
    for i, line in enumerate(lines):
        text_y_pos = text_y + i * 20
        box_svg += f'  <text x="{x + width/2}" y="{text_y_pos}" class="db-label" text-anchor="middle">{line}</text>\n'

    return box_svg

def arrow(x1, y1, x2, y2, highlight=False):
    """Generate an arrow"""
    arrow_class = "db-arrow-highlight" if highlight else "db-arrow"
    return f'  <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" class="{arrow_class}"/>\n'

def title(text, x=600, y=40):
    """Generate title text"""
    return f'  <text x="{x}" y="{y}" class="db-title" text-anchor="middle">{text}</text>\n'

def label(text, x, y, highlight=False):
    """Generate label text"""
    text_class = "db-highlight" if highlight else "db-text"
    return f'  <text x="{x}" y="{y}" class="{text_class}" text-anchor="middle">{text}</text>\n'

def generate_diagram_0():
    """Diagram 0: Complete MLOps Lifecycle Overview"""
    svg = generate_svg_header(1200, 800)
    svg += title("MLOps Advanced - Complete Lifecycle with Automated Retraining")

    # Row 1: Data → Features → Training
    svg += box(50, 100, 150, 80, "Raw Data\nIBM Telco")
    svg += arrow(200, 140, 250, 140)
    svg += box(250, 100, 180, 80, "Feature\nEngineering\nUnity Catalog")
    svg += arrow(430, 140, 480, 140)
    svg += box(480, 100, 200, 80, "Training + HPO\nOptuna 50 trials\nLightGBM")
    svg += arrow(680, 140, 730, 140)

    # Row 2: Registry → Validation → Approval
    svg += box(730, 100, 180, 80, "Unity Catalog\nChampion/\nChallenger")
    svg += arrow(910, 140, 960, 140)
    svg += box(960, 100, 200, 80, "7-Stage\nValidation\n(04a)")

    # Arrow down to approval
    svg += arrow(1060, 180, 1060, 230)
    svg += box(960, 230, 200, 80, "Auto-Promotion\n(04b)")

    # Row 3: Inference
    svg += arrow(1060, 310, 1060, 360)
    svg += box(880, 360, 360, 80, "Batch Inference (Champion Model)\nDistributed Spark • Business Impact")

    # Row 4: Monitoring with Ground Truth
    svg += arrow(1060, 440, 1060, 490)
    svg += box(730, 490, 560, 100, "Production Monitoring + Ground Truth (3mo delay)\nReal F1/Precision/Recall • Feature Drift • Prediction Drift", "db-box-feature")

    # Row 5: Drift Detection & Retraining
    svg += arrow(1010, 590, 1010, 640)
    svg += box(730, 640, 560, 80, "Drift Detection: 6 Triggers\nPSI>0.25 • F1 Drop>5% • Null Spike • Time-based")

    # Retraining loop back to top (highlighted)
    svg += arrow(730, 680, 400, 680, highlight=True)
    svg += arrow(400, 680, 400, 90, highlight=True)
    svg += arrow(400, 90, 480, 90, highlight=True)

    svg += label("⟲ Automated Retraining Loop", 565, 700, highlight=True)

    # Key callouts
    svg += label("✓ Ground Truth Integration", 200, 550, highlight=True)
    svg += label("✓ Real Accuracy Tracking", 600, 550, highlight=True)
    svg += label("✓ Closed-Loop MLOps", 1000, 550, highlight=True)

    svg += generate_svg_footer()
    return svg

def generate_diagram_1():
    """Diagram 1: Feature Engineering"""
    svg = generate_svg_header(1200, 400)
    svg += title("Feature Engineering with Unity Catalog")

    # Bronze layer
    svg += box(100, 150, 200, 100, "Bronze Layer\nIBM Telco\n7K customers\nRaw CSV")
    svg += arrow(300, 200, 380, 200)

    # Feature engineering
    svg += box(380, 120, 240, 160, "Feature Engineering\n• Clean TotalCharges\n• One-hot encoding\n• Binary labels\n• 80/20 split")
    svg += arrow(620, 200, 700, 200)

    # Feature store
    svg += box(700, 150, 200, 100, "Unity Catalog\nFeature Table\n+ CDF enabled")

    # Highlight
    svg += label("Change Data Feed (CDF) for drift tracking", 600, 320, highlight=True)

    svg += generate_svg_footer()
    return svg

def generate_diagram_2():
    """Diagram 2: Model Training with HPO"""
    svg = generate_svg_header(1200, 450)
    svg += title("Model Training with Hyperparameter Optimization")

    # Training data
    svg += box(100, 150, 180, 100, "Training Data\n80% train\n20% test\nBalanced")
    svg += arrow(280, 200, 350, 200)

    # Optuna HPO
    svg += box(350, 120, 220, 160, "Optuna HPO\n50 Trials\nLightGBM\nBinary Class\nF1 Score max")
    svg += arrow(570, 200, 650, 200)

    # MLflow tracking
    svg += box(650, 150, 200, 100, "MLflow Tracking\nAll 50 trials\nBest F1 logged\nArtifacts saved")

    # Final model (below)
    svg += arrow(750, 250, 750, 300)
    svg += box(600, 300, 300, 100, "Final Champion Model\nBest hyperparameters\n300 rounds • Early stopping\nSignature + Input example")

    # Highlight
    svg += label("Enterprise-grade: Full 50-trial optimization (not simplified)", 600, 50, highlight=True)

    svg += generate_svg_footer()
    return svg

def generate_diagram_3():
    """Diagram 3: Model Registration"""
    svg = generate_svg_header(1200, 400)
    svg += title("Model Registration to Unity Catalog")

    # MLflow run
    svg += box(100, 150, 180, 100, "MLflow Run\nTrained model\nArtifacts\nMetadata")
    svg += arrow(280, 200, 380, 200)

    # Unity Catalog
    svg += box(380, 130, 280, 140, "Unity Catalog Registry\ncatalog.schema.model\nVersion N\nFull lineage\nTags + Description")
    svg += arrow(660, 200, 760, 200)

    # Aliases
    svg += box(760, 100, 200, 220, "@Challenger\n(new model)\n\n@Champion\n(production)\n\n@Previous\n(rollback)")

    # Highlight
    svg += label("Governance: Champion/Challenger pattern", 600, 340, highlight=True)

    svg += generate_svg_footer()
    return svg

def generate_diagram_4():
    """Diagram 4: 7-Stage Validation"""
    svg = generate_svg_header(1200, 550)
    svg += title("7-Stage Automated Validation Framework")

    # Challenger at top
    svg += box(480, 80, 240, 60, "@Challenger Model")
    svg += arrow(600, 140, 600, 180)

    # Validation framework
    svg += box(200, 180, 800, 280, "", "db-box-feature")

    # Test list
    tests = [
        ("1. Performance Metrics", "F1≥0.55, Acc≥0.70, Prec≥0.55"),
        ("2. Confusion Matrix", "False Negative Rate ≤50%"),
        ("3. Class Performance", "Minority class F1≥0.50"),
        ("4. Prediction Distribution", "No all-0 or all-1 predictions"),
        ("5. Inference Latency", "Single prediction <50ms"),
        ("6. Champion Comparison", "F1 delta ≥-2% vs Champion"),
        ("7. Robustness", "Handles 10% missing values")
    ]

    y_pos = 210
    for test_name, criteria in tests:
        svg += f'  <text x="220" y="{y_pos}" class="db-label">{test_name}</text>\n'
        svg += f'  <text x="480" y="{y_pos}" class="db-text">{criteria}</text>\n'
        y_pos += 35

    # Results
    svg += arrow(600, 460, 520, 510)
    svg += box(380, 510, 160, 40, "PASS →")
    svg += label("04b_approval.py", 630, 530)

    svg += arrow(600, 460, 680, 510)
    svg += box(660, 510, 160, 40, "FAIL →")
    svg += label("Review & retrain", 930, 530)

    # Highlight
    svg += label("Realistic quality gates for imbalanced churn datasets", 600, 40, highlight=True)

    svg += generate_svg_footer()
    return svg

def generate_diagram_5():
    """Diagram 5: Automated Promotion"""
    svg = generate_svg_header(1200, 500)
    svg += title("Automated Promotion Workflow")

    # Validation results
    svg += box(480, 80, 240, 60, "Validation Results\nfrom 04a")
    svg += arrow(600, 140, 600, 180)

    # Decision logic
    svg += box(400, 180, 400, 80, "Decision Logic:\nIF validation PASSED\nAND (no Champion OR F1 improved)")
    svg += arrow(600, 260, 600, 300)

    # Promotion workflow
    svg += box(300, 300, 600, 120, "", "db-box-feature")
    svg += f'  <text x="600" y="330" class="db-label" text-anchor="middle">Promotion Workflow</text>\n'
    svg += f'  <text x="320" y="360" class="db-text">1. Archive: Champion → Previous_Champion</text>\n'
    svg += f'  <text x="320" y="385" class="db-text">2. Promote: Challenger → Champion</text>\n'
    svg += f'  <text x="320" y="410" class="db-text">3. Remove: Challenger alias</text>\n'

    # Production ready
    svg += arrow(600, 420, 600, 460)
    svg += box(400, 460, 400, 40, "Production Ready @Champion")

    # Highlight
    svg += label("Zero-downtime promotion with rollback capability", 600, 40, highlight=True)

    svg += generate_svg_footer()
    return svg

def generate_diagram_6():
    """Diagram 6: Batch Inference at Scale"""
    svg = generate_svg_header(1200, 500)
    svg += title("Batch Inference: Distributed Scoring at 2.4M Scale")

    # Champion model
    svg += box(100, 150, 220, 80, "Unity Catalog\\n@Champion\\nModel")
    svg += arrow(320, 190, 380, 190)

    # Load model
    svg += box(380, 150, 240, 80, "Load with\\nmlflow.pyfunc\\nSpark UDF")
    svg += arrow(620, 190, 700, 190)

    # Distributed scoring
    svg += box(700, 100, 420, 180, "", "db-box-feature")
    svg += f'  <text x="910" y="130" class="db-label" text-anchor="middle">Distributed Spark Scoring</text>\n'
    svg += f'  <text x="720" y="160" class="db-text">• 2.4M customers in 200+ partitions</text>\n'
    svg += f'  <text x="720" y="185" class="db-text">• Parallel execution across cluster</text>\n'
    svg += f'  <text x="720" y="210" class="db-text">• 12 minutes end-to-end</text>\n'
    svg += f'  <text x="720" y="235" class="db-text">• Output: Delta table with predictions</text>\n'
    svg += f'  <text x="720" y="260" class="db-text">• Column: churn_probability [0-1]</text>\n'

    # Output
    svg += arrow(910, 280, 910, 330)
    svg += box(700, 330, 420, 80, "Delta Table: main.predictions.weekly_churn_scores\\nPartitioned by prediction_date • Ready for Power BI")

    # Highlight
    svg += label("12 minutes vs 16+ hours sequential", 200, 50, highlight=True)
    svg += label("Zero-code updates: @Champion alias auto-loaded", 800, 50, highlight=True)

    svg += generate_svg_footer()
    return svg

def generate_diagram_6_serving():
    """Diagram 6: Real-Time Model Serving (Optional)"""
    svg = generate_svg_header(1200, 500)
    svg += title("Real-Time Model Serving with Auto-Scaling")

    # Champion model
    svg += box(100, 150, 200, 80, "Unity Catalog\\n@Champion\\nModel")
    svg += arrow(300, 190, 370, 190)

    # Serving endpoint
    svg += box(370, 100, 460, 180, "", "db-box-feature")
    svg += f'  <text x="600" y="130" class="db-label" text-anchor="middle">Model Serving Endpoint</text>\n'
    svg += f'  <text x="390" y="160" class="db-text">• REST API with auto-scaling</text>\n'
    svg += f'  <text x="390" y="185" class="db-text">• Scale-to-zero enabled (cost-effective)</text>\n'
    svg += f'  <text x="390" y="210" class="db-text">• Workload size: Small/Medium/Large</text>\n'
    svg += f'  <text x="390" y="235" class="db-text">• Automatic inference logging</text>\n'
    svg += f'  <text x="390" y="260" class="db-text">• Latency: <50ms P95</text>\n'

    # Inference logging
    svg += arrow(600, 280, 600, 330)
    svg += box(420, 330, 360, 80, "Inference Table\\n(Auto-generated)\\nLogged predictions + inputs")

    # API clients
    svg += arrow(830, 190, 900, 140)
    svg += box(900, 100, 200, 80, "API Clients\\nPython • cURL\\nJavaScript")

    # Highlight
    svg += label("Optional: Use if real-time predictions needed", 300, 50, highlight=True)
    svg += label("Otherwise use Batch Inference (Notebook 05)", 800, 50, highlight=True)

    svg += generate_svg_footer()
    return svg

def generate_diagram_7():
    """Diagram 7: Production Monitoring with Ground Truth"""
    svg = generate_svg_header(1200, 650)
    svg += title("Production Monitoring: Lakehouse Monitoring + Ground Truth")

    # Predictions table
    svg += box(100, 100, 200, 80, "Prediction Table\\nweekly_churn_\\nscores")
    svg += arrow(300, 140, 370, 140)

    # Lakehouse Monitoring
    svg += box(370, 80, 460, 120, "", "db-box-feature")
    svg += f'  <text x="600" y="110" class="db-label" text-anchor="middle">Lakehouse Monitoring</text>\n'
    svg += f'  <text x="390" y="140" class="db-text">• PSI (Population Stability Index)</text>\n'
    svg += f'  <text x="390" y="165" class="db-text">• KS-test for distribution shifts</text>\n'
    svg += f'  <text x="390" y="190" class="db-text">• Null % tracking, freshness checks</text>\n'

    # Ground truth integration
    svg += arrow(600, 200, 600, 250)
    svg += box(370, 250, 460, 140, "", "db-box-feature")
    svg += f'  <text x="600" y="280" class="db-label" text-anchor="middle">Ground Truth Integration (3-month lag)</text>\n'
    svg += f'  <text x="390" y="310" class="db-text">August predictions → November outcomes known</text>\n'
    svg += f'  <text x="390" y="335" class="db-text">Join: prediction_id + member_id</text>\n'
    svg += f'  <text x="390" y="360" class="db-text">Calculate: Production F1, Precision, Recall</text>\n'
    svg += f'  <text x="390" y="385" class="db-text">Alert: If metrics < training by 5%+</text>\n'

    # Monitoring dashboard
    svg += arrow(600, 390, 600, 440)
    svg += box(250, 440, 700, 100, "Unified Monitoring Dashboard\\nFeature Drift • Prediction Distribution • Production Accuracy\\nAccessible to: Data Scientists, ML Engineers, Business Stakeholders")

    # Alert system
    svg += arrow(600, 540, 600, 580)
    svg += box(420, 580, 360, 50, "Automated Alerts → Retraining Triggers", "db-box-feature")

    # Highlight
    svg += label("Real production metrics, not just drift", 150, 50, highlight=True)
    svg += label("APRA compliance ready", 1000, 50, highlight=True)

    svg += generate_svg_footer()
    return svg

def generate_diagram_8():
    """Diagram 8: Automated Retraining & Deployment Job"""
    svg = generate_svg_header(1200, 700)
    svg += title("Automated Retraining: 6 Triggers + Orchestrated Deployment")

    # Monitoring triggers (top)
    svg += box(50, 80, 1100, 180, "", "db-box-feature")
    svg += f'  <text x="600" y="110" class="db-label" text-anchor="middle">6 Retraining Triggers (OR logic)</text>\n'
    svg += f'  <text x="70" y="140" class="db-text">1. Feature Drift: PSI > 0.25</text>\n'
    svg += f'  <text x="70" y="165" class="db-text">2. Prediction Drift: Distribution shift</text>\n'
    svg += f'  <text x="70" y="190" class="db-text">3. Null Spike: >5% increase in nulls</text>\n'
    svg += f'  <text x="70" y="215" class="db-text">4. F1 Drop: Production F1 < training F1 by 5%+</text>\n'
    svg += f'  <text x="70" y="240" class="db-text">5. Data Quality: Schema changes, range violations</text>\n'
    svg += f'  <text x="70" y="265" class="db-text">6. Monthly Fallback: Retrain every 30 days (even if no alerts)</text>\n'

    # Databricks Job orchestration
    svg += arrow(600, 260, 600, 300)
    svg += box(200, 300, 800, 60, "Databricks Job: mlops_churn_pipeline", "db-box-feature")

    # Task workflow
    svg += arrow(600, 360, 600, 400)
    svg += box(50, 400, 1100, 180, "", "db-box")
    svg += f'  <text x="600" y="430" class="db-label" text-anchor="middle">6-Task Orchestrated Workflow</text>\n'
    svg += f'  <text x="70" y="460" class="db-text">Task 1: Feature Engineering → Task 2: Model Training (HPO)</text>\n'
    svg += f'  <text x="70" y="485" class="db-text">Task 3: Model Registration → Task 4: Challenger Validation (7 tests)</text>\n'
    svg += f'  <text x="70" y="510" class="db-text">Task 5: Challenger Approval → Task 6: Batch Inference</text>\n'
    svg += f'  <text x="70" y="540" class="db-highlight">Quality Gates:</text>\n'
    svg += f'  <text x="70" y="565" class="db-text">• All 7 validation tests PASS + F1 improvement → Promote to @Champion</text>\n'

    # Results
    svg += arrow(600, 580, 600, 620)
    svg += box(350, 620, 500, 60, "New @Champion in Production\\nZero-downtime deployment • Rollback ready")

    # Highlight
    svg += label("Fully automated: Trigger → Train → Test → Deploy", 600, 50, highlight=True)

    svg += generate_svg_footer()
    return svg

def generate_diagram_9():
    """Diagram 9: Deployment Job Orchestration"""
    svg = generate_svg_header(1200, 600)
    svg += title("Databricks Job: End-to-End MLOps Pipeline Orchestration")

    # Job header
    svg += box(300, 80, 600, 60, "Databricks Job: mlops_churn_pipeline", "db-box-feature")

    # Task flow
    svg += arrow(600, 140, 600, 180)

    # Tasks in sequence
    tasks = [
        ("Task 1: Feature Engineering", 180),
        ("Task 2: Model Training (HPO)", 260),
        ("Task 3: Model Registration", 340),
        ("Task 4: Challenger Validation", 420),
        ("Task 5: Challenger Approval", 500)
    ]

    for task_name, y_pos in tasks:
        svg += box(350, y_pos, 500, 60, task_name)
        if y_pos < 500:
            svg += arrow(600, y_pos + 60, 600, y_pos + 80)

    # Final arrow to batch inference
    svg += arrow(600, 560, 600, 580)

    # Highlight boxes
    svg += box(100, 220, 200, 180, "", "db-box-feature")
    svg += f'  <text x="200" y="250" class="db-label" text-anchor="middle">Features</text>\n'
    svg += f'  <text x="120" y="280" class="db-text">• Cluster config</text>\n'
    svg += f'  <text x="120" y="305" class="db-text">• Dependencies</text>\n'
    svg += f'  <text x="120" y="330" class="db-text">• Timeouts</text>\n'
    svg += f'  <text x="120" y="355" class="db-text">• Max retries</text>\n'
    svg += f'  <text x="120" y="380" class="db-text">• Email alerts</text>\n'

    svg += box(900, 220, 200, 180, "", "db-box-feature")
    svg += f'  <text x="1000" y="250" class="db-label" text-anchor="middle">Scheduling</text>\n'
    svg += f'  <text x="920" y="280" class="db-text">• Manual trigger</text>\n'
    svg += f'  <text x="920" y="305" class="db-text">• Cron schedule</text>\n'
    svg += f'  <text x="920" y="330" class="db-text">• Drift alerts</text>\n'
    svg += f'  <text x="920" y="355" class="db-text">• API triggered</text>\n'
    svg += f'  <text x="920" y="380" class="db-text">• CI/CD pipeline</text>\n'

    # Highlight
    svg += label("Fully orchestrated: One-click execution", 350, 50, highlight=True)
    svg += label("Production-ready: Email notifications, retries, monitoring", 750, 50, highlight=True)

    svg += generate_svg_footer()
    return svg

def generate_all_diagrams(output_dir="diagrams"):
    """Generate all diagrams and save as HTML files"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    diagrams = {
        "diagram_0_overview.html": generate_diagram_0(),
        "diagram_1_features.html": generate_diagram_1(),
        "diagram_2_training.html": generate_diagram_2(),
        "diagram_3_registration.html": generate_diagram_3(),
        "diagram_4_validation.html": generate_diagram_4(),
        "diagram_5_promotion.html": generate_diagram_5(),
        "diagram_6_serving.html": generate_diagram_6_serving(),
        "diagram_7_monitoring.html": generate_diagram_7(),
        "diagram_8_retraining.html": generate_diagram_8(),
        "diagram_9_deployment.html": generate_diagram_9(),
    }

    for filename, svg_content in diagrams.items():
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MLOps Diagram - {filename}</title>
    <style>
        body {{
            margin: 0;
            padding: 20px;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            background: #f5f5f5;
            font-family: 'Barlow', sans-serif;
        }}
    </style>
</head>
<body>
    {svg_content}
</body>
</html>"""

        filepath = output_path / filename
        with open(filepath, 'w') as f:
            f.write(html_content)

        print(f"Created: {filepath}")

    print(f"\n✓ Generated {len(diagrams)} diagrams in '{output_dir}/' directory")
    print("  Open any .html file in a browser to view")
    print("  Take screenshots or export as PNG for use in notebooks")

if __name__ == "__main__":
    generate_all_diagrams()
