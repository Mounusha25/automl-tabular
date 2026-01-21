# AutoML Tabular

**Explainable AutoML for Tabular Data**

_By Mounusha Ram Metti_

AutoML Tabular is a small, production‑style AutoML engine for tabular datasets.  
It handles **binary classification**, **multiclass classification**, and **regression**, and automatically generates a **human‑readable HTML report** for each run.

The focus is not just on raw metric, but also on **explainability**, **data quality**, and **sane model selection**.

> 🌐 **[Try Live Demo](https://automl-tabular-demo.streamlit.app)** (coming soon) | 📊 **[View Sample Reports](examples/sample_reports/)** | 🔗 **[GitHub](https://github.com/Mounusha25/automl-tabular)**

---

## Demo

### 🌐 Interactive Web Demo

**30-second recruiter-friendly demo** - See AutoML in action:

```bash
# Install Streamlit (one-time)
pip3 install streamlit --user

# Run demo
python3 -m streamlit run demo_streamlit.py
```

**What you'll see:**
- ✨ Modern, professional UI with gradient design
- 🎯 One-click AutoML on Titanic dataset
- ⚡ Real-time progress tracking
- 📊 Model leaderboard with all trials
- 📄 Embedded HTML report preview
- 📥 Download trained model + report

**Perfect for:** Interviews, portfolio showcase, live demonstrations

_Full demo guide: See [STREAMLIT_DEMO.md](STREAMLIT_DEMO.md)_

### 📄 Generated HTML Reports

Each AutoML run produces a comprehensive, professional HTML report with:
- Executive summary & model selection rationale
- Data quality warnings & recommendations
- Interactive leaderboard & model family comparisons
- Feature importance visualizations
- Confusion matrices & performance metrics

**Sample reports:**
- [Titanic (Binary Classification)](examples/sample_reports/titanic_report.html)
- [Adult Income (Binary Classification)](examples/sample_reports/adult_income_report.html)
- [California Housing (Regression)](examples/sample_reports/california_housing_report.html)
- [Wine Quality (Multiclass - 6 classes)](examples/sample_reports/wine_quality_report.html)

---

## Features

### 🔧 Core AutoML Engine

- **Task detection**
  - `Binary Classification`
  - `Multiclass Classification (N classes)`
  - `Regression`
- **Generic label encoding** for all classification tasks (binary + multiclass)
- **Unified model artifact** (saved via `joblib`):
  - `pipeline` – preprocessing + model in a single object
  - `label_encoder` – for mapping encoded classes back to original labels
  - `class_labels` – ordered class names
  - `problem_type`, `metric_name`, and other metadata
- **Hyperparameter search** with [Optuna](https://optuna.org/) across multiple model families:
  - Logistic Regression
  - Random Forest
  - XGBoost
  - LightGBM
- **Robust validation**
  - 80/20 train/validation split (stratified for classification)
  - 3‑fold cross‑validation inside the hyperparameter search
- **Performance optimizations** (~1.4x speedup):
  - Optuna MedianPruner for early trial termination
  - Dynamic CV folds (adjusts by dataset size)
  - Sparse matrix support for high-cardinality categoricals
  - Float32 precision for faster computation
  - See [PERFORMANCE.md](PERFORMANCE.md) for details

### 🧠 Smart Model Selection

Instead of blindly picking the highest metric, AutoML Tabular uses:

- **Primary metric** chosen by task:
  - Binary / Multiclass: `ROC_AUC` or `ACCURACY` (configurable)
  - Regression: `RMSE` or `MAE` (configurable)
- **Tolerance‑based comparison**:
  - Models within a small margin (e.g. 0.5% of the best metric) are treated as **top contenders**
- **Simplicity‑aware tie‑breaking**:
  - If several models are within tolerance, the system can prefer **simpler, more interpretable** models (e.g. logistic regression vs boosted trees)
- **Explicit explanation** in the report:
  - Either: _"chosen because it achieved the highest validation metric"_
  - Or: _"chosen because it is simpler and within tolerance of the best model"_

### 📊 HTML Reporting & UX

Each run produces an HTML report with:

- **Executive Summary**
  - Problem type
  - Recommended model & metric
  - Model selection insight (best vs recommended, tolerance, rationale)
  - Methodology overview (split, metric, CV, search, preprocessing)
- **Dataset Information**
  - # samples, # features, target column
- **Data Quality Warnings**
  - Duplicate rows
  - Severe class imbalance (with note that stratified splitting is applied)
- **Leaderboard**
  - Per‑trial results with metric, time, and "Top Contender" flags
- **Model Family Summary**
  - Best score, mean score, # trials, avg training time per model family
- **Feature Importance**
  - Aggregated **per original feature group** (not noisy one‑hot dummies)
  - Optional note about excluding high‑cardinality identifier‑like columns
- **Visualizations**
  - Target distribution
  - Model performance comparison (per family)
  - Confusion matrix (classification)
  - (For regression: error/metric summaries)
- **Recommendations**
  - Data quality / feature engineering / tuning / monitoring hints
- **Model Export section**
  - Path to saved model artifact
  - Minimal code snippet to load and predict

---

## Validated Datasets

The system has been tested end‑to‑end on several public datasets:

- **Titanic**  
  Binary classification (`Survived`), 891 rows, mixed numeric/categorical.
- **Adult Income (Census)**  
  Binary classification (`income <=50K` vs `>50K`), ~32K rows, categorical‑heavy.
- **California Housing**  
  Regression (house value), ~20K rows, numeric.
- **Wine Quality**  
  Multiclass classification (6 classes of quality), 1,599 rows, numeric, imbalanced, with duplicates.

Each dataset surfaced different edge cases (class imbalance, duplicates, multiclass label encoding, template context bugs), which were fixed **generically** in the pipeline – not with dataset‑specific hacks.

---

## Installation

> Requires **Python 3.10+**

```bash
git clone https://github.com/your-username/automl-tabular.git
cd automl-tabular

# Option A: classic
pip install -r requirements.txt

# Option B: editable install (recommended for development)
pip install -e .
```

---

## Quickstart

### 1️⃣ CLI Usage

Train an AutoML model and generate a report:

```bash
python3 run_automl.py \
  --data examples/titanic.csv \
  --target Survived \
  --output output/titanic_run
```

This will:

- Detect the task as **Binary Classification**
- Run Optuna search across the configured model families
- Save:
  - An HTML report at `output/titanic_run/reports/automl_report_<timestamp>.html`
  - A model artifact at `output/titanic_run/models/<timestamp>_model.joblib`

You can open the report in your browser:

```bash
open output/titanic_run/reports/automl_report_*.html      # macOS
# or
xdg-open output/titanic_run/reports/automl_report_*.html  # Linux
```

#### Override task / metric (optional)

If you want to override auto‑detected task or default metric:

```bash
python3 run_automl.py \
  --data data.csv \
  --target target_col \
  --metric accuracy
```

Supported metrics examples:
- `--metric roc_auc` (binary/multiclass)
- `--metric accuracy` (binary/multiclass)
- `--metric rmse` (regression)
- `--metric mae` (regression)

---

### 2️⃣ Using the Saved Model in Python

```python
import joblib
import pandas as pd

# Load the saved artifacts
artifacts = joblib.load("output/titanic_run/models/20260113_190226_model.joblib")

pipeline = artifacts["pipeline"]
label_encoder = artifacts.get("label_encoder", None)
class_labels = artifacts.get("class_labels", None)

# Load new data
new_data = pd.read_csv("your_data.csv")

# Class predictions
raw_preds = pipeline.predict(new_data)

if label_encoder is not None:
    preds = label_encoder.inverse_transform(raw_preds)
else:
    preds = raw_preds

print(preds[:10])

# Probabilities (for classification)
if hasattr(pipeline, "predict_proba"):
    proba = pipeline.predict_proba(new_data)
    # columns correspond to `class_labels` in order
    print(class_labels)
    print(proba[:3])
```

---

## How It Works

### 1. Problem Type Inference

Given the target column `y`:

- If dtype is `categorical` / `object` → **Classification**
- If numeric:
  - Few unique values (e.g. ≤ 20) → **Classification** (e.g. quality scores)
  - Many unique values → **Regression**

For classification:
- Task string is either:
  - `Binary Classification` (2 unique labels), or
  - `Multiclass Classification (N classes)`

### 2. Target Preprocessing

**For classification:**
- Uses `sklearn.preprocessing.LabelEncoder` to map labels to `0..N-1`
- Stores:
  - `label_encoder`
  - `class_labels = label_encoder.classes_`

**For regression:**
- Uses the numeric target as‑is.

### 3. Preprocessing Pipeline

**Numeric features:**
- Median imputation
- Optional scaling (depending on model)

**Categorical features:**
- Most‑frequent imputation
- One‑hot encoding (`handle_unknown="ignore"`)

A `ColumnTransformer + Pipeline` wraps preprocessing together with the model.

### 4. Model Search (Optuna)

For each enabled `ModelSpec` (Logistic Regression, Random Forest, XGBoost, LightGBM):

1. Sample hyperparameters via Optuna (`n_trials` per model family)
2. Evaluate with 3‑fold CV on the training set
3. Track:
   - Mean CV metric
   - Training time
   - Hyperparameters

Results from all trials are combined into a leaderboard.

### 5. Model Selection

1. Sort models by primary metric (descending if "higher is better").
2. Compute a tolerance region around the best metric (e.g. 0.5%).
3. Mark all models within tolerance as **Top Contenders**.
4. Recommended model:
   - If exactly one model in tolerance → that one.
   - If multiple:
     - Choose the **simplest** (based on a small numeric simplicity score), e.g.:
       - Logistic Regression < Random Forest < XGBoost/LightGBM

### 6. Reporting

The orchestrator builds a context dictionary:

- `summary` – task, primary metric, dataset info
- `selection` – tolerance, best/recommended metrics, reason text
- `leaderboard` – per trial
- `families` – aggregated stats by model family
- `feature_importance` – per original feature group
- `plots` – paths to generated PNGs (target dist, confusion matrix, etc.)

This context is rendered using a **Jinja2 HTML template** to produce `report.html`.

---

## Project Structure

```
.
├── README.md
├── requirements.txt
├── run_automl.py              # Main CLI entry point
├── src/
│   └── automl_tabular/
│       ├── __init__.py
│       ├── orchestrator.py    # Main AutoML workflow
│       ├── config/
│       ├── data/
│       ├── preprocessing/
│       ├── models/
│       ├── evaluation/
│       ├── explainability/
│       ├── reporting/
│       │   └── templates/
│       │       └── report_template.html
│       └── interfaces/
│           └── cli.py
├── examples/
│   ├── titanic.csv
│   ├── adult_income.csv
│   ├── california_housing.csv
│   ├── wine_quality.csv
│   └── sample_reports/       # optional static HTML reports
├── output/                    # generated at runtime (gitignored)
└── tests/
    ├── test_loader.py
    ├── test_preprocessing.py
    ├── test_metrics.py
    └── test_end_to_end.py
```

---

## Development & Testing

Run tests:

```bash
pytest
```

Recommended end‑to‑end tests (examples):

- Run AutoML on each example dataset with a small search budget.
- Assert that:
  - A report is generated.
  - A model artifact is saved.
  - No exceptions occur.

---

## Roadmap / Ideas

Future improvements that could be fun:

- Class‑weight handling / resampling for highly imbalanced datasets.
- More metrics (macro‑F1, recall@k, etc.) in the reports.
- Partial dependence / SHAP plots for the recommended model.
- Simple REST API wrapper around the orchestrator.
- Dockerized version for easy deployment.

---

## License

This project is licensed under the MIT License – see the LICENSE file for details.

---

## Author

**Mounusha Ram Metti**

---

**Made with ❤️ for the ML community**
