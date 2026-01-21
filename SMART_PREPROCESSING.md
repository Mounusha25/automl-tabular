# Smart Preprocessing System

## Overview

The smart preprocessing system adds **intelligent, data-driven preprocessing** to your AutoML pipeline. Instead of using hardcoded strategies (always median/mode), it:

1. **Profiles** your dataset to understand distributions, outliers, and missing patterns
2. **Selects** optimal preprocessing strategies per feature based on the profile
3. **Explains** every decision in the HTML report for full transparency

## ✅ Backward Compatibility

**Your existing system still works exactly as before!**

- Default behavior: `enable_smart_strategies: false` (uses simple median/mode)
- No breaking changes to existing code
- All existing pipelines continue to function

## 🚀 How to Enable

### Option 1: Config File

Edit `src/automl_tabular/config/default_config.yaml`:

```yaml
preprocessing:
  enable_smart_strategies: true  # Enable smart preprocessing
```

### Option 2: Runtime Override

```python
from automl_tabular import run_automl_job

config = {
    'preprocessing': {
        'enable_smart_strategies': True
    }
}

result = run_automl_job(
    data='your_data.csv',
    target_column='target',
    config=config
)
```

## 📊 What It Does

### 1. Data Profiling

For each feature, computes:

**Numeric Features:**
- Mean, median, std
- Min, max, quantiles (Q1, Q3)
- Skewness
- Outlier ratio (via IQR method)
- Missing ratio

**Categorical Features:**
- Unique values, cardinality
- Missing ratio
- Identifier detection (high uniqueness + 'id' in name)

### 2. Intelligent Strategy Selection

#### Numeric Features

**Missingness Handling:**
| Missing % | Strategy | Explanation |
|-----------|----------|-------------|
| 0% | None | No imputation needed |
| 0-5% | Mean (if symmetric) or Median (if skewed) | Low missingness → distribution-aware |
| 5-40% | Median + missing indicator | Moderate → robust + flag |
| 40-95% | Median + missing indicator | High → warn user |
| >95% | Drop feature | Too much missing |

**Outlier Handling:**
| Outlier % | Strategy | Explanation |
|-----------|----------|-------------|
| <5% | None | Few outliers, keep them |
| 5-20% | Clip to IQR bounds | Moderate outliers → clip |
| >20% | Clip to IQR bounds + warning | Heavy outliers → clip + note |

**Scaling:**
| Condition | Strategy | Reason |
|-----------|----------|--------|
| Low skew (<1) & few outliers | StandardScaler | Normal distribution |
| High skew or many outliers | RobustScaler | Robust to outliers |

#### Categorical Features

**Missingness Handling:**
| Missing % | Strategy | Explanation |
|-----------|----------|-------------|
| 0% | None | No imputation needed |
| 0-10% | Most frequent | Low → use mode |
| 10-95% | Constant ("__MISSING__") | Treat as separate category |
| >95% | Drop | Too much missing |

**High Cardinality:**
- Detects high cardinality (>50 unique or >50% unique)
- Flags identifier-like features (>90% unique + name contains 'id')
- Recommends dropping identifiers

### 3. Explanations in Report

Every decision is documented:

```
Age: Moderate missingness (19.9%) - using median + missing indicator. 
     Using standard scaling.

Fare: No missing values. 13.0% outliers detected (IQR method) - 
      clipping to [Q1-1.5×IQR, Q3+1.5×IQR]. Using robust scaling.

Cabin: High cardinality (147 unique values). Missingness (77.1%) 
       encoded as separate '__MISSING__' category.
```

## 📁 New Modules

### `preprocessing/profile.py`
```python
from automl_tabular.preprocessing import profile_dataset

profile = profile_dataset(df, target_column='target')

# Access feature stats
age_profile = profile.features['Age']
print(age_profile.mean, age_profile.skewness, age_profile.outlier_ratio)
```

### `preprocessing/strategy.py`
```python
from automl_tabular.preprocessing import build_preprocessing_plan

plan = build_preprocessing_plan(profile, enable_smart_strategies=True)

# Access preprocessing strategies
age_plan = plan.features['Age']
print(age_plan.imputation_strategy)  # 'median'
print(age_plan.add_missing_indicator)  # True
print(age_plan.explanation)  # Human-readable
```

## 🧪 Testing

### Quick Test (Doesn't break existing functionality)
```bash
python3 demo_smart_preprocessing.py
```

### Full Integration Test
```bash
python3 test_smart_preprocessing.py
```

This runs AutoML twice:
1. With smart preprocessing OFF (default)
2. With smart preprocessing ON

Both should complete successfully.

## 📈 Configuration Options

```yaml
preprocessing:
  # Enable/disable smart strategies
  enable_smart_strategies: false
  
  # Thresholds for smart decisions
  high_missing_threshold: 0.95  # Drop if >95% missing
  drop_high_missing_features: false  # Actually drop vs warn
  outlier_iqr_multiplier: 1.5  # IQR multiplier for outliers
  
  # Fallback strategies (used when smart=false)
  numeric_impute_strategy: "median"
  categorical_impute_strategy: "most_frequent"
  scaling: "standard"
  encoding: "onehot"
  handle_unknown: "ignore"
```

## 🎯 Example Output

### Before (Simple)
```
Step 3: Preprocessing data...
   Features after cleaning: 10
   Numeric features: 5
   Categorical features: 5
```

### After (Smart)
```
Step 3: Preprocessing data...
   📊 Profiling dataset for intelligent preprocessing...
   🗑️  Smart preprocessing dropping: PassengerId
   Features after cleaning: 10
   Numeric features: 5
   Categorical features: 5
```

Report includes detailed section:
```
🧪 Preprocessing Decisions

The following per-feature preprocessing strategies were applied, 
based on dataset-specific distributions:

• Age: Moderate missingness (19.9%) - using median + missing 
  indicator. Using standard scaling.
  
• Fare: No missing values. 13.0% outliers detected (IQR method) - 
  clipping to [Q1-1.5×IQR, Q3+1.5×IQR]. Using robust scaling.
  
• Cabin: High cardinality (147 unique values). Missingness (77.1%) 
  encoded as separate '__MISSING__' category.
```

## 🔧 Architecture

```
Dataset
   ↓
[profile_dataset()] → FeatureProfile per feature
   ↓
[build_preprocessing_plan()] → FeaturePreprocessingPlan per feature
   ↓
[TabularPreprocessor] → sklearn Pipeline
   ↓
Fitted Pipeline (saved with model)
```

## 💡 Why This is Better

### Before
- Always median for numeric
- Always mode for categorical
- No consideration of outliers
- No explanation of choices

### After
- Chooses strategy based on:
  - Missingness level
  - Distribution shape (skewness)
  - Outlier presence
  - Cardinality
- Clips outliers when needed
- Uses robust scaling for skewed data
- **Explains every decision**
- **Portfolio-worthy sophistication**

## 🚨 Important Notes

1. **Validation Handling is Correct**
   - Pipeline fits on training data only
   - Same fitted pipeline transforms validation/test
   - No data leakage

2. **Backward Compatible**
   - Default: `enable_smart_strategies: false`
   - Existing code unchanged
   - Can enable per-run via config

3. **Explainable**
   - Every strategy has a human-readable explanation
   - Shows up in HTML report
   - Great for interviews/presentations

## 📚 For Interviews

**What you can say:**

> "I built a profiling → planning → pipeline architecture that analyzes each feature's statistical properties—missingness, skewness, outliers—and automatically selects appropriate imputation and scaling strategies. The system generates explanations for every decision, making the preprocessing fully transparent and auditable."

**Technical depth:**

- Data profiling with scipy.stats
- Heuristic-based strategy selection
- IQR-based outlier detection
- Skewness-aware imputation
- Missing indicator addition for moderate missingness
- Per-feature explanations in report

## 🔍 Next Steps

To extend this further, you could add:

1. **Rare category grouping** for high-cardinality categoricals
2. **Log transforms** for heavily skewed numeric features
3. **Correlation-based feature dropping**
4. **Boxplot visualizations** in report for outlier diagnostics
5. **Per-feature pipeline** (different strategies per column)

But start simple—what's here is already production-quality!
