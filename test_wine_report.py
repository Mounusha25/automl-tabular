#!/usr/bin/env python3
"""
Quick test script to generate a professional HTML report for Wine Quality dataset
"""
import sys
sys.path.insert(0, '/Users/mounusha/Downloads/Projects/myautoml/src')

from automl_tabular.core import AutoMLTabular

# Run AutoML on Wine Quality
automl = AutoMLTabular(
    max_trials=15,
    n_cv_folds=3
)

result = automl.run(
    file_path='examples/wine_quality.csv',
    target_column='quality',
    problem_type='classification'
)

print(f"\n✅ AutoML complete!")
print(f"📊 Best Model: {result.best_model_name}")
print(f"🎯 Validation {result.metric_name.upper()}: {result.metric_value:.4f}")
print(f"📁 Report: {result.report_path}")
print(f"\n🌐 Open the report in your browser to see the professional design!")
