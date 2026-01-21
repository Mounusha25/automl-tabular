#!/usr/bin/env python3
"""
Demonstration of the smart preprocessing system.
Shows how data profiling and intelligent strategy selection works.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import pandas as pd
from automl_tabular.preprocessing import profile_dataset, build_preprocessing_plan

print("="*70)
print("SMART PREPROCESSING DEMONSTRATION")
print("="*70)

# Load Titanic dataset
df = pd.read_csv('examples/titanic.csv')
target = 'Survived'

print(f"\nDataset: {len(df)} rows, {len(df.columns)} columns")
print(f"Target: {target}")

# Step 1: Profile the dataset
print("\n" + "="*70)
print("STEP 1: DATA PROFILING")
print("="*70)

profile = profile_dataset(df, target)

print(f"\nProfiled {len(profile.features)} features:")
for name, feat in list(profile.features.items())[:5]:  # Show first 5
    print(f"\n{name}:")
    print(f"  Type: {feat.dtype}")
    print(f"  Missing: {feat.missing_ratio*100:.1f}%")
    if feat.dtype == "numeric":
        print(f"  Mean: {feat.mean:.2f}, Median: {feat.median:.2f}")
        if feat.skewness:
            print(f"  Skewness: {feat.skewness:.2f}")
        if feat.outlier_ratio:
            print(f"  Outliers: {feat.outlier_ratio*100:.1f}%")

# Step 2: Build preprocessing plan
print("\n" + "="*70)
print("STEP 2: INTELLIGENT STRATEGY SELECTION")
print("="*70)

plan = build_preprocessing_plan(profile, enable_smart_strategies=True)

print(f"\nNumeric features: {len(plan.numeric_features)}")
print(f"Categorical features: {len(plan.categorical_features)}")
print(f"Dropped features: {len(plan.dropped_features)}")

if plan.dropped_features:
    print(f"\n🗑️  Features to drop: {', '.join(plan.dropped_features)}")

# Step 3: Show preprocessing strategies
print("\n" + "="*70)
print("STEP 3: PREPROCESSING STRATEGIES (Sample)")
print("="*70)

# Show strategies for a few interesting features
interesting_features = ['Age', 'Cabin', 'Embarked', 'Fare']
for feat_name in interesting_features:
    if feat_name in plan.features:
        feat_plan = plan.features[feat_name]
        print(f"\n📋 {feat_name}:")
        print(f"   Imputation: {feat_plan.imputation_strategy}")
        if feat_plan.add_missing_indicator:
            print(f"   Missing indicator: YES")
        if feat_plan.outlier_handling != "none":
            print(f"   Outlier handling: {feat_plan.outlier_handling}")
        print(f"   Scaling: {feat_plan.scaling}")
        print(f"   Explanation: {feat_plan.explanation}")

# Step 4: Show full explanations
print("\n" + "="*70)
print("STEP 4: ALL PREPROCESSING EXPLANATIONS")
print("="*70)

print("\nThese would appear in the HTML report:\n")
for i, (name, feat_plan) in enumerate(plan.features.items(), 1):
    if feat_plan.imputation_strategy != "drop":
        print(f"{i}. {feat_plan.explanation}")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("✓ Data profiled with statistical analysis")
print("✓ Strategies chosen based on missingness, skew, outliers")
print("✓ Explanations generated for transparency")
print("✓ Ready to build sklearn pipeline from this plan")
