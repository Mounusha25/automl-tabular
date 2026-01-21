#!/usr/bin/env python3
"""
Demonstration: How Your AutoML System Handles Missing Data Imputation
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

print('='*70)
print('REAL EXAMPLE: HOW YOUR AUTOML HANDLES TITANIC MISSING DATA')
print('='*70)

# Load Titanic data
df = pd.read_csv('examples/titanic.csv')

# Simulate what your system does
target = 'Survived'
X = df.drop(columns=[target])
y = df[target]

# Define feature types (your system auto-detects these)
numeric_features = ['Age', 'Fare', 'SibSp', 'Parch']
categorical_features = ['Sex', 'Pclass', 'Embarked', 'Cabin']

print(f'\nOriginal data shape: {X.shape}')
print(f'\nMissing values BEFORE imputation:')
print(X[numeric_features + categorical_features].isnull().sum())

# Build the EXACT pipeline your AutoML uses
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='drop',
    sparse_threshold=0
)

# Split train/test
train_size = int(0.8 * len(X))
X_train = X.iloc[:train_size]
X_test = X.iloc[train_size:]

print(f'\n' + '='*70)
print('STEP 1: FIT - Learn imputation values from training data')
print('='*70)

# Fit on training data (learns median/mode)
preprocessor.fit(X_train)

# Extract the learned values
num_imputer = preprocessor.named_transformers_['num'].named_steps['imputer']
cat_imputer = preprocessor.named_transformers_['cat'].named_steps['imputer']

print('\nLearned values from TRAINING data:')
print('  Numeric imputation values (medians):')
for feat, val in zip(numeric_features, num_imputer.statistics_):
    print(f'    {feat}: {val:.2f}')

print('\n  Categorical imputation values (modes):')
for feat, val in zip(categorical_features, cat_imputer.statistics_):
    print(f'    {feat}: "{val}"')

print(f'\n' + '='*70)
print('STEP 2: TRANSFORM - Apply learned values to fill missing data')
print('='*70)

# Transform training data
X_train_transformed = preprocessor.transform(X_train)
print(f'\nTraining data transformed:')
print(f'  Input shape: {X_train.shape} (with missing values)')
print(f'  Output shape: {X_train_transformed.shape} (NO missing values)')
print(f'  Missing values after transform: {np.isnan(X_train_transformed).sum()}')

# Transform test data (uses SAME learned values from training)
X_test_transformed = preprocessor.transform(X_test)
print(f'\nTest data transformed:')
print(f'  Input shape: {X_test.shape} (with missing values)')
print(f'  Output shape: {X_test_transformed.shape} (NO missing values)')
print(f'  Missing values after transform: {np.isnan(X_test_transformed).sum()}')

print(f'\n' + '='*70)
print('STEP 3: HOW IT WORKS IN PRODUCTION')
print('='*70)

# Simulate new data with missing values
print('\nSimulate new passenger data with missing Age:')
new_passenger = pd.DataFrame({
    'Age': [np.nan],  # Missing!
    'Fare': [50.0],
    'SibSp': [1],
    'Parch': [0],
    'Sex': ['male'],
    'Pclass': [1],
    'Embarked': ['S'],
    'Cabin': [np.nan]  # Also missing!
})

print('  New passenger Age: NaN (missing)')
print('  New passenger Cabin: NaN (missing)')

# Transform with fitted pipeline (uses learned values)
new_transformed = preprocessor.transform(new_passenger)
print(f'\n  After transformation: Complete data, no NaN!')
print(f'  Age filled with: {num_imputer.statistics_[0]:.2f} (median from training)')
print(f'  Cabin filled with: "{cat_imputer.statistics_[3]}" (mode from training)')

print(f'\n' + '='*70)
print('KEY INSIGHTS')
print('='*70)
print('✓ Imputer LEARNS from training data only (prevents leakage)')
print('✓ Same learned values used for test AND new data')
print('✓ Pipeline saved with model - imputation automatic in production')
print('✓ No manual intervention needed - fully automated')
