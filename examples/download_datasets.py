"""Download real-world datasets for testing AutoML generality."""

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from pathlib import Path

def download_california_housing():
    """Download California Housing dataset (regression)."""
    print("📥 Downloading California Housing dataset...")
    
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame
    
    # Save to CSV
    output_path = Path(__file__).parent / 'california_housing.csv'
    df.to_csv(output_path, index=False)
    
    print(f"✅ Saved to: {output_path}")
    print(f"   Shape: {df.shape}")
    print(f"   Target: MedHouseVal (median house value)")
    print(f"   Type: Regression")
    print(f"   Features: {', '.join(df.columns[:-1])}\n")
    
    return output_path


def download_adult_income():
    """Download Adult Income dataset (binary classification)."""
    print("📥 Downloading Adult Income dataset...")
    
    # UCI Adult Income dataset
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    
    columns = [
        'age', 'workclass', 'fnlwgt', 'education', 'education-num',
        'marital-status', 'occupation', 'relationship', 'race', 'sex',
        'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
    ]
    
    try:
        df = pd.read_csv(url, names=columns, skipinitialspace=True)
        
        # Clean up
        df['income'] = df['income'].map({'<=50K': 0, '>50K': 1})
        
        # Replace '?' with NaN
        df = df.replace('?', np.nan)
        
        # Save to CSV
        output_path = Path(__file__).parent / 'adult_income.csv'
        df.to_csv(output_path, index=False)
        
        print(f"✅ Saved to: {output_path}")
        print(f"   Shape: {df.shape}")
        print(f"   Target: income (0: <=50K, 1: >50K)")
        print(f"   Type: Binary Classification")
        print(f"   Features: age, workclass, education, occupation, sex, etc.\n")
        
        return output_path
        
    except Exception as e:
        print(f"⚠️  Could not download Adult Income dataset: {e}")
        print("   Continuing with other datasets...\n")
        return None


def download_wine_quality():
    """Download Wine Quality dataset (regression/classification)."""
    print("📥 Downloading Wine Quality dataset...")
    
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    
    try:
        df = pd.read_csv(url, sep=';')
        
        # Save to CSV
        output_path = Path(__file__).parent / 'wine_quality.csv'
        df.to_csv(output_path, index=False)
        
        print(f"✅ Saved to: {output_path}")
        print(f"   Shape: {df.shape}")
        print(f"   Target: quality (score 0-10)")
        print(f"   Type: Regression (or multiclass if binned)")
        print(f"   Features: acidity, sugar, chlorides, sulfur, alcohol, etc.\n")
        
        return output_path
        
    except Exception as e:
        print(f"⚠️  Could not download Wine Quality dataset: {e}")
        print("   Continuing with other datasets...\n")
        return None


if __name__ == "__main__":
    print("="*80)
    print("DOWNLOADING REAL-WORLD DATASETS")
    print("="*80)
    print()
    
    datasets = []
    
    # Download datasets
    housing_path = download_california_housing()
    if housing_path:
        datasets.append(('california_housing.csv', 'MedHouseVal', 'rmse'))
    
    adult_path = download_adult_income()
    if adult_path:
        datasets.append(('adult_income.csv', 'income', 'roc_auc'))
    
    wine_path = download_wine_quality()
    if wine_path:
        datasets.append(('wine_quality.csv', 'quality', 'rmse'))
    
    print("="*80)
    print("READY TO TEST!")
    print("="*80)
    print("\nTest the AutoML system with these commands:\n")
    
    for dataset, target, metric in datasets:
        dataset_name = dataset.replace('.csv', '')
        print(f"# {dataset_name.replace('_', ' ').title()}")
        print(f"python3 run_automl.py \\")
        print(f"  --data examples/{dataset} \\")
        print(f"  --target {target} \\")
        print(f"  --output output/{dataset_name} \\")
        print(f"  --metric {metric}")
        print()
    
    print("This will prove the system works on:")
    print("✅ Classification (Adult Income, Titanic)")
    print("✅ Regression (California Housing, Wine Quality)")
    print("✅ Different feature types (numeric, categorical, mixed)")
    print("✅ Different data sizes (150 rows to 30K+ rows)")
