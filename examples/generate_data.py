"""Generate example datasets for testing AutoML."""

import pandas as pd
import numpy as np
from pathlib import Path


def generate_classification_dataset(n_samples=1000, save_path="examples/example_classification.csv"):
    """Generate a synthetic classification dataset."""
    np.random.seed(42)
    
    # Features
    age = np.random.randint(18, 80, n_samples)
    income = np.random.normal(50000, 20000, n_samples)
    credit_score = np.random.randint(300, 850, n_samples)
    years_employed = np.random.randint(0, 40, n_samples).astype(float)  # Convert to float to allow NaN
    debt_to_income = np.random.uniform(0, 1, n_samples)
    
    # Categorical features
    education = np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples, p=[0.3, 0.4, 0.2, 0.1])
    employment_type = np.random.choice(['Full-time', 'Part-time', 'Self-employed', 'Unemployed'], n_samples, p=[0.6, 0.2, 0.15, 0.05])
    marital_status = np.random.choice(['Single', 'Married', 'Divorced'], n_samples, p=[0.4, 0.45, 0.15])
    
    # Target: loan approval (0 = rejected, 1 = approved)
    # Create a pattern: higher income, credit score, and education increase approval chance
    approval_prob = (
        0.3 +
        0.15 * (income > 60000) +
        0.2 * (credit_score > 700) +
        0.15 * np.isin(education, ['Master', 'PhD']) +
        0.1 * (years_employed > 5) +
        0.1 * (debt_to_income < 0.4)
    )
    
    loan_approved = (np.random.random(n_samples) < approval_prob).astype(int)
    
    # Add some missing values
    income[np.random.choice(n_samples, 50, replace=False)] = np.nan
    years_employed[np.random.choice(n_samples, 30, replace=False)] = np.nan
    
    # Create DataFrame
    df = pd.DataFrame({
        'age': age,
        'income': income,
        'credit_score': credit_score,
        'years_employed': years_employed,
        'debt_to_income_ratio': debt_to_income,
        'education': education,
        'employment_type': employment_type,
        'marital_status': marital_status,
        'loan_approved': loan_approved
    })
    
    # Save
    Path(save_path).parent.mkdir(exist_ok=True, parents=True)
    df.to_csv(save_path, index=False)
    print(f"✅ Classification dataset saved to: {save_path}")
    print(f"   Shape: {df.shape}")
    print(f"   Target distribution: {df['loan_approved'].value_counts().to_dict()}")
    
    return df


def generate_regression_dataset(n_samples=1000, save_path="examples/example_regression.csv"):
    """Generate a synthetic regression dataset."""
    np.random.seed(42)
    
    # Features for house price prediction
    square_feet = np.random.randint(800, 5000, n_samples)
    bedrooms = np.random.randint(1, 6, n_samples)
    bathrooms = np.random.randint(1, 5, n_samples)
    year_built = np.random.randint(1950, 2024, n_samples).astype(float)  # Convert to float to allow NaN
    lot_size = np.random.randint(2000, 20000, n_samples).astype(float)  # Convert to float to allow NaN
    garage_spaces = np.random.randint(0, 4, n_samples)
    
    # Categorical features
    neighborhood = np.random.choice(['Urban', 'Suburban', 'Rural'], n_samples, p=[0.4, 0.45, 0.15])
    condition = np.random.choice(['Excellent', 'Good', 'Fair', 'Poor'], n_samples, p=[0.2, 0.4, 0.3, 0.1])
    
    # Target: house price
    # Create a realistic pattern
    base_price = 100000
    price = (
        base_price +
        square_feet * 150 +
        bedrooms * 20000 +
        bathrooms * 15000 +
        (2024 - year_built) * (-500) +  # Newer homes worth more
        lot_size * 5 +
        garage_spaces * 10000 +
        (neighborhood == 'Urban') * 50000 +
        (neighborhood == 'Suburban') * 30000 +
        (condition == 'Excellent') * 40000 +
        (condition == 'Good') * 20000 +
        np.random.normal(0, 30000, n_samples)  # Random noise
    )
    
    # Ensure positive prices
    price = np.maximum(price, 50000)
    
    # Add some missing values
    lot_size[np.random.choice(n_samples, 40, replace=False)] = np.nan
    year_built[np.random.choice(n_samples, 25, replace=False)] = np.nan
    
    # Create DataFrame
    df = pd.DataFrame({
        'square_feet': square_feet,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'year_built': year_built,
        'lot_size': lot_size,
        'garage_spaces': garage_spaces,
        'neighborhood': neighborhood,
        'condition': condition,
        'price': price
    })
    
    # Save
    Path(save_path).parent.mkdir(exist_ok=True, parents=True)
    df.to_csv(save_path, index=False)
    print(f"✅ Regression dataset saved to: {save_path}")
    print(f"   Shape: {df.shape}")
    print(f"   Price range: ${df['price'].min():.0f} - ${df['price'].max():.0f}")
    print(f"   Mean price: ${df['price'].mean():.0f}")
    
    return df


if __name__ == "__main__":
    print("Generating example datasets...\n")
    generate_classification_dataset()
    print()
    generate_regression_dataset()
    print("\n✅ All datasets generated!")
