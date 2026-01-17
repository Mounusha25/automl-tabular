"""Data validation utilities."""

import pandas as pd
from typing import List, Tuple


class DataValidator:
    """Validates data quality and suitability for AutoML."""
    
    def __init__(self, min_rows: int = 50, max_missing_ratio: float = 0.95):
        """
        Initialize validator with thresholds.
        
        Args:
            min_rows: Minimum number of rows required
            max_missing_ratio: Maximum ratio of missing values allowed per column
        """
        self.min_rows = min_rows
        self.max_missing_ratio = max_missing_ratio
        self.warnings = []
        self.errors = []
    
    def validate(self, df: pd.DataFrame, target_column: str) -> Tuple[bool, List[str], List[str]]:
        """
        Validate the dataset.
        
        Args:
            df: Input DataFrame
            target_column: Name of target column
            
        Returns:
            Tuple of (is_valid, errors, warnings)
        """
        self.warnings = []
        self.errors = []
        
        # Check if target column exists
        if target_column not in df.columns:
            self.errors.append(f"Target column '{target_column}' not found in dataset")
            return False, self.errors, self.warnings
        
        # Check minimum rows
        if len(df) < self.min_rows:
            self.errors.append(
                f"Dataset has only {len(df)} rows, minimum {self.min_rows} required"
            )
        
        # Check for empty DataFrame
        if df.empty:
            self.errors.append("Dataset is empty")
            return False, self.errors, self.warnings
        
        # Check for constant columns
        constant_cols = self._check_constant_columns(df, target_column)
        if constant_cols:
            self.warnings.append(
                f"Columns with constant values detected (will be removed): {constant_cols}"
            )
        
        # Check for high missing value ratio
        high_missing_cols = self._check_missing_values(df, target_column)
        if high_missing_cols:
            self.warnings.append(
                f"Columns with >{self.max_missing_ratio*100}% missing values: {high_missing_cols}"
            )
        
        # Check target column for nulls
        if df[target_column].isnull().any():
            self.errors.append(f"Target column '{target_column}' contains missing values")
        
        # Check if target has at least 2 classes/values
        if df[target_column].nunique() < 2:
            self.errors.append(
                f"Target column has only {df[target_column].nunique()} unique value(s). "
                "Need at least 2 for meaningful predictions."
            )
        
        # Check for duplicate rows
        n_duplicates = df.duplicated().sum()
        if n_duplicates > 0:
            self.warnings.append(
                f"Dataset contains {n_duplicates} duplicate rows ({n_duplicates/len(df)*100:.1f}%)"
            )
        
        # Check class imbalance for classification
        if df[target_column].dtype == 'object' or df[target_column].nunique() < 20:
            self._check_class_imbalance(df[target_column])
        
        is_valid = len(self.errors) == 0
        return is_valid, self.errors, self.warnings
    
    def _check_constant_columns(self, df: pd.DataFrame, target_column: str) -> List[str]:
        """Identify columns with constant values."""
        constant_cols = []
        for col in df.columns:
            if col != target_column and df[col].nunique() <= 1:
                constant_cols.append(col)
        return constant_cols
    
    def _check_missing_values(self, df: pd.DataFrame, target_column: str) -> List[str]:
        """Identify columns with high missing value ratio."""
        high_missing = []
        for col in df.columns:
            if col != target_column:
                missing_ratio = df[col].isnull().sum() / len(df)
                if missing_ratio > self.max_missing_ratio:
                    high_missing.append(f"{col} ({missing_ratio*100:.1f}%)")
        return high_missing
    
    def _check_class_imbalance(self, target_series: pd.Series) -> None:
        """Check for severe class imbalance."""
        value_counts = target_series.value_counts()
        if len(value_counts) > 1:
            imbalance_ratio = value_counts.max() / value_counts.min()
            if imbalance_ratio > 10:
                self.warnings.append(
                    f"Severe class imbalance detected (ratio: {imbalance_ratio:.1f}:1). "
                    "Stratified splitting is applied; consider using class weights or resampling "
                    "if minority classes are important."
                )


__all__ = ["DataValidator"]
