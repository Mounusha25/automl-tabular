"""Data loading and validation utilities."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class DataLoader:
    """Handles loading and initial processing of tabular data."""
    
    def __init__(self):
        self.df = None
        self.schema = None
    
    def load_csv(self, path: str) -> pd.DataFrame:
        """
        Load a CSV file into a pandas DataFrame.
        
        Args:
            path: Path to the CSV file
            
        Returns:
            Loaded DataFrame
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is empty or invalid
        """
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        try:
            df = pd.read_csv(path)
        except Exception as e:
            raise ValueError(f"Error reading CSV file: {str(e)}")
        
        if df.empty:
            raise ValueError("CSV file is empty")
        
        self.df = df
        return df
    
    def infer_column_types(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Infer numeric and categorical columns.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with 'numeric' and 'categorical' column lists
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Check for numeric columns that should be treated as categorical
        # (e.g., few unique values)
        for col in numeric_cols[:]:
            if df[col].nunique() <= 10:  # Heuristic: <=10 unique values
                categorical_cols.append(col)
                numeric_cols.remove(col)
        
        schema = {
            'numeric': numeric_cols,
            'categorical': categorical_cols
        }
        
        self.schema = schema
        return schema
    
    def suggest_target_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Suggest potential target columns based on heuristics.
        
        Args:
            df: Input DataFrame
            
        Returns:
            List of suggested column names
        """
        suggestions = []
        
        # Common target column names
        common_names = ['target', 'label', 'class', 'y', 'outcome', 'prediction']
        
        for col in df.columns:
            col_lower = col.lower()
            if any(name in col_lower for name in common_names):
                suggestions.append(col)
        
        # If no common names found, suggest last column
        if not suggestions:
            suggestions.append(df.columns[-1])
        
        return suggestions
    
    def infer_problem_type(self, target_series: pd.Series) -> str:
        """
        Infer whether this is a classification or regression problem.
        
        Args:
            target_series: Target column
            
        Returns:
            'classification' or 'regression'
        """
        # If target is numeric
        if pd.api.types.is_numeric_dtype(target_series):
            n_unique = target_series.nunique()
            
            # Heuristic: if less than 20 unique values, treat as classification
            if n_unique < 20:
                return 'classification'
            else:
                return 'regression'
        else:
            # Non-numeric targets are classification
            return 'classification'
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict:
        """
        Get summary statistics of the dataset.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'n_rows': len(df),
            'n_columns': len(df.columns),
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'missing_percentage': (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
        }
        
        return summary


__all__ = ["DataLoader"]
