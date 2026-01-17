"""Generic column analysis using statistics, not hardcoded names."""

import pandas as pd
from typing import Dict, List, Set
from dataclasses import dataclass


@dataclass
class ColumnProfile:
    """Statistical profile of a column."""
    name: str
    dtype: str
    n_unique: int
    unique_ratio: float
    missing_ratio: float
    is_numeric: bool
    is_categorical: bool
    is_identifier: bool
    is_high_cardinality: bool
    is_mostly_missing: bool


class ColumnAnalyzer:
    """Analyze columns using generic statistics-based heuristics."""
    
    def __init__(
        self,
        high_cardinality_threshold: float = 0.5,
        mostly_missing_threshold: float = 0.6,
        identifier_patterns: List[str] = None,
        identifier_uniqueness_threshold: float = 0.95
    ):
        """
        Initialize analyzer.
        
        Args:
            high_cardinality_threshold: Unique ratio threshold (e.g., 0.5 = 50% unique)
            mostly_missing_threshold: Missing ratio threshold (e.g., 0.6 = 60% missing)
            identifier_patterns: Column name patterns that suggest identifier
            identifier_uniqueness_threshold: Uniqueness ratio for identifier detection
        """
        self.high_cardinality_threshold = high_cardinality_threshold
        self.mostly_missing_threshold = mostly_missing_threshold
        self.identifier_patterns = identifier_patterns or ['id', 'uuid', 'guid', 'key']
        self.identifier_uniqueness_threshold = identifier_uniqueness_threshold
    
    def analyze_column(self, df: pd.DataFrame, col: str) -> ColumnProfile:
        """
        Analyze a single column using statistics.
        
        Args:
            df: Input DataFrame
            col: Column name
            
        Returns:
            ColumnProfile with statistical characteristics
        """
        n_unique = df[col].nunique()
        n_total = len(df)
        unique_ratio = n_unique / n_total if n_total > 0 else 0
        missing_ratio = df[col].isna().mean()
        
        # Determine type
        is_numeric = pd.api.types.is_numeric_dtype(df[col])
        is_categorical = pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col])
        
        # Detect identifier-like columns (generic, stats-based)
        col_lower = col.lower()
        is_identifier = (
            # Pattern-based detection
            any(pattern in col_lower for pattern in self.identifier_patterns)
            # AND high uniqueness (most rows have unique values)
            and unique_ratio >= self.identifier_uniqueness_threshold
        )
        
        # High cardinality detection (generic)
        is_high_cardinality = (
            is_categorical
            and unique_ratio > self.high_cardinality_threshold
            and n_unique > 50  # Absolute threshold too
        )
        
        # Mostly missing detection (generic)
        is_mostly_missing = missing_ratio > self.mostly_missing_threshold
        
        return ColumnProfile(
            name=col,
            dtype=str(df[col].dtype),
            n_unique=n_unique,
            unique_ratio=unique_ratio,
            missing_ratio=missing_ratio,
            is_numeric=is_numeric,
            is_categorical=is_categorical,
            is_identifier=is_identifier,
            is_high_cardinality=is_high_cardinality,
            is_mostly_missing=is_mostly_missing
        )
    
    def analyze_dataframe(self, df: pd.DataFrame) -> Dict[str, ColumnProfile]:
        """
        Analyze all columns in a DataFrame.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary mapping column name to ColumnProfile
        """
        profiles = {}
        for col in df.columns:
            profiles[col] = self.analyze_column(df, col)
        return profiles
    
    def get_columns_by_type(
        self,
        profiles: Dict[str, ColumnProfile],
        include_identifiers: bool = False,
        include_high_cardinality: bool = True,
        include_mostly_missing: bool = True
    ) -> Dict[str, List[str]]:
        """
        Categorize columns based on their profiles.
        
        Args:
            profiles: Column profiles from analyze_dataframe
            include_identifiers: Whether to include identifier columns
            include_high_cardinality: Whether to include high-cardinality columns
            include_mostly_missing: Whether to include mostly-missing columns
            
        Returns:
            Dictionary with 'numeric', 'categorical', 'exclude' lists
        """
        result = {
            'numeric': [],
            'categorical': [],
            'exclude': []
        }
        
        for name, profile in profiles.items():
            # Exclude identifiers (unless explicitly included)
            if profile.is_identifier and not include_identifiers:
                result['exclude'].append(name)
                continue
            
            # Exclude high-cardinality categoricals (unless explicitly included)
            if profile.is_high_cardinality and not include_high_cardinality:
                result['exclude'].append(name)
                continue
            
            # Exclude mostly-missing columns (unless explicitly included)
            if profile.is_mostly_missing and not include_mostly_missing:
                result['exclude'].append(name)
                continue
            
            # Categorize remaining columns
            if profile.is_numeric:
                result['numeric'].append(name)
            elif profile.is_categorical:
                result['categorical'].append(name)
        
        return result
    
    def get_feature_engineering_suggestions(
        self,
        profiles: Dict[str, ColumnProfile]
    ) -> Dict[str, List[str]]:
        """
        Get suggestions for feature engineering based on column profiles.
        
        Args:
            profiles: Column profiles
            
        Returns:
            Dictionary with suggestions
        """
        suggestions = {
            'create_missing_indicator': [],
            'consider_binning': [],
            'consider_grouping': []
        }
        
        for name, profile in profiles.items():
            # Suggest missing indicators for columns with some (but not too much) missingness
            if 0.05 < profile.missing_ratio < self.mostly_missing_threshold:
                suggestions['create_missing_indicator'].append(name)
            
            # Suggest binning for numeric columns with high cardinality
            if profile.is_numeric and profile.n_unique > 100:
                suggestions['consider_binning'].append(name)
            
            # Suggest grouping for categorical columns approaching high cardinality
            if (profile.is_categorical and 
                20 < profile.n_unique < 50 and
                profile.unique_ratio > 0.1):
                suggestions['consider_grouping'].append(name)
        
        return suggestions


__all__ = ['ColumnAnalyzer', 'ColumnProfile']
