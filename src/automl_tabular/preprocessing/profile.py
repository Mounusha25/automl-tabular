"""
Data profiling module for intelligent preprocessing strategy selection.

Analyzes dataset characteristics to inform preprocessing decisions.
"""

from dataclasses import dataclass
from typing import Dict, Optional
import hashlib
import pandas as pd
import numpy as np
from scipy import stats


# Simple cache for profiling results (keyed by fingerprint)
_PROFILE_CACHE = {}


@dataclass
class FeatureProfile:
    """Statistical profile of a single feature."""
    
    name: str
    dtype: str  # "numeric" | "categorical" | "datetime" | "constant"
    n: int
    n_missing: int
    missing_ratio: float
    n_unique: int
    unique_ratio: float
    
    # Numeric-specific stats
    mean: Optional[float] = None
    std: Optional[float] = None
    median: Optional[float] = None
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    q1: Optional[float] = None
    q3: Optional[float] = None
    skewness: Optional[float] = None
    outlier_ratio: Optional[float] = None
    
    # Flags
    is_constant: bool = False
    is_high_cardinality: bool = False
    is_identifier_like: bool = False


@dataclass
class ProfileResult:
    """Complete dataset profile."""
    
    features: Dict[str, FeatureProfile]
    n_rows: int
    n_features: int
    target_column: str


def profile_feature(series: pd.Series, name: str) -> FeatureProfile:
    """
    Profile a single feature.
    
    Args:
        series: Feature data
        name: Feature name
        
    Returns:
        FeatureProfile object
    """
    n = len(series)
    n_missing = series.isnull().sum()
    missing_ratio = n_missing / n if n > 0 else 0.0
    n_unique = series.nunique()
    unique_ratio = n_unique / n if n > 0 else 0.0
    
    # Determine dtype
    if n_unique <= 1:
        dtype = "constant"
        is_constant = True
    elif pd.api.types.is_numeric_dtype(series):
        dtype = "numeric"
        is_constant = False
    elif pd.api.types.is_datetime64_any_dtype(series):
        dtype = "datetime"
        is_constant = False
    else:
        dtype = "categorical"
        is_constant = False
    
    # Check if identifier-like
    is_identifier_like = False
    if unique_ratio > 0.9:
        name_lower = name.lower()
        if any(keyword in name_lower for keyword in ['id', 'uuid', 'guid', 'key', 'index']):
            is_identifier_like = True
    
    # Numeric-specific profiling
    mean = std = median = min_val = max_val = None
    q1 = q3 = skewness = outlier_ratio = None
    is_high_cardinality = False
    
    if dtype == "numeric":
        non_missing = series.dropna()
        if len(non_missing) > 0:
            mean = float(non_missing.mean())
            std = float(non_missing.std())
            median = float(non_missing.median())
            min_val = float(non_missing.min())
            max_val = float(non_missing.max())
            q1 = float(non_missing.quantile(0.25))
            q3 = float(non_missing.quantile(0.75))
            
            # Skewness
            if len(non_missing) >= 3:
                try:
                    skewness = float(stats.skew(non_missing))
                except:
                    skewness = None
            
            # Outlier detection using IQR method
            if q1 is not None and q3 is not None:
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outliers = non_missing[(non_missing < lower_bound) | (non_missing > upper_bound)]
                outlier_ratio = len(outliers) / len(non_missing) if len(non_missing) > 0 else 0.0
    
    elif dtype == "categorical":
        # High cardinality check
        if n_unique > 50 or unique_ratio > 0.5:
            is_high_cardinality = True
    
    return FeatureProfile(
        name=name,
        dtype=dtype,
        n=n,
        n_missing=n_missing,
        missing_ratio=missing_ratio,
        n_unique=n_unique,
        unique_ratio=unique_ratio,
        mean=mean,
        std=std,
        median=median,
        min_val=min_val,
        max_val=max_val,
        q1=q1,
        q3=q3,
        skewness=skewness,
        outlier_ratio=outlier_ratio,
        is_constant=is_constant,
        is_high_cardinality=is_high_cardinality,
        is_identifier_like=is_identifier_like
    )


def profile_dataset(df: pd.DataFrame, target_column: str, use_cache: bool = True) -> ProfileResult:
    """
    Profile an entire dataset with optional caching.
    
    Args:
        df: Input DataFrame
        target_column: Name of target column
        use_cache: If True, cache results based on dataset fingerprint (default: True)
        
    Returns:
        ProfileResult object
        
    Note:
        Caching provides significant speedup for repeated profiling of the same dataset.
        The fingerprint is based on data content + target column, so it's safe.
    """
    if use_cache:
        fingerprint = _dataset_fingerprint(df, target_column)
        
        # Check cache
        if fingerprint in _PROFILE_CACHE:
            return _PROFILE_CACHE[fingerprint]
        
        # Compute and cache
        result = _profile_dataset_impl(df, target_column)
        _PROFILE_CACHE[fingerprint] = result
        
        # Keep cache size bounded
        if len(_PROFILE_CACHE) > 8:
            # Remove oldest entry (simple FIFO)
            _PROFILE_CACHE.pop(next(iter(_PROFILE_CACHE)))
        
        return result
    else:
        return _profile_dataset_impl(df, target_column)


def _dataset_fingerprint(df: pd.DataFrame, target_column: str) -> str:
    """
    Create a unique fingerprint for a dataset.
    
    This is used for caching profiling results. The fingerprint is based on:
    - Data content (via pandas hash)
    - Target column name
    - DataFrame shape
    
    Returns:
        MD5 hash string
    """
    h = hashlib.md5()
    
    # Hash the actual data content
    h.update(pd.util.hash_pandas_object(df, index=False).values)
    
    # Add target column and shape info
    h.update(target_column.encode())
    h.update(str(df.shape).encode())
    
    return h.hexdigest()


def _profile_dataset_impl(df: pd.DataFrame, target_column: str) -> ProfileResult:
    """
    Internal implementation of dataset profiling (no caching).
    
    Args:
        df: Input DataFrame
        target_column: Name of target column
        
    Returns:
        ProfileResult object
    """
    features = {}
    
    for col in df.columns:
        if col == target_column:
            continue
        features[col] = profile_feature(df[col], col)
    
    return ProfileResult(
        features=features,
        n_rows=len(df),
        n_features=len(features),
        target_column=target_column
    )


__all__ = [
    "FeatureProfile",
    "ProfileResult",
    "profile_feature",
    "profile_dataset"
]
