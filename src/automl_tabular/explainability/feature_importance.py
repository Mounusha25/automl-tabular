"""Feature importance calculation utilities."""

import numpy as np
import pandas as pd
from typing import List, Optional, Any
from sklearn.inspection import permutation_importance


def get_feature_importance(
    model: Any,
    feature_names: List[str],
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    method: str = 'auto'
) -> pd.DataFrame:
    """
    Extract feature importance from a model.
    
    Args:
        model: Trained model
        feature_names: List of feature names
        X_val: Validation features (for permutation importance)
        y_val: Validation targets (for permutation importance)
        method: 'auto', 'builtin', or 'permutation'
        
    Returns:
        DataFrame with features and their importance scores
    """
    importance_values = None
    importance_method = method
    
    # Try built-in feature importance first
    if method in ['auto', 'builtin']:
        if hasattr(model, 'feature_importances_'):
            importance_values = model.feature_importances_
            importance_method = 'builtin'
        elif hasattr(model, 'coef_'):
            # For linear models, use absolute coefficients
            coef = model.coef_
            if coef.ndim > 1:
                # Multi-class: average across classes
                importance_values = np.abs(coef).mean(axis=0)
            else:
                importance_values = np.abs(coef)
            importance_method = 'coefficients'
    
    # Fall back to permutation importance
    if importance_values is None and X_val is not None and y_val is not None:
        result = permutation_importance(
            model, X_val, y_val,
            n_repeats=10,
            random_state=42,
            n_jobs=-1
        )
        importance_values = result.importances_mean
        importance_method = 'permutation'
    
    # Create DataFrame
    if importance_values is not None:
        # Ensure we have the right number of features
        n_features = min(len(feature_names), len(importance_values))
        
        df = pd.DataFrame({
            'feature': feature_names[:n_features],
            'importance': importance_values[:n_features],
            'method': importance_method
        })
        
        # Sort by importance
        df = df.sort_values('importance', ascending=False).reset_index(drop=True)
        
        return df
    
    # Return empty DataFrame if no importance available
    return pd.DataFrame(columns=['feature', 'importance', 'method'])


def get_top_features(
    feature_importance_df: pd.DataFrame,
    top_k: int = 10
) -> pd.DataFrame:
    """
    Get top K most important features.
    
    Args:
        feature_importance_df: DataFrame with feature importance
        top_k: Number of top features
        
    Returns:
        DataFrame with top K features
    """
    return feature_importance_df.head(top_k)


def normalize_importance(feature_importance_df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize importance scores to sum to 1.
    
    Args:
        feature_importance_df: DataFrame with feature importance
        
    Returns:
        DataFrame with normalized importance
    """
    df = feature_importance_df.copy()
    total = df['importance'].sum()
    
    if total > 0:
        df['importance_normalized'] = df['importance'] / total
        df['importance_percentage'] = df['importance_normalized'] * 100
    else:
        df['importance_normalized'] = 0
        df['importance_percentage'] = 0
    
    return df


def aggregate_importance_by_column(
    feature_importance_df: pd.DataFrame,
    exclude_high_cardinality: bool = True,
    high_cardinality_threshold: int = 50
) -> pd.DataFrame:
    """
    Aggregate feature importance by original column name.
    
    For one-hot encoded features like 'Sex_male', 'Sex_female',
    this sums their importance under the parent column 'Sex'.
    
    Args:
        feature_importance_df: DataFrame with feature importance
        exclude_high_cardinality: Whether to exclude high-cardinality columns
        high_cardinality_threshold: Number of unique values to consider high cardinality
        
    Returns:
        DataFrame with aggregated importance by original column
    """
    if feature_importance_df.empty:
        return feature_importance_df
    
    from collections import defaultdict
    
    df = feature_importance_df.copy()
    column_importance = defaultdict(float)
    column_feature_counts = defaultdict(int)
    
    # Identify high-cardinality columns (like Name, Ticket)
    high_cardinality_prefixes = set()
    if exclude_high_cardinality:
        # Count unique values per column prefix
        prefix_counts = defaultdict(set)
        for feature in df['feature']:
            if '_' in feature:
                prefix = feature.split('_', 1)[0]
                suffix = feature.split('_', 1)[1]
                prefix_counts[prefix].add(suffix)
        
        # Mark columns with too many unique values
        for prefix, values in prefix_counts.items():
            if len(values) > high_cardinality_threshold:
                high_cardinality_prefixes.add(prefix)
    
    # Define identifier patterns to exclude
    identifier_patterns = ['id', 'passengerid', 'customerid', 'userid', 'index']
    
    # Aggregate importance by original column
    for _, row in df.iterrows():
        feature = row['feature']
        importance = row['importance']
        
        # Extract original column name
        if '_' in feature:
            original_col = feature.split('_', 1)[0]
        else:
            original_col = feature
        
        # Skip identifier columns
        if original_col.lower() in identifier_patterns:
            continue
        
        # Skip high-cardinality columns
        if original_col in high_cardinality_prefixes:
            continue
        
        column_importance[original_col] += importance
        column_feature_counts[original_col] += 1
    
    # Create aggregated DataFrame
    if not column_importance:
        return pd.DataFrame(columns=['feature', 'importance', 'num_features'])
    
    agg_df = pd.DataFrame([
        {
            'feature': col,
            'importance': imp,
            'num_features': column_feature_counts[col]
        }
        for col, imp in column_importance.items()
    ])
    
    # Sort by importance
    agg_df = agg_df.sort_values('importance', ascending=False).reset_index(drop=True)
    
    return agg_df


__all__ = [
    "get_feature_importance",
    "get_top_features",
    "normalize_importance",
    "aggregate_importance_by_column"
]
