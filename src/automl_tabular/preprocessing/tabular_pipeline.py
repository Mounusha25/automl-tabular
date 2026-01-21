"""Preprocessing pipeline construction for tabular data."""

import pandas as pd
import numpy as np
import scipy.sparse
from typing import Dict, List, Optional
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer


class TabularPreprocessor:
    """Builds and manages preprocessing pipelines for tabular data."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the preprocessor.
        
        Args:
            config: Configuration dictionary for preprocessing options
        """
        self.config = config or {}
        self.pipeline = None
        self.numeric_features = []
        self.categorical_features = []
        self.feature_names = []
    
    def build_pipeline(
        self,
        numeric_features: List[str],
        categorical_features: List[str]
    ) -> Pipeline:
        """
        Build a preprocessing pipeline.
        
        Args:
            numeric_features: List of numeric column names
            categorical_features: List of categorical column names
            
        Returns:
            Scikit-learn Pipeline object
        """
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        
        transformers = []
        
        # Numeric pipeline
        if numeric_features:
            numeric_transformer = self._build_numeric_pipeline()
            transformers.append(('num', numeric_transformer, numeric_features))
        
        # Categorical pipeline
        if categorical_features:
            categorical_transformer = self._build_categorical_pipeline()
            transformers.append(('cat', categorical_transformer, categorical_features))
        
        # Combine transformers
        # If using sparse matrices, keep them sparse through the pipeline
        use_sparse = self.config.get('use_sparse_matrices', True)
        sparse_threshold = 0.3 if use_sparse else 0  # Keep sparse if >30% sparse
        
        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='drop',  # Drop any columns not specified
            sparse_threshold=sparse_threshold
        )
        
        self.pipeline = Pipeline([
            ('preprocessor', preprocessor)
        ])
        
        return self.pipeline
    
    def _build_numeric_pipeline(self) -> Pipeline:
        """Build pipeline for numeric features."""
        steps = []
        
        # Imputation
        impute_strategy = self.config.get('numeric_impute_strategy', 'median')
        steps.append(('imputer', SimpleImputer(strategy=impute_strategy)))
        
        # Scaling
        scaling = self.config.get('scaling', 'standard')
        if scaling == 'standard':
            steps.append(('scaler', StandardScaler()))
        elif scaling == 'minmax':
            steps.append(('scaler', MinMaxScaler()))
        # else: no scaling
        
        return Pipeline(steps)
    
    def _build_categorical_pipeline(self) -> Pipeline:
        """Build pipeline for categorical features."""
        steps = []
        
        # Imputation
        impute_strategy = self.config.get('categorical_impute_strategy', 'most_frequent')
        steps.append(('imputer', SimpleImputer(strategy=impute_strategy)))
        
        # Encoding
        encoding = self.config.get('encoding', 'onehot')
        handle_unknown = self.config.get('handle_unknown', 'ignore')
        
        if encoding == 'onehot':
            # Use sparse matrices for efficiency (especially with high cardinality)
            use_sparse = self.config.get('use_sparse_matrices', True)
            steps.append((
                'encoder',
                OneHotEncoder(
                    handle_unknown=handle_unknown,
                    sparse_output=use_sparse,
                    drop='first'  # Avoid dummy variable trap
                )
            ))
        # For other encoding types (target, label), we'll implement as needed
        
        return Pipeline(steps)
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> np.ndarray:
        """
        Fit the pipeline and transform data.
        
        Args:
            X: Input features
            y: Target (optional, for target encoding)
            
        Returns:
            Transformed feature array (optionally as float32 for efficiency)
        """
        if self.pipeline is None:
            raise ValueError("Pipeline not built. Call build_pipeline() first.")
        
        X_transformed = self.pipeline.fit_transform(X, y)
        
        # Convert to float32 for memory/speed (unless sparse)
        use_float32 = self.config.get('use_float32', True)
        if use_float32 and not isinstance(X_transformed, (scipy.sparse.spmatrix, scipy.sparse.sparray)):
            X_transformed = X_transformed.astype(np.float32)
        
        return X_transformed
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform data using fitted pipeline.
        
        Args:
            X: Input features
            
        Returns:
            Transformed feature array (optionally as float32 for efficiency)
        """
        if self.pipeline is None:
            raise ValueError("Pipeline not built. Call build_pipeline() first.")
        
        X_transformed = self.pipeline.transform(X)
        
        # Convert to float32 for memory/speed (unless sparse)
        use_float32 = self.config.get('use_float32', True)
        if use_float32 and not isinstance(X_transformed, (scipy.sparse.spmatrix, scipy.sparse.sparray)):
            X_transformed = X_transformed.astype(np.float32)
        
        return X_transformed
    
    def get_feature_names(self) -> List[str]:
        """
        Get feature names after transformation.
        
        Returns:
            List of feature names
        """
        if self.pipeline is None:
            return []
        
        feature_names = []
        
        # Get transformer from pipeline
        preprocessor = self.pipeline.named_steps['preprocessor']
        
        # Numeric features (keep original names)
        if self.numeric_features:
            feature_names.extend(self.numeric_features)
        
        # Categorical features (get encoded names)
        if self.categorical_features:
            try:
                # Try to get feature names from one-hot encoder
                cat_transformer = preprocessor.named_transformers_['cat']
                encoder = cat_transformer.named_steps.get('encoder')
                
                if encoder and hasattr(encoder, 'get_feature_names_out'):
                    cat_names = encoder.get_feature_names_out(self.categorical_features)
                    feature_names.extend(cat_names)
                else:
                    # Fallback to original categorical names
                    feature_names.extend(self.categorical_features)
            except:
                feature_names.extend(self.categorical_features)
        
        self.feature_names = feature_names
        return feature_names


def remove_constant_columns(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """
    Remove columns with constant values.
    
    Args:
        df: Input DataFrame
        target_column: Name of target column (will not be removed)
        
    Returns:
        DataFrame with constant columns removed
    """
    cols_to_keep = []
    
    for col in df.columns:
        if col == target_column or df[col].nunique() > 1:
            cols_to_keep.append(col)
    
    return df[cols_to_keep]


def remove_high_missing_columns(
    df: pd.DataFrame,
    target_column: str,
    threshold: float = 0.95
) -> pd.DataFrame:
    """
    Remove columns with high percentage of missing values.
    
    Args:
        df: Input DataFrame
        target_column: Name of target column (will not be removed)
        threshold: Maximum allowed missing ratio
        
    Returns:
        DataFrame with high-missing columns removed
    """
    cols_to_keep = []
    
    for col in df.columns:
        missing_ratio = df[col].isnull().sum() / len(df)
        if col == target_column or missing_ratio <= threshold:
            cols_to_keep.append(col)
    
    return df[cols_to_keep]


__all__ = [
    "TabularPreprocessor",
    "remove_constant_columns",
    "remove_high_missing_columns"
]
