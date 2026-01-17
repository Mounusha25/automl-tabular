"""Test preprocessing module."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from automl_tabular.preprocessing import TabularPreprocessor


def test_preprocessing_pipeline():
    """Test building preprocessing pipeline."""
    # Sample data
    df = pd.DataFrame({
        'num1': [1.0, 2.0, 3.0, np.nan, 5.0],
        'num2': [10, 20, 30, 40, 50],
        'cat1': ['a', 'b', 'a', 'c', 'b'],
        'cat2': ['x', 'y', None, 'x', 'y']
    })
    
    preprocessor = TabularPreprocessor()
    pipeline = preprocessor.build_pipeline(
        numeric_features=['num1', 'num2'],
        categorical_features=['cat1', 'cat2']
    )
    
    assert pipeline is not None
    
    # Test transform
    X_transformed = preprocessor.fit_transform(df)
    assert X_transformed.shape[0] == 5
    assert X_transformed.shape[1] > 0  # Should have multiple columns after one-hot encoding


def test_missing_value_handling():
    """Test that missing values are properly handled."""
    df = pd.DataFrame({
        'num': [1.0, np.nan, 3.0, 4.0],
        'cat': ['a', 'b', None, 'a']
    })
    
    preprocessor = TabularPreprocessor()
    pipeline = preprocessor.build_pipeline(
        numeric_features=['num'],
        categorical_features=['cat']
    )
    
    X_transformed = preprocessor.fit_transform(df)
    
    # No NaN in output
    assert not np.isnan(X_transformed).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
