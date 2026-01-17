"""Test data loader module."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from automl_tabular.data import DataLoader, DataValidator


def test_load_csv():
    """Test CSV loading."""
    # Create temporary CSV
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("a,b,c\n1,2,3\n4,5,6\n")
        temp_path = f.name
    
    try:
        loader = DataLoader()
        df = loader.load_csv(temp_path)
        
        assert len(df) == 2
        assert len(df.columns) == 3
        assert list(df.columns) == ['a', 'b', 'c']
    finally:
        Path(temp_path).unlink()


def test_infer_problem_type():
    """Test problem type inference."""
    loader = DataLoader()
    
    # Classification (few unique values)
    y_class = pd.Series([0, 1, 0, 1, 0, 1])
    assert loader.infer_problem_type(y_class) == 'classification'
    
    # Regression (many unique values)
    y_reg = pd.Series(np.random.randn(100))
    assert loader.infer_problem_type(y_reg) == 'regression'


def test_data_validation():
    """Test data validation."""
    validator = DataValidator(min_rows=10)
    
    # Valid data
    df = pd.DataFrame({
        'a': range(50),
        'b': range(50),
        'target': [0, 1] * 25
    })
    
    is_valid, errors, warnings = validator.validate(df, 'target')
    assert is_valid
    assert len(errors) == 0


def test_constant_column_detection():
    """Test detection of constant columns."""
    validator = DataValidator()
    
    df = pd.DataFrame({
        'const': [1] * 50,
        'varied': range(50),
        'target': [0, 1] * 25
    })
    
    is_valid, errors, warnings = validator.validate(df, 'target')
    assert is_valid
    assert any('constant' in w.lower() for w in warnings)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
