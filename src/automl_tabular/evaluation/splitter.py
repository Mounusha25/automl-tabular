"""Data splitting utilities."""

import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from typing import Tuple, Iterator


def split_train_validation(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    problem_type: str = 'classification',
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into training and validation sets.
    
    Args:
        X: Features
        y: Target
        test_size: Proportion for validation set
        problem_type: 'classification' or 'regression'
        random_state: Random seed
        
    Returns:
        X_train, X_val, y_train, y_val
    """
    # Use stratified split for classification
    stratify = y if problem_type == 'classification' else None
    
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify
    )


def get_cv_splitter(
    problem_type: str = 'classification',
    n_splits: int = 5,
    random_state: int = 42
):
    """
    Get cross-validation splitter.
    
    Args:
        problem_type: 'classification' or 'regression'
        n_splits: Number of folds
        random_state: Random seed
        
    Returns:
        CV splitter object
    """
    if problem_type == 'classification':
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    else:
        return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)


__all__ = [
    "split_train_validation",
    "get_cv_splitter"
]
