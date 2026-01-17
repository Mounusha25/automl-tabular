"""Base preprocessing utilities and interfaces."""

from typing import Protocol, Any
import pandas as pd


class BasePreprocessor(Protocol):
    """Protocol for preprocessing components."""
    
    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> Any:
        """Fit the preprocessor on training data."""
        ...
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the data."""
        ...
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """Fit and transform in one step."""
        ...


__all__ = ["BasePreprocessor"]
