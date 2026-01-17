"""Model ensemble utilities (optional for v1)."""

import numpy as np
from typing import List, Any


class SimpleEnsemble:
    """Simple averaging ensemble for multiple models."""
    
    def __init__(self, models: List[Any], weights: List[float] = None):
        """
        Initialize ensemble.
        
        Args:
            models: List of fitted models
            weights: Optional weights for each model
        """
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using weighted average of models."""
        predictions = []
        
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        weighted_pred = np.average(predictions, axis=0, weights=self.weights)
        
        return weighted_pred
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities using weighted average."""
        predictions = []
        
        for model in self.models:
            if hasattr(model, 'predict_proba'):
                pred = model.predict_proba(X)
            else:
                # Fallback to binary predictions
                pred = model.predict(X)
                pred = np.column_stack([1 - pred, pred])
            
            predictions.append(pred)
        
        predictions = np.array(predictions)
        weighted_pred = np.average(predictions, axis=0, weights=self.weights)
        
        return weighted_pred


__all__ = ["SimpleEnsemble"]
