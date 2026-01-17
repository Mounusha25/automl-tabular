"""Metric computation and evaluation utilities."""

import numpy as np
from typing import Callable, Dict, List
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    log_loss
)


def get_metric_function(metric_name: str, problem_type: str) -> Callable:
    """
    Get scikit-learn scoring function name for the metric.
    
    Args:
        metric_name: Name of the metric
        problem_type: 'classification' or 'regression'
        
    Returns:
        Scoring function name (string) for use with cross_val_score
    """
    metric_name = metric_name.lower()
    
    if problem_type == 'classification':
        metric_map = {
            'accuracy': 'accuracy',
            'roc_auc': 'roc_auc',
            'f1': 'f1',
            'f1_weighted': 'f1_weighted',
            'precision': 'precision',
            'recall': 'recall'
        }
    else:  # regression
        metric_map = {
            'rmse': 'neg_root_mean_squared_error',
            'mse': 'neg_mean_squared_error',
            'mae': 'neg_mean_absolute_error',
            'r2': 'r2'
        }
    
    return metric_map.get(metric_name, metric_map[list(metric_map.keys())[0]])


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    problem_type: str,
    y_pred_proba: np.ndarray = None
) -> Dict[str, float]:
    """
    Compute multiple metrics for evaluation.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        problem_type: 'classification' or 'regression'
        y_pred_proba: Predicted probabilities (for classification)
        
    Returns:
        Dictionary of metric names and values
    """
    metrics = {}
    
    if problem_type == 'classification':
        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        # Handle binary vs multiclass
        n_classes = len(np.unique(y_true))
        
        if n_classes == 2:
            # Binary classification
            metrics['f1'] = f1_score(y_true, y_pred)
            metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
            metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
            
            if y_pred_proba is not None:
                try:
                    # For binary, use probability of positive class
                    if y_pred_proba.ndim > 1 and y_pred_proba.shape[1] > 1:
                        metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
                    else:
                        metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
                    
                    metrics['log_loss'] = log_loss(y_true, y_pred_proba)
                except:
                    pass
        else:
            # Multiclass
            metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted')
            metrics['precision_weighted'] = precision_score(
                y_true, y_pred, average='weighted', zero_division=0
            )
            metrics['recall_weighted'] = recall_score(
                y_true, y_pred, average='weighted', zero_division=0
            )
            
            if y_pred_proba is not None:
                try:
                    metrics['roc_auc_ovr'] = roc_auc_score(
                        y_true, y_pred_proba, multi_class='ovr'
                    )
                    metrics['log_loss'] = log_loss(y_true, y_pred_proba)
                except:
                    pass
    
    else:  # regression
        metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['r2'] = r2_score(y_true, y_pred)
    
    return metrics


def get_primary_metric(problem_type: str, config: Dict = None) -> str:
    """
    Get the primary metric for optimization.
    
    Args:
        problem_type: 'classification' or 'regression'
        config: Configuration dictionary
        
    Returns:
        Metric name
    """
    if config and 'metrics' in config:
        if problem_type == 'classification':
            return config['metrics'].get('classification_primary', 'roc_auc')
        else:
            return config['metrics'].get('regression_primary', 'rmse')
    
    # Defaults
    if problem_type == 'classification':
        return 'roc_auc'
    else:
        return 'rmse'


__all__ = [
    "get_metric_function",
    "compute_metrics",
    "get_primary_metric"
]
