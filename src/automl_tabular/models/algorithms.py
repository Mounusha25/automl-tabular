"""Model algorithm wrappers and creation utilities."""

from typing import Any, Dict, Optional
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
import warnings


# Model simplicity scores (lower = simpler, more interpretable)
# Used for tie-breaking when models have similar performance
MODEL_SIMPLICITY = {
    "logistic_regression": 1,
    "linear_regression": 1,
    "ridge": 1,
    "lasso": 1,
    "svm": 2,
    "random_forest": 2,
    "random_forest_regressor": 2,
    "xgboost": 3,
    "xgboost_regressor": 3,
    "lightgbm": 3,
    "lightgbm_regressor": 3,
}

# Try to import XGBoost and LightGBM (optional dependencies)
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False


class ModelFactory:
    """Factory for creating machine learning models."""
    
    @staticmethod
    def create_classifier(name: str, **params) -> Any:
        """
        Create a classifier model.
        
        Args:
            name: Model name
            **params: Model parameters
            
        Returns:
            Initialized model instance
        """
        name = name.lower()
        
        if name == 'logistic_regression':
            return LogisticRegression(max_iter=1000, **params)
        
        elif name == 'random_forest':
            return RandomForestClassifier(**params)
        
        elif name == 'xgboost':
            if not HAS_XGBOOST:
                warnings.warn("XGBoost not installed. Skipping.")
                return None
            return xgb.XGBClassifier(
                use_label_encoder=False,
                eval_metric='logloss',
                **params
            )
        
        elif name == 'lightgbm':
            if not HAS_LIGHTGBM:
                warnings.warn("LightGBM not installed. Skipping.")
                return None
            return lgb.LGBMClassifier(verbosity=-1, **params)
        
        elif name == 'svm':
            return SVC(probability=True, **params)
        
        else:
            raise ValueError(f"Unknown classifier: {name}")
    
    @staticmethod
    def create_regressor(name: str, **params) -> Any:
        """
        Create a regressor model.
        
        Args:
            name: Model name
            **params: Model parameters
            
        Returns:
            Initialized model instance
        """
        name = name.lower()
        
        if name == 'linear_regression':
            return LinearRegression(**params)
        
        elif name == 'ridge':
            return Ridge(**params)
        
        elif name == 'lasso':
            return Lasso(**params)
        
        elif name == 'random_forest_regressor':
            return RandomForestRegressor(**params)
        
        elif name == 'xgboost_regressor':
            if not HAS_XGBOOST:
                warnings.warn("XGBoost not installed. Skipping.")
                return None
            return xgb.XGBRegressor(**params)
        
        elif name == 'lightgbm_regressor':
            if not HAS_LIGHTGBM:
                warnings.warn("LightGBM not installed. Skipping.")
                return None
            return lgb.LGBMRegressor(verbosity=-1, **params)
        
        elif name == 'svr':
            return SVR(**params)
        
        else:
            raise ValueError(f"Unknown regressor: {name}")


def get_default_param_space(model_name: str, problem_type: str) -> Dict:
    """
    Get default hyperparameter search space for a model.
    
    Args:
        model_name: Name of the model
        problem_type: 'classification' or 'regression'
        
    Returns:
        Dictionary defining parameter search space
    """
    model_name = model_name.lower()
    
    if problem_type == 'classification':
        if model_name == 'logistic_regression':
            return {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
        
        elif model_name == 'random_forest':
            return {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2']
            }
        
        elif model_name == 'xgboost':
            return {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7, 9],
                'learning_rate': [0.01, 0.05, 0.1, 0.3],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0]
            }
        
        elif model_name == 'lightgbm':
            return {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7, -1],
                'learning_rate': [0.01, 0.05, 0.1, 0.3],
                'num_leaves': [15, 31, 63],
                'subsample': [0.6, 0.8, 1.0]
            }
    
    else:  # regression
        if model_name in ['linear_regression', 'ridge', 'lasso']:
            return {
                'alpha': [0.001, 0.01, 0.1, 1, 10, 100]
            } if model_name != 'linear_regression' else {}
        
        elif model_name == 'random_forest_regressor':
            return {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            }
        
        elif model_name == 'xgboost_regressor':
            return {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7, 9],
                'learning_rate': [0.01, 0.05, 0.1, 0.3],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0]
            }
        
        elif model_name == 'lightgbm_regressor':
            return {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7, -1],
                'learning_rate': [0.01, 0.05, 0.1, 0.3],
                'num_leaves': [15, 31, 63],
                'subsample': [0.6, 0.8, 1.0]
            }
    
    return {}


__all__ = ["ModelFactory", "get_default_param_space", "HAS_XGBOOST", "HAS_LIGHTGBM"]
