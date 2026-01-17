"""Model registry for pluggable algorithm architecture."""

from dataclasses import dataclass
from typing import Callable, Set, Dict, Any
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb


@dataclass
class ModelSpec:
    """Specification for a model algorithm."""
    
    name: str
    family: str
    estimator_cls: Callable
    problem_types: Set[str]
    simplicity: int  # Lower = simpler, for tie-breaking
    param_space: Callable  # (trial) -> dict
    enabled: bool = True
    
    def is_compatible(self, problem_type: str) -> bool:
        """Check if model is compatible with problem type."""
        return self.enabled and problem_type in self.problem_types


def _logistic_param_space(trial):
    """Hyperparameter space for Logistic Regression."""
    return {
        'C': trial.suggest_float('C', 0.01, 100.0, log=True),
        'penalty': 'l2',
        'solver': 'liblinear',
        'max_iter': 1000,
        'random_state': 42
    }


def _linear_param_space(trial):
    """Hyperparameter space for Linear Regression."""
    return {
        'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False])
    }


def _random_forest_clf_param_space(trial):
    """Hyperparameter space for Random Forest Classifier."""
    return {
        'n_estimators': trial.suggest_int('n_estimators', 100, 400),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
        'random_state': 42,
        'n_jobs': -1
    }


def _random_forest_reg_param_space(trial):
    """Hyperparameter space for Random Forest Regressor."""
    return {
        'n_estimators': trial.suggest_int('n_estimators', 100, 400),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'random_state': 42,
        'n_jobs': -1
    }


def _xgboost_clf_param_space(trial):
    """Hyperparameter space for XGBoost Classifier."""
    return {
        'n_estimators': trial.suggest_int('n_estimators', 100, 400),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0.0, 1.0),
        'random_state': 42,
        'n_jobs': -1,
        'use_label_encoder': False,
        'eval_metric': 'logloss'
    }


def _xgboost_reg_param_space(trial):
    """Hyperparameter space for XGBoost Regressor."""
    return {
        'n_estimators': trial.suggest_int('n_estimators', 100, 400),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0.0, 1.0),
        'random_state': 42,
        'n_jobs': -1
    }


def _lightgbm_clf_param_space(trial):
    """Hyperparameter space for LightGBM Classifier."""
    return {
        'n_estimators': trial.suggest_int('n_estimators', 100, 400),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }


def _lightgbm_reg_param_space(trial):
    """Hyperparameter space for LightGBM Regressor."""
    return {
        'n_estimators': trial.suggest_int('n_estimators', 100, 400),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 200),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }


# Global model registry
MODEL_REGISTRY: Dict[str, ModelSpec] = {
    'logistic_regression': ModelSpec(
        name='logistic_regression',
        family='linear',
        estimator_cls=LogisticRegression,
        problem_types={'binary', 'multiclass'},
        simplicity=1,
        param_space=_logistic_param_space
    ),
    'linear_regression': ModelSpec(
        name='linear_regression',
        family='linear',
        estimator_cls=LinearRegression,
        problem_types={'regression'},
        simplicity=1,
        param_space=_linear_param_space
    ),
    'random_forest': ModelSpec(
        name='random_forest',
        family='tree_bagging',
        estimator_cls=lambda problem_type: (
            RandomForestClassifier if problem_type in {'binary', 'multiclass'}
            else RandomForestRegressor
        ),
        problem_types={'binary', 'multiclass', 'regression'},
        simplicity=2,
        param_space=lambda trial, problem_type: (
            _random_forest_clf_param_space(trial) if problem_type in {'binary', 'multiclass'}
            else _random_forest_reg_param_space(trial)
        )
    ),
    'xgboost': ModelSpec(
        name='xgboost',
        family='gradient_boosting',
        estimator_cls=lambda problem_type: (
            xgb.XGBClassifier if problem_type in {'binary', 'multiclass'}
            else xgb.XGBRegressor
        ),
        problem_types={'binary', 'multiclass', 'regression'},
        simplicity=3,
        param_space=lambda trial, problem_type: (
            _xgboost_clf_param_space(trial) if problem_type in {'binary', 'multiclass'}
            else _xgboost_reg_param_space(trial)
        )
    ),
    'lightgbm': ModelSpec(
        name='lightgbm',
        family='gradient_boosting',
        estimator_cls=lambda problem_type: (
            lgb.LGBMClassifier if problem_type in {'binary', 'multiclass'}
            else lgb.LGBMRegressor
        ),
        problem_types={'binary', 'multiclass', 'regression'},
        simplicity=3,
        param_space=lambda trial, problem_type: (
            _lightgbm_clf_param_space(trial) if problem_type in {'binary', 'multiclass'}
            else _lightgbm_reg_param_space(trial)
        )
    ),
}


def get_compatible_models(problem_type: str, algorithms: list = None) -> Dict[str, ModelSpec]:
    """
    Get models compatible with the problem type.
    
    Args:
        problem_type: 'binary', 'multiclass', or 'regression'
        algorithms: Optional list of algorithm names to filter
        
    Returns:
        Dictionary of compatible model specs
    """
    models = {}
    for name, spec in MODEL_REGISTRY.items():
        if algorithms and name not in algorithms:
            continue
        if spec.is_compatible(problem_type):
            models[name] = spec
    return models


def get_model_simplicity(model_name: str) -> int:
    """Get simplicity score for a model (lower = simpler)."""
    if model_name in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_name].simplicity
    return 999  # Unknown models considered most complex


__all__ = [
    'ModelSpec',
    'MODEL_REGISTRY',
    'get_compatible_models',
    'get_model_simplicity'
]
