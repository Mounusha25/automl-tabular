"""Hyperparameter search and model optimization."""

import time
import warnings
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
import numpy as np
from sklearn.model_selection import cross_val_score

try:
    import optuna
    HAS_OPTUNA = True
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    HAS_OPTUNA = False

from automl_tabular.models.algorithms import ModelFactory, get_default_param_space


@dataclass
class TrialResult:
    """Result from a single hyperparameter trial."""
    model_name: str
    params: Dict[str, Any]
    score: float
    train_time: float
    model: Any = None


class ModelSearcher:
    """Handles hyperparameter search and model selection."""
    
    def __init__(
        self,
        problem_type: str,
        metric_func: Callable,
        max_trials_per_model: int = 20,
        time_limit_seconds: Optional[int] = None,
        cv_folds: int = 5,
        random_state: int = 42
    ):
        """
        Initialize model searcher.
        
        Args:
            problem_type: 'classification' or 'regression'
            metric_func: Function to compute metric (higher is better)
            max_trials_per_model: Maximum trials per model type
            time_limit_seconds: Time limit for entire search
            cv_folds: Number of cross-validation folds
            random_state: Random seed
        """
        self.problem_type = problem_type
        self.metric_func = metric_func
        self.max_trials_per_model = max_trials_per_model
        self.time_limit_seconds = time_limit_seconds
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.results: List[TrialResult] = []
        self.start_time = None
    
    def search(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        model_names: List[str],
        use_optuna: bool = True
    ) -> List[TrialResult]:
        """
        Search for best models and hyperparameters.
        
        Args:
            X_train: Training features
            y_train: Training targets
            model_names: List of model names to try
            use_optuna: Whether to use Optuna for optimization
            
        Returns:
            List of trial results, sorted by score (best first)
        """
        self.start_time = time.time()
        self.results = []
        
        for model_name in model_names:
            if self._should_stop():
                print(f"Time limit reached. Stopping search.")
                break
            
            print(f"\nSearching {model_name}...")
            
            try:
                if use_optuna and HAS_OPTUNA:
                    self._optuna_search(X_train, y_train, model_name)
                else:
                    self._random_search(X_train, y_train, model_name)
            except Exception as e:
                warnings.warn(f"Error searching {model_name}: {str(e)}")
                continue
        
        # Sort by score (descending)
        self.results.sort(key=lambda x: x.score, reverse=True)
        
        return self.results
    
    def _should_stop(self) -> bool:
        """Check if search should stop due to time limit."""
        if self.time_limit_seconds is None:
            return False
        
        elapsed = time.time() - self.start_time
        return elapsed >= self.time_limit_seconds
    
    def _optuna_search(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        model_name: str
    ) -> None:
        """Perform Optuna-based hyperparameter search."""
        
        def objective(trial):
            # Get parameter space
            param_space = get_default_param_space(model_name, self.problem_type)
            
            # Sample parameters
            params = {}
            for param_name, param_values in param_space.items():
                if isinstance(param_values, list):
                    if all(isinstance(v, (int, float)) for v in param_values if v is not None):
                        # Numeric parameter
                        valid_values = [v for v in param_values if v is not None]
                        if valid_values:
                            params[param_name] = trial.suggest_categorical(
                                param_name, param_values
                            )
                    else:
                        # Categorical parameter
                        params[param_name] = trial.suggest_categorical(
                            param_name, param_values
                        )
            
            # Create and evaluate model
            start_time = time.time()
            
            if self.problem_type == 'classification':
                model = ModelFactory.create_classifier(model_name, **params)
            else:
                model = ModelFactory.create_regressor(model_name, **params)
            
            if model is None:
                raise optuna.TrialPruned()
            
            # Cross-validation
            scores = cross_val_score(
                model, X_train, y_train,
                cv=self.cv_folds,
                scoring=self.metric_func,
                n_jobs=-1
            )
            
            train_time = time.time() - start_time
            score = scores.mean()
            
            # Store result
            self.results.append(TrialResult(
                model_name=model_name,
                params=params,
                score=score,
                train_time=train_time,
                model=model
            ))
            
            return score
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=self.random_state)
        )
        
        # Run optimization
        study.optimize(
            objective,
            n_trials=self.max_trials_per_model,
            timeout=self.time_limit_seconds,
            show_progress_bar=False,
            catch=(Exception,)
        )
    
    def _random_search(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        model_name: str
    ) -> None:
        """Perform random hyperparameter search."""
        from sklearn.model_selection import RandomizedSearchCV
        
        param_space = get_default_param_space(model_name, self.problem_type)
        
        if not param_space:
            # No hyperparameters to tune, just evaluate with defaults
            param_space = {}
        
        # Create model
        if self.problem_type == 'classification':
            model = ModelFactory.create_classifier(model_name)
        else:
            model = ModelFactory.create_regressor(model_name)
        
        if model is None:
            return
        
        # Random search
        search = RandomizedSearchCV(
            model,
            param_space,
            n_iter=min(self.max_trials_per_model, 10),
            cv=self.cv_folds,
            scoring=self.metric_func,
            n_jobs=-1,
            random_state=self.random_state,
            verbose=0
        )
        
        start_time = time.time()
        search.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        # Store best result
        self.results.append(TrialResult(
            model_name=model_name,
            params=search.best_params_,
            score=search.best_score_,
            train_time=train_time,
            model=search.best_estimator_
        ))
    
    def get_best_model(self) -> Optional[TrialResult]:
        """Get the best model result."""
        if not self.results:
            return None
        return self.results[0]


__all__ = ["ModelSearcher", "TrialResult"]
