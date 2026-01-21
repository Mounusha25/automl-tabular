"""Hyperparameter search and model optimization."""

import time
import warnings
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import get_scorer

try:
    import optuna
    from optuna.pruners import MedianPruner
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
        random_state: int = 42,
        enable_pruning: bool = True
    ):
        """
        Initialize model searcher.
        
        Args:
            problem_type: 'classification' or 'regression'
            metric_func: Function to compute metric (higher is better)
            max_trials_per_model: Maximum trials per model type
            time_limit_seconds: Time limit for entire search
            cv_folds: Number of cross-validation folds (auto-adjusted based on dataset size)
            random_state: Random seed
            enable_pruning: Enable Optuna MedianPruner for early stopping (default: True)
        """
        self.problem_type = problem_type
        self.metric_func = metric_func
        self.max_trials_per_model = max_trials_per_model
        self.time_limit_seconds = time_limit_seconds
        self.cv_folds_config = cv_folds  # Store config value
        self.random_state = random_state
        self.enable_pruning = enable_pruning
        self.results: List[TrialResult] = []
        self.start_time = None
    
    def _get_cv_folds(self, n_samples: int) -> int:
        """
        Determine number of CV folds based on dataset size.
        
        Heuristic (safe & explainable):
        - If dataset size < 5,000 → use configured folds (typically 3)
        - If 5,000-50,000 → 2-fold CV
        - If > 50,000 → holdout only (1-fold)
        
        This keeps results stable while saving time on larger datasets.
        
        Args:
            n_samples: Number of samples in training set
            
        Returns:
            Number of CV folds to use
        """
        if n_samples < 5000:
            return self.cv_folds_config
        elif n_samples < 50000:
            return 2
        else:
            return 1  # Holdout only for very large datasets
    
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
            
            # Dynamic CV folds based on dataset size
            # Handle both dense and sparse matrices
            n_samples = X_train.shape[0]
            cv_folds = self._get_cv_folds(n_samples)
            
            # Cross-validation with per-fold reporting for pruning
            from sklearn.model_selection import cross_validate
            
            start_time = time.time()
            
            # If pruning enabled, report per-fold scores
            if self.enable_pruning and cv_folds > 1:
                # Manual CV with per-fold reporting
                from sklearn.model_selection import StratifiedKFold, KFold
                
                if self.problem_type == 'classification':
                    splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
                else:
                    splitter = KFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
                
                fold_scores = []
                for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X_train, y_train)):
                    X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
                    y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
                    
                    # Train on this fold
                    model.fit(X_fold_train, y_fold_train)
                    
                    # Evaluate
                    scorer = get_scorer(self.metric_func)
                    fold_score = scorer(model, X_fold_val, y_fold_val)
                    fold_scores.append(fold_score)
                    
                    # Report intermediate value for pruning
                    trial.report(np.mean(fold_scores), step=fold_idx)
                    
                    # Check if trial should be pruned
                    if trial.should_prune():
                        raise optuna.TrialPruned()
                
                score = np.mean(fold_scores)
            else:
                # Standard cross_val_score (no per-fold reporting)
                scores = cross_val_score(
                    model, X_train, y_train,
                    cv=cv_folds,
                    scoring=self.metric_func,
                    n_jobs=-1
                )
                score = scores.mean()
            
            train_time = time.time() - start_time
            
            # Store result
            self.results.append(TrialResult(
                model_name=model_name,
                params=params,
                score=score,
                train_time=train_time,
                model=model
            ))
            
            return score
        
        # Create study with optional pruning
        if self.enable_pruning:
            pruner = MedianPruner(
                n_startup_trials=5,  # Wait for 5 trials before pruning
                n_warmup_steps=1      # Start pruning after 1 fold
            )
        else:
            pruner = optuna.pruners.NopPruner()
        
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
            pruner=pruner
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
