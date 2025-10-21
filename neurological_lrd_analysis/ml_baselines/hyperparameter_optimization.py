"""
Hyperparameter optimization for ML baseline estimators using Optuna.

This module provides efficient hyperparameter tuning for machine learning
models using Optuna's Bayesian optimization and pruning capabilities.

Author: Davian R. Chin (PhD Candidate in Biomedical Engineering, University of Reading, UK)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass
import warnings
import time

# Lazy imports
def _lazy_import_optuna():
    """Lazy import of optuna"""
    try:
        import optuna
        from optuna.pruners import MedianPruner
        from optuna.samplers import TPESampler
        return optuna, MedianPruner, TPESampler
    except ImportError:
        warnings.warn("Optuna not available, hyperparameter optimization will be disabled")
        return None, None, None

def _lazy_import_sklearn():
    """Lazy import of sklearn modules"""
    try:
        from sklearn.model_selection import cross_val_score, StratifiedKFold
        from sklearn.metrics import mean_squared_error, make_scorer
        return cross_val_score, StratifiedKFold, mean_squared_error, make_scorer
    except ImportError:
        warnings.warn("scikit-learn not available, hyperparameter optimization will be disabled")
        return None, None, None, None


@dataclass
class OptimizationResult:
    """Results from hyperparameter optimization."""
    best_params: Dict[str, Any]
    best_score: float
    best_trial: int
    optimization_time: float
    n_trials: int
    study: Any


class OptunaOptimizer:
    """Hyperparameter optimizer using Optuna."""
    
    def __init__(self,
                 study_name: Optional[str] = None,
                 direction: str = 'minimize',
                 n_trials: int = 100,
                 timeout: Optional[float] = None,
                 pruner: Optional[str] = 'median',
                 sampler: Optional[str] = 'tpe',
                 random_state: int = 42):
        """
        Initialize the Optuna optimizer.
        
        Parameters:
        -----------
        study_name : str, optional
            Name of the study
        direction : str
            Optimization direction ('minimize' or 'maximize')
        n_trials : int
            Number of trials to run
        timeout : float, optional
            Timeout in seconds
        pruner : str, optional
            Pruning strategy ('median', 'percentile', 'successive_halving', None)
        sampler : str, optional
            Sampling strategy ('tpe', 'random', 'cmaes', 'grid')
        random_state : int
            Random state for reproducibility
        """
        self.optuna, self.MedianPruner, self.TPESampler = _lazy_import_optuna()
        
        if self.optuna is None:
            raise ImportError("Optuna not available")
        
        self.study_name = study_name
        self.direction = direction
        self.n_trials = n_trials
        self.timeout = timeout
        self.random_state = random_state
        
        # Setup pruner
        if pruner == 'median':
            self.pruner = self.MedianPruner()
        elif pruner == 'percentile':
            self.pruner = self.optuna.pruners.PercentilePruner(25.0)
        elif pruner == 'successive_halving':
            self.pruner = self.optuna.pruners.SuccessiveHalvingPruner()
        else:
            self.pruner = None
        
        # Setup sampler
        if sampler == 'tpe':
            self.sampler = self.TPESampler(seed=random_state)
        elif sampler == 'random':
            self.sampler = self.optuna.samplers.RandomSampler(seed=random_state)
        elif sampler == 'cmaes':
            self.sampler = self.optuna.samplers.CmaEsSampler(seed=random_state)
        elif sampler == 'grid':
            self.sampler = self.optuna.samplers.GridSampler()
        else:
            self.sampler = None
        
        self.study = None
    
    def optimize_random_forest(self,
                              X: np.ndarray,
                              y: np.ndarray,
                              cv_folds: int = 5,
                              scoring: str = 'neg_mean_squared_error') -> OptimizationResult:
        """
        Optimize Random Forest hyperparameters.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target values
        cv_folds : int
            Number of cross-validation folds
        scoring : str
            Scoring metric
            
        Returns:
        --------
        OptimizationResult
            Optimization results
        """
        def objective(trial):
            # Suggest hyperparameters
            n_estimators = trial.suggest_int('n_estimators', 10, 500)
            max_depth = trial.suggest_int('max_depth', 3, 20)
            min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
            min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
            max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
            
            # Create model
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                random_state=self.random_state,
                n_jobs=-1
            )
            
            # Cross-validation
            from sklearn.model_selection import cross_val_score
            scores = cross_val_score(model, X, y, cv=cv_folds, scoring=scoring, n_jobs=-1)
            return scores.mean()
        
        return self._run_optimization(objective, "RandomForest")
    
    def optimize_svr(self,
                    X: np.ndarray,
                    y: np.ndarray,
                    cv_folds: int = 5,
                    scoring: str = 'neg_mean_squared_error') -> OptimizationResult:
        """
        Optimize SVR hyperparameters.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target values
        cv_folds : int
            Number of cross-validation folds
        scoring : str
            Scoring metric
            
        Returns:
        --------
        OptimizationResult
            Optimization results
        """
        def objective(trial):
            # Suggest hyperparameters
            C = trial.suggest_float('C', 0.1, 100.0, log=True)
            gamma = trial.suggest_categorical('gamma', ['scale', 'auto']) or trial.suggest_float('gamma', 1e-4, 1e-1, log=True)
            epsilon = trial.suggest_float('epsilon', 0.01, 1.0)
            kernel = trial.suggest_categorical('kernel', ['rbf', 'linear', 'poly', 'sigmoid'])
            
            # Create model
            from sklearn.svm import SVR
            model = SVR(
                C=C,
                gamma=gamma,
                epsilon=epsilon,
                kernel=kernel
            )
            
            # Cross-validation
            from sklearn.model_selection import cross_val_score
            scores = cross_val_score(model, X, y, cv=cv_folds, scoring=scoring, n_jobs=-1)
            return scores.mean()
        
        return self._run_optimization(objective, "SVR")
    
    def optimize_gradient_boosting(self,
                                  X: np.ndarray,
                                  y: np.ndarray,
                                  cv_folds: int = 5,
                                  scoring: str = 'neg_mean_squared_error') -> OptimizationResult:
        """
        Optimize Gradient Boosting hyperparameters.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target values
        cv_folds : int
            Number of cross-validation folds
        scoring : str
            Scoring metric
            
        Returns:
        --------
        OptimizationResult
            Optimization results
        """
        def objective(trial):
            # Suggest hyperparameters
            n_estimators = trial.suggest_int('n_estimators', 50, 500)
            learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
            max_depth = trial.suggest_int('max_depth', 3, 10)
            min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
            min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
            subsample = trial.suggest_float('subsample', 0.5, 1.0)
            
            # Create model
            from sklearn.ensemble import GradientBoostingRegressor
            model = GradientBoostingRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                subsample=subsample,
                random_state=self.random_state
            )
            
            # Cross-validation
            from sklearn.model_selection import cross_val_score
            scores = cross_val_score(model, X, y, cv=cv_folds, scoring=scoring, n_jobs=-1)
            return scores.mean()
        
        return self._run_optimization(objective, "GradientBoosting")
    
    def _run_optimization(self, objective: Callable, study_name: str) -> OptimizationResult:
        """Run the optimization process."""
        start_time = time.time()
        
        # Create study
        self.study = self.optuna.create_study(
            direction=self.direction,
            study_name=study_name,
            pruner=self.pruner,
            sampler=self.sampler
        )
        
        # Run optimization
        self.study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout
        )
        
        optimization_time = time.time() - start_time
        
        return OptimizationResult(
            best_params=self.study.best_params,
            best_score=self.study.best_value,
            best_trial=self.study.best_trial.number,
            optimization_time=optimization_time,
            n_trials=len(self.study.trials),
            study=self.study
        )


def create_optuna_study(study_name: str,
                       direction: str = 'minimize',
                       pruner: str = 'median',
                       sampler: str = 'tpe',
                       random_state: int = 42) -> Any:
    """
    Create an Optuna study for hyperparameter optimization.
    
    Parameters:
    -----------
    study_name : str
        Name of the study
    direction : str
        Optimization direction
    pruner : str
        Pruning strategy
    sampler : str
        Sampling strategy
    random_state : int
        Random state
        
    Returns:
    --------
    optuna.Study
        Created study
    """
    optuna, MedianPruner, TPESampler = _lazy_import_optuna()
    
    if optuna is None:
        raise ImportError("Optuna not available")
    
    # Setup pruner
    if pruner == 'median':
        pruner_obj = MedianPruner()
    elif pruner == 'percentile':
        pruner_obj = optuna.pruners.PercentilePruner(25.0)
    elif pruner == 'successive_halving':
        pruner_obj = optuna.pruners.SuccessiveHalvingPruner()
    else:
        pruner_obj = None
    
    # Setup sampler
    if sampler == 'tpe':
        sampler_obj = TPESampler(seed=random_state)
    elif sampler == 'random':
        sampler_obj = optuna.samplers.RandomSampler(seed=random_state)
    elif sampler == 'cmaes':
        sampler_obj = optuna.samplers.CmaEsSampler(seed=random_state)
    else:
        sampler_obj = None
    
    return optuna.create_study(
        direction=direction,
        study_name=study_name,
        pruner=pruner_obj,
        sampler=sampler_obj
    )


def optimize_hyperparameters(estimator_type: str,
                            X: np.ndarray,
                            y: np.ndarray,
                            n_trials: int = 100,
                            timeout: Optional[float] = None,
                            cv_folds: int = 5,
                            scoring: str = 'neg_mean_squared_error',
                            random_state: int = 42) -> OptimizationResult:
    """
    Optimize hyperparameters for a specific estimator type.
    
    Parameters:
    -----------
    estimator_type : str
        Type of estimator ('random_forest', 'svr', 'gradient_boosting')
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target values
    n_trials : int
        Number of optimization trials
    timeout : float, optional
        Timeout in seconds
    cv_folds : int
        Number of cross-validation folds
    scoring : str
        Scoring metric
    random_state : int
        Random state
        
    Returns:
    --------
    OptimizationResult
        Optimization results
    """
    optimizer = OptunaOptimizer(
        study_name=f"{estimator_type}_optimization",
        n_trials=n_trials,
        timeout=timeout,
        random_state=random_state
    )
    
    if estimator_type == 'random_forest':
        return optimizer.optimize_random_forest(X, y, cv_folds, scoring)
    elif estimator_type == 'svr':
        return optimizer.optimize_svr(X, y, cv_folds, scoring)
    elif estimator_type == 'gradient_boosting':
        return optimizer.optimize_gradient_boosting(X, y, cv_folds, scoring)
    else:
        raise ValueError(f"Unknown estimator type: {estimator_type}")


def optimize_all_estimators(X: np.ndarray,
                           y: np.ndarray,
                           estimator_types: Optional[List[str]] = None,
                           n_trials: int = 100,
                           timeout: Optional[float] = None,
                           cv_folds: int = 5,
                           scoring: str = 'neg_mean_squared_error',
                           random_state: int = 42) -> Dict[str, OptimizationResult]:
    """
    Optimize hyperparameters for all estimator types.
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target values
    estimator_types : List[str], optional
        Types of estimators to optimize
    n_trials : int
        Number of optimization trials
    timeout : float, optional
        Timeout in seconds
    cv_folds : int
        Number of cross-validation folds
    scoring : str
        Scoring metric
    random_state : int
        Random state
        
    Returns:
    --------
    Dict[str, OptimizationResult]
        Optimization results for each estimator
    """
    if estimator_types is None:
        estimator_types = ['random_forest', 'svr', 'gradient_boosting']
    
    results = {}
    
    for estimator_type in estimator_types:
        try:
            result = optimize_hyperparameters(
                estimator_type, X, y, n_trials, timeout, cv_folds, scoring, random_state
            )
            results[estimator_type] = result
        except Exception as e:
            warnings.warn(f"Failed to optimize {estimator_type}: {e}")
    
    return results


def create_optimized_estimators(X: np.ndarray,
                               y: np.ndarray,
                               estimator_types: Optional[List[str]] = None,
                               n_trials: int = 100,
                               timeout: Optional[float] = None,
                               cv_folds: int = 5,
                               scoring: str = 'neg_mean_squared_error',
                               random_state: int = 42) -> Dict[str, Any]:
    """
    Create optimized estimators with best hyperparameters.
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target values
    estimator_types : List[str], optional
        Types of estimators to optimize
    n_trials : int
        Number of optimization trials
    timeout : float, optional
        Timeout in seconds
    cv_folds : int
        Number of cross-validation folds
    scoring : str
        Scoring metric
    random_state : int
        Random state
        
    Returns:
    --------
    Dict[str, Any]
        Optimized estimators
    """
    # Optimize hyperparameters
    optimization_results = optimize_all_estimators(
        X, y, estimator_types, n_trials, timeout, cv_folds, scoring, random_state
    )
    
    # Create optimized estimators
    optimized_estimators = {}
    
    for estimator_type, result in optimization_results.items():
        try:
            if estimator_type == 'random_forest':
                from sklearn.ensemble import RandomForestRegressor
                estimator = RandomForestRegressor(**result.best_params, random_state=random_state)
            elif estimator_type == 'svr':
                from sklearn.svm import SVR
                estimator = SVR(**result.best_params)
            elif estimator_type == 'gradient_boosting':
                from sklearn.ensemble import GradientBoostingRegressor
                estimator = GradientBoostingRegressor(**result.best_params, random_state=random_state)
            else:
                continue
            
            optimized_estimators[estimator_type] = estimator
            
        except Exception as e:
            warnings.warn(f"Failed to create optimized {estimator_type}: {e}")
    
    return optimized_estimators
