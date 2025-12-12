"""
Models Module
=============

This module contains functions for model training, tuning, and saving.
Use these functions in notebook 03_model_training.ipynb

Functions:
----------
- train_model()           : Train a classification model
- tune_hyperparameters()  : Hyperparameter tuning with cross-validation
- save_model()            : Save trained model with metadata
- load_model()            : Load a trained model

IMPORTANT - Medical Context:
----------------------------
For NSMP endometrial cancer, we prioritize SENSITIVITY (Recall) because:
- False Negative (missing a recurrence) = FATAL
- False Positive (overtreatment) = Toxicity, but survivable

Usage:
------
    from src.models import train_model, save_model
    
    model, metrics = train_model(X_train, y_train, model_type='xgboost')
    save_model(model, version='v1', metadata={'features': list(X_train.columns)})
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
from datetime import datetime
import joblib
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str = 'xgboost',
    optimize_for: str = 'recall',
    **kwargs
) -> Tuple[Any, Dict]:
    """
    Train a classification model.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training labels
    model_type : str
        Type of model: 'xgboost', 'random_forest', 'logistic_regression'
    optimize_for : str
        Metric to optimize: 'recall' (recommended for medical), 'f1', 'accuracy'
    **kwargs : dict
        Additional parameters for the model
    
    Returns
    -------
    tuple
        (trained model, dict of cross-validation metrics)
    
    Example
    -------
        model, metrics = train_model(X_train, y_train, model_type='xgboost')
        print(f"CV Recall: {metrics['recall_mean']:.3f}")
    """
    # Select model
    if model_type == 'xgboost':
        from xgboost import XGBClassifier
        default_params = {
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1,
            'scale_pos_weight': 1,  # Adjust for class imbalance
            'random_state': 42,
            'eval_metric': 'logloss'
        }
        default_params.update(kwargs)
        model = XGBClassifier(**default_params)
        
    elif model_type == 'random_forest':
        from sklearn.ensemble import RandomForestClassifier
        default_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'class_weight': 'balanced',  # Handle class imbalance
            'random_state': 42
        }
        default_params.update(kwargs)
        model = RandomForestClassifier(**default_params)
        
    elif model_type == 'logistic_regression':
        from sklearn.linear_model import LogisticRegression
        default_params = {
            'class_weight': 'balanced',
            'max_iter': 1000,
            'random_state': 42
        }
        default_params.update(kwargs)
        model = LogisticRegression(**default_params)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Train model
    model.fit(X_train, y_train)
    
    # Cross-validation metrics
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    metrics = {}
    for metric_name, scorer in [('accuracy', 'accuracy'), ('recall', 'recall'), 
                                  ('precision', 'precision'), ('f1', 'f1'), ('roc_auc', 'roc_auc')]:
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scorer)
        metrics[f'{metric_name}_mean'] = scores.mean()
        metrics[f'{metric_name}_std'] = scores.std()
    
    print(f"✅ Trained {model_type} model")
    print(f"   📊 CV Recall: {metrics['recall_mean']:.3f} ± {metrics['recall_std']:.3f}")
    print(f"   📊 CV AUC: {metrics['roc_auc_mean']:.3f} ± {metrics['roc_auc_std']:.3f}")
    
    return model, metrics


def tune_hyperparameters(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str = 'xgboost',
    param_grid: Optional[Dict] = None,
    scoring: str = 'recall'
) -> Tuple[Any, Dict]:
    """
    Perform hyperparameter tuning using GridSearchCV.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training labels
    model_type : str
        Type of model
    param_grid : dict, optional
        Parameter grid to search. If None, uses default grid.
    scoring : str
        Metric to optimize
    
    Returns
    -------
    tuple
        (best model, best parameters)
    """
    from sklearn.model_selection import GridSearchCV
    
    # Default parameter grids
    default_grids = {
        'xgboost': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2]
        },
        'random_forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10]
        }
    }
    
    if param_grid is None:
        param_grid = default_grids.get(model_type, {})
    
    # Get base model
    base_model, _ = train_model(X_train, y_train, model_type=model_type)
    
    # Grid search
    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=5,
        scoring=scoring,
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)
    
    print(f"✅ Best parameters: {grid_search.best_params_}")
    print(f"   📊 Best {scoring}: {grid_search.best_score_:.3f}")
    
    return grid_search.best_estimator_, grid_search.best_params_


def save_model(
    model: Any,
    version: str,
    metadata: Optional[Dict] = None,
    models_dir: Optional[Path] = None
) -> Path:
    """
    Save trained model with metadata.
    
    Parameters
    ----------
    model : sklearn model
        Trained model to save
    version : str
        Version identifier (e.g., 'v1', 'final')
    metadata : dict, optional
        Additional metadata (features, metrics, etc.)
    models_dir : Path, optional
        Directory to save model. Defaults to project's models/ folder.
    
    Returns
    -------
    Path
        Path to saved model file
    
    Example
    -------
        save_model(model, version='v1', metadata={
            'features': list(X_train.columns),
            'metrics': metrics,
            'threshold': 0.3
        })
    """
    if models_dir is None:
        models_dir = Path(__file__).parent.parent / "models"
    
    models_dir.mkdir(exist_ok=True)
    
    # Prepare metadata
    full_metadata = {
        'version': version,
        'saved_at': datetime.now().isoformat(),
        'model_type': type(model).__name__,
    }
    if metadata:
        full_metadata.update(metadata)
    
    # Save model
    model_path = models_dir / f"model_{version}.joblib"
    joblib.dump(model, model_path)
    print(f"✅ Saved model: {model_path}")
    
    # Save metadata
    metadata_path = models_dir / f"model_{version}_metadata.joblib"
    joblib.dump(full_metadata, metadata_path)
    print(f"✅ Saved metadata: {metadata_path}")
    
    return model_path


def load_model(version: str = "latest", models_dir: Optional[Path] = None) -> Tuple[Any, Dict]:
    """
    Load a trained model and its metadata.
    
    Parameters
    ----------
    version : str
        Version to load, or 'latest' for most recent
    models_dir : Path, optional
        Directory containing models
    
    Returns
    -------
    tuple
        (model, metadata dict)
    
    Example
    -------
        model, metadata = load_model('v1')
        print(f"Loaded model trained on {metadata['saved_at']}")
    """
    if models_dir is None:
        models_dir = Path(__file__).parent.parent / "models"
    
    if version == "latest":
        model_files = list(models_dir.glob("model_*.joblib"))
        model_files = [f for f in model_files if 'metadata' not in f.name]
        if not model_files:
            raise FileNotFoundError("No model files found")
        model_path = max(model_files, key=lambda x: x.stat().st_mtime)
        version = model_path.stem.replace('model_', '')
    else:
        model_path = models_dir / f"model_{version}.joblib"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model = joblib.load(model_path)
    
    # Load metadata
    metadata_path = models_dir / f"model_{version}_metadata.joblib"
    if metadata_path.exists():
        metadata = joblib.load(metadata_path)
    else:
        metadata = {'version': version}
    
    print(f"✅ Loaded model: {model_path.name}")
    return model, metadata
