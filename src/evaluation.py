"""
Evaluation Module
=================

This module contains functions for model evaluation and visualization.
Use these functions in notebook 04_model_evaluation.ipynb

Functions:
----------
- evaluate_model()          : Comprehensive model evaluation
- plot_confusion_matrix()   : Visualize confusion matrix
- plot_roc_curve()          : Plot ROC curve
- compare_models()          : Compare multiple models
- find_optimal_threshold()  : Find threshold that maximizes recall

IMPORTANT - Medical Context:
----------------------------
For NSMP endometrial cancer risk prediction:
- We use a "Traffic Light" system: Low (Green), Intermediate (Yellow), High (Red)
- False Negatives are CRITICAL (missing a recurrence)
- Consider adjusting classification threshold to maximize Sensitivity

Usage:
------
    from src.evaluation import evaluate_model, find_optimal_threshold
    
    metrics = evaluate_model(model, X_test, y_test)
    optimal_thresh = find_optimal_threshold(model, X_test, y_test, target_recall=0.90)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score
)


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5,
    display: bool = True
) -> Dict:
    """
    Comprehensive model evaluation with medical context.
    
    Parameters
    ----------
    model : sklearn model
        Trained model
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        True labels
    threshold : float
        Classification threshold (default 0.5, lower for higher sensitivity)
    display : bool
        Whether to print results
    
    Returns
    -------
    dict
        Dictionary with all metrics
    
    Example
    -------
        metrics = evaluate_model(model, X_test, y_test, threshold=0.3)
    """
    # Get predictions
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),  # SENSITIVITY
        'specificity': recall_score(y_test, y_pred, pos_label=0, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'threshold_used': threshold
    }
    
    # Confusion matrix components
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    metrics['true_negatives'] = int(tn)
    metrics['false_positives'] = int(fp)
    metrics['false_negatives'] = int(fn)  # CRITICAL - these are missed recurrences!
    metrics['true_positives'] = int(tp)
    
    if display:
        print("=" * 50)
        print("📊 MODEL EVALUATION RESULTS")
        print("=" * 50)
        print(f"\n🎯 Classification Threshold: {threshold}")
        print(f"\n📈 Key Metrics:")
        print(f"   Accuracy:    {metrics['accuracy']:.3f}")
        print(f"   Precision:   {metrics['precision']:.3f}")
        print(f"   🔴 RECALL:   {metrics['recall']:.3f}  ← CRITICAL for medical")
        print(f"   Specificity: {metrics['specificity']:.3f}")
        print(f"   F1 Score:    {metrics['f1']:.3f}")
        print(f"   ROC AUC:     {metrics['roc_auc']:.3f}")
        print(f"\n⚠️ Confusion Matrix:")
        print(f"   True Negatives:  {tn}")
        print(f"   False Positives: {fp}  (overtreatment)")
        print(f"   🚨 FALSE NEGATIVES: {fn}  (MISSED RECURRENCES!)")
        print(f"   True Positives:  {tp}")
        print("=" * 50)
    
    return metrics


def find_optimal_threshold(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    target_recall: float = 0.90,
    min_precision: float = 0.10
) -> Tuple[float, Dict]:
    """
    Find classification threshold that achieves target recall.
    
    For medical applications where missing a positive case is critical,
    we want to maximize recall while maintaining acceptable precision.
    
    Parameters
    ----------
    model : sklearn model
        Trained model
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        True labels
    target_recall : float
        Minimum recall to achieve (default 0.90 = catch 90% of recurrences)
    min_precision : float
        Minimum acceptable precision
    
    Returns
    -------
    tuple
        (optimal threshold, metrics at that threshold)
    
    Example
    -------
        threshold, metrics = find_optimal_threshold(model, X_test, y_test, target_recall=0.95)
        print(f"Use threshold {threshold:.2f} to catch 95% of recurrences")
    """
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Try different thresholds
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_threshold = 0.5
    best_f1 = 0
    
    print(f"🔍 Searching for threshold with recall ≥ {target_recall}")
    print("-" * 60)
    print(f"{'Threshold':>10} {'Recall':>10} {'Precision':>10} {'F1':>10}")
    print("-" * 60)
    
    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        recall = recall_score(y_test, y_pred, zero_division=0)
        precision = precision_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        print(f"{thresh:>10.2f} {recall:>10.3f} {precision:>10.3f} {f1:>10.3f}")
        
        if recall >= target_recall and precision >= min_precision:
            if f1 > best_f1:
                best_threshold = thresh
                best_f1 = f1
    
    print("-" * 60)
    print(f"✅ Optimal threshold: {best_threshold:.2f}")
    
    # Get metrics at optimal threshold
    metrics = evaluate_model(model, X_test, y_test, threshold=best_threshold, display=False)
    
    return best_threshold, metrics


def plot_confusion_matrix(
    y_true: pd.Series,
    y_pred: np.ndarray,
    labels: List[str] = ['No Recurrence', 'Recurrence'],
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Plot confusion matrix with medical context annotations.
    
    Parameters
    ----------
    y_true : pd.Series
        True labels
    y_pred : np.ndarray
        Predicted labels
    labels : list
        Class labels
    figsize : tuple
        Figure size
    
    Returns
    -------
    matplotlib.Figure
        The figure object
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=labels, yticklabels=labels,
        ax=ax
    )
    
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title('Confusion Matrix\n(⚠️ Bottom-left = Missed Recurrences!)', fontsize=14)
    
    plt.tight_layout()
    return fig


def plot_roc_curve(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Plot ROC curve with AUC score.
    
    Parameters
    ----------
    model : sklearn model
        Trained model
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        True labels
    figsize : tuple
        Figure size
    
    Returns
    -------
    matplotlib.Figure
        The figure object
    """
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    auc_score = roc_auc_score(y_test, y_proba)
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC Curve (AUC = {auc_score:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    ax.set_ylabel('True Positive Rate (Sensitivity/Recall)', fontsize=12)
    ax.set_title('ROC Curve', fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def compare_models(
    models: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> pd.DataFrame:
    """
    Compare multiple models side by side.
    
    Parameters
    ----------
    models : dict
        Dictionary of {model_name: trained_model}
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        True labels
    
    Returns
    -------
    pd.DataFrame
        Comparison table with metrics for each model
    
    Example
    -------
        models = {
            'XGBoost': xgb_model,
            'Random Forest': rf_model,
            'Logistic Regression': lr_model
        }
        comparison = compare_models(models, X_test, y_test)
    """
    results = []
    
    for name, model in models.items():
        metrics = evaluate_model(model, X_test, y_test, display=False)
        metrics['model'] = name
        results.append(metrics)
    
    comparison_df = pd.DataFrame(results)
    comparison_df = comparison_df.set_index('model')
    
    # Sort by recall (most important for medical)
    comparison_df = comparison_df.sort_values('recall', ascending=False)
    
    print("\n📊 MODEL COMPARISON (sorted by Recall)")
    print("=" * 80)
    print(comparison_df[['recall', 'precision', 'f1', 'roc_auc', 'false_negatives']].to_string())
    print("=" * 80)
    
    return comparison_df
