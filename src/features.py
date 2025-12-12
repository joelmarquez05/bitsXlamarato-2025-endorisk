"""
Feature Engineering Module
==========================

This module contains functions for creating and selecting features.
Use these functions in notebook 02_data_preprocessing.ipynb

Functions:
----------
- create_clinical_features()   : Create domain-specific features for NSMP
- select_features()            : Feature selection using various methods
- get_feature_importance()     : Extract feature importance from models

Usage:
------
    from src.features import create_clinical_features, select_features
    
    df_features = create_clinical_features(df)
    selected_cols = select_features(df_features, target='recurrence', method='correlation')
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif


def create_clinical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create domain-specific clinical features for endometrial cancer prediction.
    
    This function creates features based on medical knowledge about NSMP 
    endometrial cancer risk factors.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data with raw clinical variables
    
    Returns
    -------
    pd.DataFrame
        DataFrame with additional engineered features
    
    Notes
    -----
    Key NSMP risk factors to consider:
    - Tumor grade (Grade 3 = higher risk)
    - LVSI (Lymphovascular Space Invasion)
    - Myometrial invasion depth
    - Tumor size
    - Patient age
    
    Example
    -------
        df_features = create_clinical_features(df)
    """
    df = df.copy()
    
    # TODO: Add domain-specific feature engineering based on actual data columns
    # Examples of features that could be created:
    
    # Age-related features
    if 'age' in df.columns:
        df['age_group'] = pd.cut(
            df['age'], 
            bins=[0, 50, 60, 70, 100], 
            labels=['<50', '50-60', '60-70', '>70']
        )
        df['is_elderly'] = (df['age'] > 65).astype(int)
    
    # Risk score combination (if relevant columns exist)
    # df['high_risk_factors'] = (
    #     (df['grade'] == 3).astype(int) +
    #     (df['lvsi'] == 'Yes').astype(int) +
    #     (df['myometrial_invasion'] > 0.5).astype(int)
    # )
    
    print(f"✅ Created clinical features. New shape: {df.shape}")
    return df


def select_features(
    df: pd.DataFrame,
    target_column: str,
    method: str = 'correlation',
    n_features: int = 10,
    threshold: float = 0.1
) -> List[str]:
    """
    Select most important features using various methods.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data including target variable
    target_column : str
        Name of the target variable
    method : str
        Selection method: 'correlation', 'mutual_info', 'f_test'
    n_features : int
        Number of top features to select (for ranking methods)
    threshold : float
        Minimum correlation/importance threshold
    
    Returns
    -------
    list
        List of selected feature names
    
    Example
    -------
        selected = select_features(df, target='recurrence', method='mutual_info', n_features=15)
        df_selected = df[selected + ['recurrence']]
    """
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Only use numerical columns for selection
    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    X_numerical = X[numerical_cols]
    
    if method == 'correlation':
        # Correlation-based selection
        correlations = X_numerical.corrwith(y).abs()
        selected = correlations[correlations > threshold].nlargest(n_features).index.tolist()
        
    elif method == 'mutual_info':
        # Mutual information for classification
        mi_scores = mutual_info_classif(X_numerical.fillna(0), y, random_state=42)
        mi_series = pd.Series(mi_scores, index=numerical_cols)
        selected = mi_series.nlargest(n_features).index.tolist()
        
    elif method == 'f_test':
        # F-test (ANOVA) for classification
        selector = SelectKBest(f_classif, k=min(n_features, len(numerical_cols)))
        selector.fit(X_numerical.fillna(0), y)
        mask = selector.get_support()
        selected = X_numerical.columns[mask].tolist()
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    print(f"✅ Selected {len(selected)} features using '{method}' method")
    return selected


def get_feature_importance(
    model,
    feature_names: List[str],
    top_n: int = 20
) -> pd.DataFrame:
    """
    Extract and display feature importance from a trained model.
    
    Parameters
    ----------
    model : sklearn model
        Trained model with feature_importances_ attribute
    feature_names : list
        List of feature names
    top_n : int
        Number of top features to return
    
    Returns
    -------
    pd.DataFrame
        DataFrame with feature importance scores, sorted descending
    
    Example
    -------
        importance_df = get_feature_importance(model, X_train.columns, top_n=15)
        print(importance_df)
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_).flatten()
    else:
        raise ValueError("Model doesn't have feature_importances_ or coef_ attribute")
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    return importance_df.head(top_n)
