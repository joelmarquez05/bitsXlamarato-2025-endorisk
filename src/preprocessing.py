"""
Preprocessing Module
====================

This module contains functions for data cleaning and transformation.
Use these functions in notebook 02_data_preprocessing.ipynb

Functions:
----------
- clean_missing_values()    : Handle missing data
- encode_categorical()      : Encode categorical variables
- scale_numerical()         : Normalize/standardize numerical features
- save_preprocessed_data()  : Save processed data for next notebook

Usage:
------
    from src.preprocessing import clean_missing_values, encode_categorical
    
    df_clean = clean_missing_values(df, strategy='median')
    df_encoded, encoder = encode_categorical(df_clean, columns=['grade', 'histology'])
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional, Dict
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer


def clean_missing_values(
    df: pd.DataFrame, 
    numerical_strategy: str = 'median',
    categorical_strategy: str = 'most_frequent',
    drop_threshold: float = 0.5
) -> pd.DataFrame:
    """
    Handle missing values in the DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data
    numerical_strategy : str
        Strategy for numerical columns ('mean', 'median', 'drop')
    categorical_strategy : str
        Strategy for categorical columns ('most_frequent', 'drop', 'unknown')
    drop_threshold : float
        Drop columns with more than this fraction of missing values
    
    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame
    """
    df = df.copy()
    
    # Drop columns with too many missing values
    missing_ratio = df.isnull().mean()
    cols_to_drop = missing_ratio[missing_ratio > drop_threshold].index.tolist()
    if cols_to_drop:
        print(f"⚠️ Dropping columns with >{drop_threshold*100}% missing: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)
    
    # Separate numerical and categorical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Handle numerical missing values
    if numerical_cols and numerical_strategy != 'drop':
        imputer = SimpleImputer(strategy=numerical_strategy)
        df[numerical_cols] = imputer.fit_transform(df[numerical_cols])
    
    # Handle categorical missing values
    if categorical_cols:
        if categorical_strategy == 'unknown':
            df[categorical_cols] = df[categorical_cols].fillna('Unknown')
        elif categorical_strategy == 'most_frequent':
            imputer = SimpleImputer(strategy='most_frequent')
            df[categorical_cols] = imputer.fit_transform(df[categorical_cols])
    
    print(f"✅ Cleaned missing values. Remaining nulls: {df.isnull().sum().sum()}")
    return df


def encode_categorical(
    df: pd.DataFrame, 
    columns: List[str],
    method: str = 'onehot'
) -> Tuple[pd.DataFrame, dict]:
    """
    Encode categorical variables.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data
    columns : list
        Columns to encode
    method : str
        Encoding method ('onehot', 'label', 'ordinal')
    
    Returns
    -------
    tuple
        (encoded DataFrame, dict of encoders for saving)
    """
    df = df.copy()
    encoders = {}
    
    for col in columns:
        if col not in df.columns:
            print(f"⚠️ Column '{col}' not found, skipping")
            continue
            
        if method == 'label':
            encoder = LabelEncoder()
            df[col] = encoder.fit_transform(df[col].astype(str))
            encoders[col] = encoder
            
        elif method == 'onehot':
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df.drop(col, axis=1), dummies], axis=1)
            encoders[col] = list(dummies.columns)
    
    print(f"✅ Encoded {len(columns)} categorical columns using '{method}' method")
    return df, encoders


def scale_numerical(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = 'standard'
) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Scale numerical features.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data
    columns : list, optional
        Columns to scale. If None, scales all numerical columns.
    method : str
        Scaling method ('standard', 'minmax')
    
    Returns
    -------
    tuple
        (scaled DataFrame, fitted scaler for saving)
    """
    df = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if method == 'standard':
        scaler = StandardScaler()
    else:
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
    
    df[columns] = scaler.fit_transform(df[columns])
    print(f"✅ Scaled {len(columns)} numerical columns using '{method}' method")
    
    return df, scaler


def save_preprocessed_data(
    df: pd.DataFrame,
    version: str,
    artifacts: Optional[Dict] = None
) -> None:
    """
    Save preprocessed data and artifacts (scalers, encoders) for the next notebook.
    
    Parameters
    ----------
    df : pd.DataFrame
        Processed data to save
    version : str
        Version identifier (e.g., 'v1', 'v2')
    artifacts : dict, optional
        Dictionary of preprocessing artifacts (scalers, encoders) to save
    
    Example
    -------
        save_preprocessed_data(
            df_processed, 
            version='v1',
            artifacts={'scaler': scaler, 'encoders': encoders}
        )
    """
    base_dir = Path(__file__).parent.parent
    
    # Save DataFrame
    data_path = base_dir / "data" / "processed" / f"features_{version}.csv"
    df.to_csv(data_path, index=False)
    print(f"✅ Saved processed data: {data_path}")
    
    # Save artifacts
    if artifacts:
        for name, artifact in artifacts.items():
            artifact_path = base_dir / "models" / f"{name}_{version}.joblib"
            joblib.dump(artifact, artifact_path)
            print(f"✅ Saved artifact: {artifact_path}")
