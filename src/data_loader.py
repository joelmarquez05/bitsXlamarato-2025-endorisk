"""
Data Loader Module
==================

This module contains functions to load and validate raw data.
Use these functions in notebook 01_exploratory_analysis.ipynb

Functions:
----------
- load_patient_data()     : Load patient CSV/Excel data
- load_processed_data()   : Load preprocessed data from previous notebook
- validate_data()         : Check for missing columns, data types, etc.

Usage:
------
    from src.data_loader import load_patient_data, validate_data
    
    df = load_patient_data('data/raw/patients.csv')
    is_valid, errors = validate_data(df)
"""

import pandas as pd
from pathlib import Path
from typing import Tuple, List, Optional
import warnings


def load_patient_data(filepath: str, **kwargs) -> pd.DataFrame:
    """
    Load patient data from CSV or Excel file.
    
    Parameters
    ----------
    filepath : str
        Path to the data file (supports .csv, .xlsx, .xls)
    **kwargs : dict
        Additional arguments passed to pd.read_csv or pd.read_excel
    
    Returns
    -------
    pd.DataFrame
        Loaded data
    
    Example
    -------
        df = load_patient_data('../data/raw/patients.csv')
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    if filepath.suffix == '.csv':
        df = pd.read_csv(filepath, **kwargs)
    elif filepath.suffix in ['.xlsx', '.xls']:
        df = pd.read_excel(filepath, **kwargs)
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")
    
    print(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns from {filepath.name}")
    return df


def load_processed_data(version: str = "latest") -> pd.DataFrame:
    """
    Load preprocessed data from data/processed/ folder.
    
    Parameters
    ----------
    version : str
        Version identifier (e.g., 'v1', 'latest')
    
    Returns
    -------
    pd.DataFrame
        Processed data ready for modeling
    
    Example
    -------
        df = load_processed_data('v1')  # Loads features_v1.csv
    """
    processed_dir = Path(__file__).parent.parent / "data" / "processed"
    
    if version == "latest":
        # Find the most recent file
        csv_files = list(processed_dir.glob("features_*.csv"))
        if not csv_files:
            raise FileNotFoundError("No processed data files found")
        filepath = max(csv_files, key=lambda x: x.stat().st_mtime)
    else:
        filepath = processed_dir / f"features_{version}.csv"
    
    if not filepath.exists():
        raise FileNotFoundError(f"Processed data not found: {filepath}")
    
    df = pd.read_csv(filepath)
    print(f"✅ Loaded processed data: {filepath.name} ({len(df)} rows)")
    return df


def validate_data(df: pd.DataFrame, required_columns: Optional[List[str]] = None) -> Tuple[bool, List[str]]:
    """
    Validate DataFrame for common issues.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data to validate
    required_columns : list, optional
        List of columns that must be present
    
    Returns
    -------
    tuple
        (is_valid: bool, errors: list of error messages)
    
    Example
    -------
        is_valid, errors = validate_data(df, required_columns=['age', 'grade'])
        if not is_valid:
            print("Errors:", errors)
    """
    errors = []
    
    # Check for empty DataFrame
    if len(df) == 0:
        errors.append("DataFrame is empty")
    
    # Check required columns
    if required_columns:
        missing = set(required_columns) - set(df.columns)
        if missing:
            errors.append(f"Missing required columns: {missing}")
    
    # Check for completely empty columns
    empty_cols = df.columns[df.isnull().all()].tolist()
    if empty_cols:
        errors.append(f"Completely empty columns: {empty_cols}")
    
    # Check for duplicate rows
    n_duplicates = df.duplicated().sum()
    if n_duplicates > 0:
        warnings.warn(f"Found {n_duplicates} duplicate rows")
    
    is_valid = len(errors) == 0
    return is_valid, errors
