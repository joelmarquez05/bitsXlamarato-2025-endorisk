"""
NEST - Source Code Module
=========================

This package contains reusable code for the NEST project.
Import functions from here in your Jupyter notebooks to avoid code duplication.

Usage in notebooks:
-------------------
    import sys
    sys.path.append('..')  # Add parent directory to path
    
    from src.data_loader import load_patient_data
    from src.preprocessing import preprocess_features
    from src.models import train_xgboost_model

Modules:
--------
- data_loader.py    : Functions to load and validate raw data
- preprocessing.py  : Data cleaning and transformation functions
- features.py       : Feature engineering functions
- models.py         : Model training and prediction functions
- evaluation.py     : Metrics, plots, and model evaluation tools

Authors: Team Pau Overfitting
Created: December 2024
"""

# Version of the src package
__version__ = "0.1.0"

# Make key functions available at package level for convenience
# (These will be populated as we build the modules)
