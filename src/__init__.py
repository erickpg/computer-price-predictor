"""
src - Computer Price Predictor Library

This package contains shared modules for the computer price prediction pipeline:
- features.py: Feature engineering functions
- modeling.py: ML pipeline construction and training
- backend_api.py: API wrapper for the Streamlit frontend
"""

from . import features
from . import modeling
from . import backend_api
