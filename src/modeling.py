"""
modeling.py - Machine Learning Pipeline Module

This module contains functions for building, training, and evaluating
the computer price prediction model using scikit-learn pipelines.

Key Design Decisions:
- Use sklearn Pipeline for reproducible preprocessing + modeling
- Primary models: RandomForestRegressor, GradientBoostingRegressor
- Evaluation metrics: RMSE (primary), MAE (secondary)
- Cross-validation for robust performance estimation
- Export trained pipeline to models/price_model.pkl for frontend use
"""

import pandas as pd
import numpy as np
from typing import List, Tuple

# TODO: Uncomment when implementing
# from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.impute import SimpleImputer
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# from sklearn.model_selection import cross_val_score
# from sklearn.metrics import mean_squared_error, mean_absolute_error
# import joblib


# =============================================================================
# COLUMN DEFINITIONS
# =============================================================================

# TODO: Define after EDA confirms which features are most useful
# These should include BOTH original columns (with Spanish names) and
# engineered features (prefixed with _)

NUMERIC_COLS: List[str] = [
    # Engineered features (to be populated after feature engineering)
    # '_cpu_mark',
    # '_gpu_mark',
    # '_ram_gb',
    # '_ssd_gb',
    # '_tamano_pantalla_pulgadas',
    #
    # Original numeric columns (to be identified in EDA)
    # 'Procesador_Número de núcleos del procesador',
    # 'Alimentación_Vatios-hora',
    # ... etc
]

CATEGORICAL_COLS: List[str] = [
    # Likely candidates (to be confirmed in EDA)
    # 'Tipo de producto',
    # 'Serie',
    # 'Marca',  # Need to extract from title or another column
    # 'Procesador_Fabricante del procesador',
    # 'Gráfica_Fabricante de la tarjeta gráfica',
    # ... etc
]

# Target column (price)
# Note: We'll use the engineered '_precio_num' as target, not the raw 'Precio_Rango' string
TARGET_COL: str = '_precio_num'


# =============================================================================
# PIPELINE CONSTRUCTION
# =============================================================================

def construir_pipeline_modelo(modelo: str = 'random_forest') -> 'Pipeline':
    """
    Build a complete sklearn Pipeline for price prediction.

    The pipeline includes:
    1. Preprocessing (using ColumnTransformer):
       - Numeric features: Imputation (median) + StandardScaler
       - Categorical features: Imputation (most_frequent) + OneHotEncoder
    2. Regression model (RandomForest or GradientBoosting)

    Parameters
    ----------
    modelo : str, default='random_forest'
        Which regression model to use:
        - 'random_forest': RandomForestRegressor
        - 'gradient_boosting': GradientBoostingRegressor

    Returns
    -------
    sklearn.pipeline.Pipeline
        A fitted-ready pipeline that can be used with .fit() and .predict()

    Notes
    -----
    - Categorical encoding uses handle_unknown='ignore' for robustness
    - Numeric imputation uses median (robust to outliers)
    - Consider sparse_threshold for one-hot encoding if many categories

    TODO:
    - Implement the full pipeline construction
    - Add hyperparameter tuning capability
    - Consider adding feature selection step
    - Add polynomial features for numeric columns (optional)

    Example Usage (after implementation)
    ------------------------------------
    >>> from src.modeling import construir_pipeline_modelo
    >>> pipeline = construir_pipeline_modelo('gradient_boosting')
    >>> pipeline.fit(X_train, y_train)
    >>> predictions = pipeline.predict(X_test)
    """
    # TODO: Implement pipeline construction
    #
    # numeric_transformer = Pipeline(steps=[
    #     ('imputer', SimpleImputer(strategy='median')),
    #     ('scaler', StandardScaler())
    # ])
    #
    # categorical_transformer = Pipeline(steps=[
    #     ('imputer', SimpleImputer(strategy='most_frequent')),
    #     ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    # ])
    #
    # preprocessor = ColumnTransformer(
    #     transformers=[
    #         ('num', numeric_transformer, NUMERIC_COLS),
    #         ('cat', categorical_transformer, CATEGORICAL_COLS)
    #     ]
    # )
    #
    # if modelo == 'random_forest':
    #     regressor = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    # elif modelo == 'gradient_boosting':
    #     regressor = GradientBoostingRegressor(n_estimators=100, random_state=42)
    # else:
    #     raise ValueError(f"Unknown model: {modelo}")
    #
    # pipeline = Pipeline(steps=[
    #     ('preprocessor', preprocessor),
    #     ('regressor', regressor)
    # ])
    #
    # return pipeline
    raise NotImplementedError("construir_pipeline_modelo() not yet implemented")


# =============================================================================
# TRAINING AND EVALUATION
# =============================================================================

def entrenar_modelo(pipeline: 'Pipeline',
                    X: pd.DataFrame,
                    y: pd.Series) -> 'Pipeline':
    """
    Train the pipeline on the provided data.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        The pipeline to train
    X : pd.DataFrame
        Feature matrix (should contain NUMERIC_COLS and CATEGORICAL_COLS)
    y : pd.Series
        Target variable (prices)

    Returns
    -------
    sklearn.pipeline.Pipeline
        The fitted pipeline

    TODO:
    - Implement training with progress logging
    - Add early stopping for GradientBoosting if validation set provided
    """
    # TODO: Implement
    # return pipeline.fit(X, y)
    raise NotImplementedError()


def evaluar_modelo(pipeline: 'Pipeline',
                   X: pd.DataFrame,
                   y: pd.Series,
                   cv: int = 5) -> dict:
    """
    Evaluate the model using cross-validation.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        The pipeline to evaluate (can be fitted or unfitted)
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target variable
    cv : int, default=5
        Number of cross-validation folds

    Returns
    -------
    dict
        Dictionary with evaluation metrics:
        - 'rmse_mean': Mean RMSE across folds
        - 'rmse_std': Std of RMSE across folds
        - 'mae_mean': Mean MAE across folds
        - 'mae_std': Std of MAE across folds

    Notes
    -----
    - Uses negative MSE/MAE internally (sklearn convention) then converts
    - RMSE = sqrt(MSE), which is in the same units as price (€)

    TODO:
    - Implement cross-validation evaluation
    - Add R² score as optional metric
    - Consider stratified CV based on price ranges
    """
    # TODO: Implement
    # mse_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='neg_mean_squared_error')
    # mae_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='neg_mean_absolute_error')
    #
    # rmse_scores = np.sqrt(-mse_scores)
    # mae_scores = -mae_scores
    #
    # return {
    #     'rmse_mean': rmse_scores.mean(),
    #     'rmse_std': rmse_scores.std(),
    #     'mae_mean': mae_scores.mean(),
    #     'mae_std': mae_scores.std()
    # }
    raise NotImplementedError()


# =============================================================================
# MODEL PERSISTENCE
# =============================================================================

def guardar_modelo(pipeline: 'Pipeline', ruta: str = 'models/price_model.pkl') -> None:
    """
    Save the trained pipeline to a pickle file.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        The fitted pipeline to save
    ruta : str, default='models/price_model.pkl'
        Output path for the pickle file

    Notes
    -----
    - Uses joblib for efficient numpy array serialization
    - The saved file can be loaded by the Streamlit frontend

    TODO:
    - Implement with joblib.dump()
    - Add version metadata to the saved file
    - Consider compression for large models
    """
    # TODO: Implement
    # import joblib
    # joblib.dump(pipeline, ruta)
    # print(f"Model saved to {ruta}")
    raise NotImplementedError()


def cargar_modelo(ruta: str = 'models/price_model.pkl') -> 'Pipeline':
    """
    Load a trained pipeline from a pickle file.

    Parameters
    ----------
    ruta : str, default='models/price_model.pkl'
        Path to the pickle file

    Returns
    -------
    sklearn.pipeline.Pipeline
        The loaded pipeline

    TODO:
    - Implement with joblib.load()
    - Add version compatibility checks
    """
    # TODO: Implement
    # import joblib
    # return joblib.load(ruta)
    raise NotImplementedError()
