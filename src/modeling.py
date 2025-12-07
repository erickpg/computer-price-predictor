"""
modeling.py - Machine Learning Pipeline Module

This module contains functions for building, training, and evaluating
the computer price prediction model using scikit-learn pipelines and CatBoost.

Key Features:
- Automatic feature type inference (numeric vs categorical)
- sklearn Pipelines for reproducible preprocessing + modeling
- CatBoost support with native categorical handling and quantile regression
- Cross-validation with multiple metrics (RMSE, MAE, R², MAPE)
- Model persistence with joblib
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional, Union
from pathlib import Path
import warnings

# sklearn imports
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor
)
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error
)
import joblib

# CatBoost import (optional - will fall back gracefully if not installed)
try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    warnings.warn("CatBoost not installed. CatBoost models will not be available.")


# =============================================================================
# CONSTANTS
# =============================================================================

TARGET_COL = '_precio_num'

# Columns that leak the target (should be excluded from features)
LEAKAGE_COLS = [
    'Precio_Rango',  # Original price range string
    '_precio_num',   # Target variable itself
]


# =============================================================================
# FEATURE TYPE INFERENCE
# =============================================================================

def infer_feature_types(df: pd.DataFrame,
                        target_col: str = TARGET_COL,
                        exclude_cols: List[str] = None) -> Tuple[List[str], List[str], List[str]]:
    """
    Automatically infer numeric and categorical columns from a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataframe
    target_col : str
        Name of the target column to exclude from features
    exclude_cols : List[str], optional
        Additional columns to exclude (e.g., leakage columns)

    Returns
    -------
    Tuple[List[str], List[str], List[str]]
        - feature_cols: All feature column names
        - numeric_cols: Numeric feature column names
        - categorical_cols: Categorical feature column names
    """
    if exclude_cols is None:
        exclude_cols = LEAKAGE_COLS
    else:
        exclude_cols = list(set(exclude_cols + LEAKAGE_COLS))

    # Get all columns except target and leakage columns
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    # Separate numeric and categorical
    numeric_cols = []
    categorical_cols = []

    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    return feature_cols, numeric_cols, categorical_cols


def get_feature_summary(df: pd.DataFrame,
                        numeric_cols: List[str],
                        categorical_cols: List[str]) -> pd.DataFrame:
    """
    Generate a summary of features for inspection.

    Returns a DataFrame with column info: dtype, non-null count, unique values, etc.
    """
    summary = []

    for col in numeric_cols:
        summary.append({
            'column': col,
            'type': 'numeric',
            'dtype': str(df[col].dtype),
            'non_null': df[col].notna().sum(),
            'null_pct': df[col].isna().mean() * 100,
            'unique': df[col].nunique(),
            'min': df[col].min() if df[col].notna().any() else None,
            'max': df[col].max() if df[col].notna().any() else None,
        })

    for col in categorical_cols:
        summary.append({
            'column': col,
            'type': 'categorical',
            'dtype': str(df[col].dtype),
            'non_null': df[col].notna().sum(),
            'null_pct': df[col].isna().mean() * 100,
            'unique': df[col].nunique(),
            'min': None,
            'max': None,
        })

    return pd.DataFrame(summary).sort_values(['type', 'column']).reset_index(drop=True)


# =============================================================================
# SKLEARN PIPELINE CONSTRUCTION
# =============================================================================

def build_preprocessor(numeric_cols: List[str],
                       categorical_cols: List[str]) -> ColumnTransformer:
    """
    Build a ColumnTransformer for preprocessing numeric and categorical features.

    Parameters
    ----------
    numeric_cols : List[str]
        List of numeric column names
    categorical_cols : List[str]
        List of categorical column names

    Returns
    -------
    ColumnTransformer
        Preprocessing transformer
    """
    from sklearn.preprocessing import FunctionTransformer

    # Numeric: impute with median, then scale
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Categorical: convert to string, impute with constant, then one-hot encode
    # The string conversion handles mixed types (bool, str, etc.)
    categorical_transformer = Pipeline(steps=[
        ('to_string', FunctionTransformer(lambda X: X.astype(str))),
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])

    transformers = []

    if numeric_cols:
        transformers.append(('num', numeric_transformer, numeric_cols))

    if categorical_cols:
        transformers.append(('cat', categorical_transformer, categorical_cols))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop'  # Drop columns not specified
    )

    return preprocessor


def build_sklearn_pipeline(model_type: str,
                           numeric_cols: List[str],
                           categorical_cols: List[str],
                           **model_params) -> Pipeline:
    """
    Build a complete sklearn Pipeline for price prediction.

    Parameters
    ----------
    model_type : str
        Type of model: 'dummy', 'random_forest', 'gradient_boosting', 'hist_gradient_boosting'
    numeric_cols : List[str]
        List of numeric column names
    categorical_cols : List[str]
        List of categorical column names
    **model_params
        Additional parameters to pass to the regressor

    Returns
    -------
    Pipeline
        Complete preprocessing + model pipeline
    """
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    # Select regressor
    if model_type == 'dummy':
        regressor = DummyRegressor(strategy='mean')
    elif model_type == 'random_forest':
        default_params = {'n_estimators': 100, 'random_state': 42, 'n_jobs': -1, 'max_depth': 20}
        default_params.update(model_params)
        regressor = RandomForestRegressor(**default_params)
    elif model_type == 'gradient_boosting':
        default_params = {'n_estimators': 100, 'random_state': 42, 'max_depth': 5}
        default_params.update(model_params)
        regressor = GradientBoostingRegressor(**default_params)
    elif model_type == 'hist_gradient_boosting':
        default_params = {'max_iter': 100, 'random_state': 42, 'max_depth': 10}
        default_params.update(model_params)
        regressor = HistGradientBoostingRegressor(**default_params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', regressor)
    ])

    return pipeline


# =============================================================================
# CATBOOST MODELS
# =============================================================================

def build_catboost_model(categorical_cols: List[str],
                         loss_function: str = 'RMSE',
                         quantile: float = None,
                         **model_params) -> Optional['CatBoostRegressor']:
    """
    Build a CatBoostRegressor with native categorical handling.

    Parameters
    ----------
    categorical_cols : List[str]
        List of categorical column names (CatBoost handles these natively)
    loss_function : str
        Loss function: 'RMSE', 'MAE', 'Quantile', 'MAPE'
    quantile : float, optional
        If loss_function='Quantile', specify the quantile (e.g., 0.1, 0.5, 0.9)
    **model_params
        Additional parameters for CatBoostRegressor

    Returns
    -------
    CatBoostRegressor or None
        CatBoost model, or None if CatBoost is not installed
    """
    if not CATBOOST_AVAILABLE:
        warnings.warn("CatBoost not available. Install with: pip install catboost")
        return None

    default_params = {
        'iterations': 500,
        'learning_rate': 0.1,
        'depth': 6,
        'random_seed': 42,
        'verbose': False,
        'cat_features': categorical_cols,
    }

    # Handle quantile regression
    if loss_function == 'Quantile' and quantile is not None:
        default_params['loss_function'] = f'Quantile:alpha={quantile}'
    else:
        default_params['loss_function'] = loss_function

    default_params.update(model_params)

    return CatBoostRegressor(**default_params)


def prepare_catboost_data(df: pd.DataFrame,
                          numeric_cols: List[str],
                          categorical_cols: List[str],
                          target_col: str = TARGET_COL) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare data for CatBoost (handles missing values in categorical columns).

    CatBoost can handle NaN in numeric columns but needs string 'nan' for categoricals.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        X (features) and y (target)
    """
    feature_cols = numeric_cols + categorical_cols
    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # Fill NaN in categorical columns with string 'missing'
    for col in categorical_cols:
        X[col] = X[col].fillna('missing').astype(str)

    return X, y


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_sklearn_model(pipeline: Pipeline,
                        X: pd.DataFrame,
                        y: pd.Series) -> Pipeline:
    """
    Train an sklearn pipeline.

    Returns the fitted pipeline.
    """
    return pipeline.fit(X, y)


def train_catboost_model(model: 'CatBoostRegressor',
                         X: pd.DataFrame,
                         y: pd.Series,
                         eval_set: Tuple = None) -> 'CatBoostRegressor':
    """
    Train a CatBoost model.

    Parameters
    ----------
    model : CatBoostRegressor
        The model to train
    X : pd.DataFrame
        Features (already prepared with prepare_catboost_data)
    y : pd.Series
        Target
    eval_set : Tuple, optional
        Validation set as (X_val, y_val) for early stopping

    Returns
    -------
    CatBoostRegressor
        Fitted model
    """
    if eval_set is not None:
        model.fit(X, y, eval_set=eval_set, early_stopping_rounds=50)
    else:
        model.fit(X, y)

    return model


# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

def evaluate_sklearn_cv(pipeline: Pipeline,
                        X: pd.DataFrame,
                        y: pd.Series,
                        cv: int = 5) -> Dict[str, float]:
    """
    Evaluate sklearn pipeline using cross-validation.

    Returns
    -------
    Dict with metrics: rmse_mean, rmse_std, mae_mean, mae_std, r2_mean, r2_std
    """
    scoring = {
        'neg_rmse': 'neg_root_mean_squared_error',
        'neg_mae': 'neg_mean_absolute_error',
        'r2': 'r2',
    }

    results = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, return_train_score=False)

    return {
        'rmse_mean': -results['test_neg_rmse'].mean(),
        'rmse_std': results['test_neg_rmse'].std(),
        'mae_mean': -results['test_neg_mae'].mean(),
        'mae_std': results['test_neg_mae'].std(),
        'r2_mean': results['test_r2'].mean(),
        'r2_std': results['test_r2'].std(),
    }


def evaluate_predictions(y_true: pd.Series,
                         y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate evaluation metrics for predictions.

    Returns
    -------
    Dict with: rmse, mae, r2, mape
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # MAPE (handle zero values)
    mask = y_true != 0
    if mask.sum() > 0:
        mape = mean_absolute_percentage_error(y_true[mask], y_pred[mask]) * 100
    else:
        mape = np.nan

    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape
    }


def compare_models(results_dict: Dict[str, Dict]) -> pd.DataFrame:
    """
    Create a comparison table from multiple model results.

    Parameters
    ----------
    results_dict : Dict[str, Dict]
        Dictionary mapping model name to its evaluation metrics

    Returns
    -------
    pd.DataFrame
        Comparison table sorted by RMSE
    """
    rows = []
    for model_name, metrics in results_dict.items():
        row = {'Model': model_name}
        row.update(metrics)
        rows.append(row)

    df = pd.DataFrame(rows)

    # Sort by RMSE (or rmse_mean if cross-validation)
    sort_col = 'rmse_mean' if 'rmse_mean' in df.columns else 'rmse'
    if sort_col in df.columns:
        df = df.sort_values(sort_col).reset_index(drop=True)

    return df


# =============================================================================
# MODEL PERSISTENCE
# =============================================================================

def save_model(model: Union[Pipeline, 'CatBoostRegressor'],
               path: str = 'models/price_model.pkl',
               metadata: Dict = None) -> None:
    """
    Save a trained model to disk.

    Parameters
    ----------
    model : Pipeline or CatBoostRegressor
        The trained model
    path : str
        Output path
    metadata : Dict, optional
        Additional metadata to save (e.g., feature columns, metrics)
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    save_obj = {
        'model': model,
        'metadata': metadata or {}
    }

    joblib.dump(save_obj, path)
    print(f"Model saved to: {path}")


def load_model(path: str = 'models/price_model.pkl') -> Tuple[Union[Pipeline, 'CatBoostRegressor'], Dict]:
    """
    Load a trained model from disk.

    Returns
    -------
    Tuple[model, metadata]
    """
    save_obj = joblib.load(path)
    return save_obj['model'], save_obj.get('metadata', {})


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def load_features_data(path: str = 'data/db_features.parquet') -> pd.DataFrame:
    """
    Load the processed features dataset.

    Tries parquet first, falls back to CSV.
    """
    path = Path(path)

    if path.exists():
        try:
            return pd.read_parquet(path)
        except ImportError:
            pass

    # Try CSV fallback
    csv_path = path.with_suffix('.csv')
    if csv_path.exists():
        return pd.read_csv(csv_path)

    raise FileNotFoundError(f"Could not find {path} or {csv_path}")


def quick_train_evaluate(df: pd.DataFrame,
                         model_type: str = 'random_forest',
                         target_col: str = TARGET_COL,
                         cv: int = 5) -> Tuple[Pipeline, Dict]:
    """
    Quick convenience function to train and evaluate a model.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with features and target
    model_type : str
        Type of model to use
    target_col : str
        Target column name
    cv : int
        Number of CV folds

    Returns
    -------
    Tuple[Pipeline, Dict]
        Fitted pipeline and CV metrics
    """
    # Infer feature types
    feature_cols, numeric_cols, categorical_cols = infer_feature_types(df, target_col)

    # Prepare data
    X = df[feature_cols]
    y = df[target_col]

    # Remove rows with missing target
    mask = y.notna()
    X = X[mask]
    y = y[mask]

    # Build and train pipeline
    pipeline = build_sklearn_pipeline(model_type, numeric_cols, categorical_cols)

    # Cross-validate
    metrics = evaluate_sklearn_cv(pipeline, X, y, cv=cv)

    # Fit on full data
    pipeline.fit(X, y)

    return pipeline, metrics
