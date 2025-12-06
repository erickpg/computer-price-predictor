"""
backend_api.py - API Wrapper for Streamlit Frontend

This module provides a thin API layer that the Streamlit frontend teammate
can use to get price predictions and explanations without needing to
understand the underlying ML pipeline.

Usage by Frontend Teammate:
--------------------------
>>> from src.backend_api import predecir_precio, explicar_prediccion
>>>
>>> # User inputs from Streamlit form
>>> inputs = {
...     'Tipo de producto': 'Portátil gaming',
...     'Serie': 'ASUS TUF Gaming',
...     'Procesador_Procesador': 'AMD Ryzen 7 7435HS',
...     'RAM_Memoria RAM': '16 GB RAM',
...     'Disco duro_Capacidad de memoria SSD': '512 GB',
...     'Gráfica_Tarjeta gráfica': 'NVIDIA GeForce RTX 3050',
...     # ... other fields
... }
>>>
>>> precio_predicho = predecir_precio(inputs)
>>> print(f"Predicted price: {precio_predicho:.2f} €")
>>>
>>> explicacion = explicar_prediccion(inputs)
>>> print(explicacion)  # Feature importances or SHAP values
"""

import pandas as pd
from typing import Dict, Any, Optional

# TODO: Uncomment when implementing
# from .modeling import cargar_modelo
# from .features import construir_features
# import shap


# Global variable to cache the loaded model (loaded once at startup)
_modelo_cargado = None


def _cargar_modelo_si_necesario():
    """
    Load the model if not already loaded.

    Uses a global cache to avoid reloading on every prediction.

    TODO:
    - Implement model loading with caching
    - Add error handling for missing model file
    """
    global _modelo_cargado
    # TODO: Implement
    # if _modelo_cargado is None:
    #     _modelo_cargado = cargar_modelo('models/price_model.pkl')
    # return _modelo_cargado
    raise NotImplementedError()


def predecir_precio(campos_dict: Dict[str, Any]) -> float:
    """
    Predict the price for a single computer based on user inputs.

    This function is the main entry point for the Streamlit frontend.
    It takes a dictionary of user inputs (matching the column names
    from the original dataset), builds a single-row DataFrame,
    applies feature engineering, and returns the predicted price.

    Parameters
    ----------
    campos_dict : Dict[str, Any]
        Dictionary mapping column names to values.
        Keys should use the ORIGINAL Spanish column names from the dataset.
        Example:
        {
            'Tipo de producto': 'Portátil gaming',
            'Serie': 'ASUS TUF Gaming',
            'Procesador_Procesador': 'AMD Ryzen 7 7435HS',
            'RAM_Memoria RAM': '16 GB RAM',
            'Disco duro_Capacidad de memoria SSD': '512 GB',
            'Gráfica_Tarjeta gráfica': 'NVIDIA GeForce RTX 3050',
            'Pantalla_Diagonal de la pantalla': '39,624 cm',
            ...
        }

    Returns
    -------
    float
        The predicted price in euros (€)

    Raises
    ------
    ValueError
        If required fields are missing from campos_dict
    RuntimeError
        If the model is not yet trained or cannot be loaded

    Notes
    -----
    - The function handles feature engineering internally (calling construir_features)
    - Missing optional fields are handled gracefully by the pipeline's imputers
    - For the Streamlit UI, consider providing sensible defaults for optional fields

    TODO:
    - Implement the full prediction pipeline
    - Add input validation (check required fields)
    - Add logging for debugging predictions
    - Consider confidence intervals / prediction uncertainty

    Example
    -------
    >>> inputs = {
    ...     'Tipo de producto': 'Portátil gaming',
    ...     'Procesador_Procesador': 'Intel Core i7-13700H',
    ...     'RAM_Memoria RAM': '16 GB RAM',
    ...     'Disco duro_Capacidad de memoria SSD': '512 GB',
    ...     'Gráfica_Tarjeta gráfica': 'NVIDIA GeForce RTX 4060',
    ... }
    >>> precio = predecir_precio(inputs)
    >>> print(f"Estimated price: {precio:.2f} €")
    Estimated price: 1249.50 €
    """
    # TODO: Implement
    #
    # # 1. Load model if needed
    # modelo = _cargar_modelo_si_necesario()
    #
    # # 2. Create single-row DataFrame from inputs
    # df_input = pd.DataFrame([campos_dict])
    #
    # # 3. Load auxiliary data for feature engineering (CPU/GPU benchmarks)
    # # Note: These should probably be cached too for performance
    # df_cpu = pd.read_csv('data/db_cpu_raw.csv', encoding='utf-8-sig')
    # df_gpu = pd.read_csv('data/db_gpu_raw.csv', encoding='utf-8-sig')
    #
    # # 4. Apply feature engineering
    # df_features = construir_features(df_input, df_cpu, df_gpu)
    #
    # # 5. Select only the columns the model expects
    # from .modeling import NUMERIC_COLS, CATEGORICAL_COLS
    # columnas_modelo = NUMERIC_COLS + CATEGORICAL_COLS
    # X = df_features[columnas_modelo]
    #
    # # 6. Get prediction
    # prediccion = modelo.predict(X)[0]
    #
    # return float(prediccion)
    raise NotImplementedError("predecir_precio() not yet implemented")


def explicar_prediccion(campos_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Explain the price prediction using feature importance or SHAP values.

    This function provides interpretability for the prediction, showing
    which features contributed most to the predicted price. This is
    useful for users who want to understand why a certain price was
    predicted.

    Parameters
    ----------
    campos_dict : Dict[str, Any]
        Same format as predecir_precio() - dictionary of user inputs
        with Spanish column names.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing explanation information:
        {
            'prediccion': float,  # The predicted price
            'feature_importances': Dict[str, float],  # Feature name -> importance
            'top_features': List[Tuple[str, float]],  # Top 5 most important features
            'shap_values': Optional[Dict],  # SHAP values if available
        }

    Notes
    -----
    - For RandomForest: Uses built-in feature_importances_
    - For GradientBoosting: Uses built-in feature_importances_
    - SHAP values provide more nuanced per-prediction explanations
    - Consider caching SHAP explainer for performance

    TODO:
    - Implement basic feature importance extraction
    - Add SHAP integration for detailed explanations
    - Create human-readable explanation text
    - Add visualization data (for plots in Streamlit)

    Example
    -------
    >>> inputs = {...}  # Same as predecir_precio
    >>> explicacion = explicar_prediccion(inputs)
    >>> print(f"Predicted: {explicacion['prediccion']:.2f} €")
    >>> print("Top factors:")
    >>> for feature, importance in explicacion['top_features']:
    ...     print(f"  - {feature}: {importance:.2%}")
    """
    # TODO: Implement
    #
    # # Get the prediction first
    # precio = predecir_precio(campos_dict)
    #
    # # Load model
    # modelo = _cargar_modelo_si_necesario()
    #
    # # Extract feature importances from the regressor
    # regressor = modelo.named_steps['regressor']
    # feature_names = modelo.named_steps['preprocessor'].get_feature_names_out()
    # importances = regressor.feature_importances_
    #
    # # Create importance dict
    # feature_importances = dict(zip(feature_names, importances))
    #
    # # Get top 5 features
    # top_features = sorted(
    #     feature_importances.items(),
    #     key=lambda x: abs(x[1]),
    #     reverse=True
    # )[:5]
    #
    # # TODO: Add SHAP values for more detailed explanation
    # # explainer = shap.TreeExplainer(regressor)
    # # shap_values = explainer.shap_values(X_processed)
    #
    # return {
    #     'prediccion': precio,
    #     'feature_importances': feature_importances,
    #     'top_features': top_features,
    #     'shap_values': None  # TODO: Add SHAP support
    # }
    raise NotImplementedError("explicar_prediccion() not yet implemented")


def obtener_campos_disponibles() -> Dict[str, Any]:
    """
    Get information about available input fields for the frontend.

    This helper function returns metadata about the expected input fields,
    their types, and possible values (for categorical fields). This helps
    the Streamlit frontend build appropriate input widgets.

    Returns
    -------
    Dict[str, Any]
        Dictionary with field information:
        {
            'required_fields': List[str],
            'optional_fields': List[str],
            'categorical_options': Dict[str, List[str]],  # Field -> possible values
            'numeric_ranges': Dict[str, Tuple[float, float]],  # Field -> (min, max)
        }

    TODO:
    - Implement based on training data analysis
    - Add field descriptions for UI labels
    - Add default values for optional fields
    """
    # TODO: Implement after feature engineering is complete
    raise NotImplementedError("obtener_campos_disponibles() not yet implemented")
