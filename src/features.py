"""
features.py - Feature Engineering Module

This module contains functions for loading data and engineering features
for the computer price prediction model.

IMPORTANT CONVENTIONS:
- We MUST keep the original Spanish column names from the CSV files intact.
- All NEW engineered features MUST have names starting with an underscore (_).
  Examples: _cpu_mark, _gpu_mark, _ram_gb, _ssd_gb, _hdd_gb, _tamano_pantalla_pulgadas

Planned engineered features:
- _cpu_mark: CPU benchmark score from db_cpu_raw.csv
- _gpu_mark: GPU benchmark score from db_gpu_raw.csv
- _ram_gb: Extracted RAM size in GB (numeric)
- _ssd_gb: Extracted SSD capacity in GB (numeric)
- _hdd_gb: Extracted HDD capacity in GB (numeric)
- _tamano_pantalla_pulgadas: Screen size in inches (numeric)
- _precio_num: Numeric price extracted from price range string
- _es_gaming: Boolean flag for gaming laptops
- _es_portatil: Boolean flag for laptops vs desktops
"""

import pandas as pd
from pathlib import Path


def cargar_datos(ruta_computers: str, ruta_cpu: str, ruta_gpu: str) -> tuple:
    """
    Load the three main data files for the computer price prediction project.

    Parameters
    ----------
    ruta_computers : str
        Path to the main computers dataset (db_computers_2025_raw.csv)
    ruta_cpu : str
        Path to the CPU benchmark dataset (db_cpu_raw.csv)
    ruta_gpu : str
        Path to the GPU benchmark dataset (db_gpu_raw.csv)

    Returns
    -------
    tuple
        A tuple of three DataFrames: (df_computers, df_cpu, df_gpu)

    Notes
    -----
    - Uses encoding='utf-8-sig' to handle BOM in CSV files.
    - Uses index_col=0 then reset_index() as per course instructions.
    - Original Spanish column names are preserved exactly as-is.

    TODO:
    - Implement the actual data loading logic
    - Add data validation and basic sanity checks
    - Handle missing files gracefully with informative errors
    """
    # TODO: Implement data loading
    # df_computers = pd.read_csv(ruta_computers, encoding='utf-8-sig', index_col=0).reset_index()
    # df_cpu = pd.read_csv(ruta_cpu, encoding='utf-8-sig')
    # df_gpu = pd.read_csv(ruta_gpu, encoding='utf-8-sig')
    # return df_computers, df_cpu, df_gpu
    raise NotImplementedError("cargar_datos() not yet implemented")


def construir_features(df_computers: pd.DataFrame,
                       df_cpu: pd.DataFrame,
                       df_gpu: pd.DataFrame) -> pd.DataFrame:
    """
    Build engineered features from the raw computer data.

    This function takes the raw computer listings and enriches them with
    engineered features for the ML model. All new features are prefixed
    with an underscore to distinguish them from original columns.

    Parameters
    ----------
    df_computers : pd.DataFrame
        Main dataset with computer listings (original columns preserved)
    df_cpu : pd.DataFrame
        CPU benchmark data for enriching processor performance info
    df_gpu : pd.DataFrame
        GPU benchmark data for enriching graphics card performance info

    Returns
    -------
    pd.DataFrame
        The df_computers DataFrame with additional engineered columns.
        Original columns remain unchanged; new columns start with '_'.

    Engineered Features (planned)
    -----------------------------
    _cpu_mark : float
        CPU benchmark score matched from df_cpu using fuzzy matching
        on the 'Procesador_Procesador' column.

    _gpu_mark : float
        GPU benchmark score matched from df_gpu using fuzzy matching
        on the 'Gráfica_Tarjeta gráfica' column.

    _ram_gb : float
        RAM extracted from 'RAM_Memoria RAM' as numeric GB value.

    _ssd_gb : float
        SSD capacity extracted from 'Disco duro_Capacidad de memoria SSD'.

    _hdd_gb : float
        HDD capacity extracted from 'Disco duro_Capacidad del disco duro'.

    _tamano_pantalla_pulgadas : float
        Screen size in inches extracted from 'Pantalla_Diagonal de la pantalla'.

    _precio_num : float
        Numeric price (midpoint of range) extracted from 'Precio_Rango'.

    _es_gaming : bool
        True if 'Tipo de producto' contains 'gaming'.

    _es_portatil : bool
        True if the product is a laptop (based on 'Tipo' or 'Tipo de producto').

    Notes
    -----
    - Original Spanish column names must NOT be modified.
    - Use fuzzy matching (fuzzywuzzy) for CPU/GPU name matching.
    - Handle missing values gracefully (leave as NaN if can't extract).

    TODO:
    - Implement each feature extraction function
    - Create helper functions for parsing Spanish numeric formats (e.g., "1.024,5" -> 1024.5)
    - Implement fuzzy matching logic for CPU and GPU names
    - Add validation for extracted values (e.g., RAM should be positive)
    """
    # TODO: Implement feature engineering
    # df = df_computers.copy()
    #
    # # Extract numeric price from range string like "1.026,53 € – 2.287,17 €"
    # df['_precio_num'] = df['Precio_Rango'].apply(_extraer_precio_medio)
    #
    # # Extract RAM in GB
    # df['_ram_gb'] = df['RAM_Memoria RAM'].apply(_extraer_ram_gb)
    #
    # # Match CPU benchmark scores
    # df['_cpu_mark'] = df['Procesador_Procesador'].apply(
    #     lambda x: _buscar_benchmark_cpu(x, df_cpu)
    # )
    #
    # # ... etc for other features
    #
    # return df
    raise NotImplementedError("construir_features() not yet implemented")


# =============================================================================
# HELPER FUNCTIONS (stubs for later implementation)
# =============================================================================

def _extraer_precio_medio(precio_rango: str) -> float:
    """
    Extract the midpoint price from a range string.

    Example: "1.026,53 € – 2.287,17 €" -> 1656.85

    TODO: Implement parsing of Spanish number format (. for thousands, , for decimals)
    """
    # TODO: Implement
    raise NotImplementedError()


def _extraer_ram_gb(ram_str: str) -> float:
    """
    Extract RAM size in GB from a string like "16 GB RAM".

    TODO: Handle various formats and units
    """
    # TODO: Implement
    raise NotImplementedError()


def _buscar_benchmark_cpu(nombre_cpu: str, df_cpu: pd.DataFrame) -> float:
    """
    Find the benchmark score for a CPU using fuzzy matching.

    TODO: Use fuzzywuzzy to find the best match in df_cpu
    """
    # TODO: Implement fuzzy matching
    raise NotImplementedError()


def _buscar_benchmark_gpu(nombre_gpu: str, df_gpu: pd.DataFrame) -> float:
    """
    Find the benchmark score for a GPU using fuzzy matching.

    TODO: Use fuzzywuzzy to find the best match in df_gpu
    """
    # TODO: Implement fuzzy matching
    raise NotImplementedError()
