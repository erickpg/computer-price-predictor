"""
features.py - Feature Engineering Module

This module contains functions for loading data and engineering features
for the computer price prediction model.

IMPORTANT CONVENTIONS:
- We MUST keep the original Spanish column names from the CSV files intact.
- All NEW engineered features MUST have names starting with an underscore (_).
  Examples: _cpu_mark, _gpu_mark, _ram_gb, _ssd_gb, _precio_num, _brand, _serie

Engineered features (Core):
- _precio_num: Numeric price (midpoint of range) from Precio_Rango [TARGET]
- _brand: Brand extracted from Título
- _serie: Product series (from Serie column + extraction from Título)
- _cpu_mark: CPU benchmark score from db_cpu_raw.csv
- _gpu_mark: GPU benchmark score from db_gpu_raw.csv
- _ram_gb: Extracted RAM size in GB (numeric)
- _ssd_gb: Extracted SSD capacity in GB (numeric)
- _tamano_pantalla_pulgadas: Screen size in inches (from column + extraction from Título)

Engineered features (Additional):
- _peso_kg: Weight in kg (numeric)
- _resolucion_pixeles: Total pixels from resolution (width x height)
- _tasa_refresco_hz: Refresh rate in Hz (numeric)
- _tiene_wifi: Binary flag for WiFi connectivity
- _tiene_bluetooth: Binary flag for Bluetooth connectivity
- _tiene_webcam: Binary flag for webcam
- _version_bluetooth: Bluetooth version (numeric)
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
from typing import Optional

# Import fuzzywuzzy for CPU/GPU matching
try:
    from fuzzywuzzy import process, fuzz
    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False
    print("WARNING: fuzzywuzzy not available. Install with: pip install fuzzywuzzy python-Levenshtein")


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
    """
    # Load main computers dataset
    df_computers = pd.read_csv(
        ruta_computers,
        encoding='utf-8-sig',
        index_col=0,
        low_memory=False
    ).reset_index(drop=True)

    # Load CPU benchmark data
    df_cpu = pd.read_csv(
        ruta_cpu,
        encoding='utf-8-sig',
        index_col=0
    ).reset_index(drop=True)

    # Load GPU benchmark data
    df_gpu = pd.read_csv(
        ruta_gpu,
        encoding='utf-8-sig',
        index_col=0
    ).reset_index(drop=True)

    print(f"Loaded {len(df_computers):,} computer listings")
    print(f"Loaded {len(df_cpu):,} CPU benchmarks")
    print(f"Loaded {len(df_gpu):,} GPU benchmarks")

    return df_computers, df_cpu, df_gpu


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

    Engineered Features
    -------------------
    _precio_num : float
        Numeric price (midpoint of range) extracted from 'Precio_Rango'.

    _brand : str
        Brand extracted from 'Título' (first word).

    _serie : str
        Product series (from Serie column + extraction from Título).

    _cpu_mark : float
        CPU benchmark score matched from df_cpu using fuzzy matching.

    _gpu_mark : float
        GPU benchmark score matched from df_gpu using fuzzy matching.

    _ram_gb : float
        RAM extracted from 'RAM_Memoria RAM' as numeric GB value.

    _ssd_gb : float
        SSD capacity extracted from 'Disco duro_Capacidad de memoria SSD'.

    _tamano_pantalla_pulgadas : float
        Screen size in inches (from pulgadas or cm column).
    """
    # Work on a copy to avoid modifying the original
    df = df_computers.copy()

    print("\n=== Building Engineered Features ===\n")

    # 1. Extract numeric price (TARGET VARIABLE)
    print("1. Extracting _precio_num from Precio_Rango...")
    df['_precio_num'] = df['Precio_Rango'].apply(_extraer_precio_medio)
    print(f"   -> Valid prices: {df['_precio_num'].notna().sum():,} / {len(df):,}")

    # 2. Extract brand from title
    print("2. Extracting _brand from Título...")
    df['_brand'] = df['Título'].apply(_extraer_brand)
    print(f"   -> Brands extracted: {df['_brand'].notna().sum():,} / {len(df):,}")
    print(f"   -> Unique brands: {df['_brand'].nunique()}")

    # 3. Extract/combine series
    print("3. Building _serie from Serie + Título...")
    df['_serie'] = df.apply(
        lambda row: _extraer_serie(row.get('Serie'), row.get('Título'), row.get('_brand')),
        axis=1
    )
    print(f"   -> Series identified: {df['_serie'].notna().sum():,} / {len(df):,}")

    # 4. Extract RAM in GB
    print("4. Extracting _ram_gb from RAM_Memoria RAM...")
    df['_ram_gb'] = df['RAM_Memoria RAM'].apply(_extraer_ram_gb)
    print(f"   -> RAM extracted: {df['_ram_gb'].notna().sum():,} / {len(df):,}")

    # 5. Extract SSD capacity in GB
    print("5. Extracting _ssd_gb from Disco duro_Capacidad de memoria SSD...")
    df['_ssd_gb'] = df['Disco duro_Capacidad de memoria SSD'].apply(_extraer_ssd_gb)
    print(f"   -> SSD extracted: {df['_ssd_gb'].notna().sum():,} / {len(df):,}")

    # 6. Extract screen size in inches (improved with título extraction)
    print("6. Extracting _tamano_pantalla_pulgadas from screen size columns + Título...")
    df['_tamano_pantalla_pulgadas'] = df.apply(
        lambda row: _extraer_pantalla_pulgadas(
            row.get('Pantalla_Tamaño de la pantalla'),
            row.get('Pantalla_Diagonal de la pantalla'),
            row.get('Título')
        ),
        axis=1
    )
    print(f"   -> Screen size extracted: {df['_tamano_pantalla_pulgadas'].notna().sum():,} / {len(df):,}")

    # 6.5 Extract CPU cores (needed for Apple CPU matching)
    print("6.5. Extracting _cpu_cores from Procesador_Número de núcleos del procesador...")
    df['_cpu_cores'] = df['Procesador_Número de núcleos del procesador'].apply(_extraer_cpu_cores)
    print(f"   -> CPU cores extracted: {df['_cpu_cores'].notna().sum():,} / {len(df):,}")

    # 7. Match CPU benchmarks (with intelligent fallback for Apple/Intel/AMD)
    print("7. Matching _cpu_mark from Procesador_Procesador (with intelligent fallback)...")
    if FUZZY_AVAILABLE:
        df['_cpu_mark'] = df.apply(
            lambda row: _buscar_benchmark_cpu(
                row.get('Procesador_Procesador'),
                df_cpu,
                num_cores=row.get('_cpu_cores')
            ),
            axis=1
        )
        print(f"   -> CPU benchmarks matched: {df['_cpu_mark'].notna().sum():,} / {len(df):,}")
        match_rate = (df['_cpu_mark'].notna().sum() / df['Procesador_Procesador'].notna().sum() * 100) if df['Procesador_Procesador'].notna().sum() > 0 else 0
        print(f"   -> Match rate: {match_rate:.1f}% of processors with names")
    else:
        print("   -> SKIPPED: fuzzywuzzy not available")
        df['_cpu_mark'] = np.nan

    # 8. Match GPU benchmarks (with intelligent fallback for laptop/desktop variants)
    print("8. Matching _gpu_mark from Gráfica_Tarjeta gráfica (with intelligent fallback)...")
    if FUZZY_AVAILABLE:
        df['_gpu_mark'] = df['Gráfica_Tarjeta gráfica'].apply(
            lambda x: _buscar_benchmark_gpu(x, df_gpu)
        )
        print(f"   -> GPU benchmarks matched: {df['_gpu_mark'].notna().sum():,} / {len(df):,}")
        match_rate = (df['_gpu_mark'].notna().sum() / df['Gráfica_Tarjeta gráfica'].notna().sum() * 100) if df['Gráfica_Tarjeta gráfica'].notna().sum() > 0 else 0
        print(f"   -> Match rate: {match_rate:.1f}% of GPUs with names")
    else:
        print("   -> SKIPPED: fuzzywuzzy not available")
        df['_gpu_mark'] = np.nan

    # 8.5 Extract GPU memory in GB
    print("8.5. Extracting _gpu_memory_gb from Gráfica_Memoria gráfica...")
    df['_gpu_memory_gb'] = df['Gráfica_Memoria gráfica'].apply(_extraer_gpu_memory_gb)
    print(f"   -> GPU memory extracted: {df['_gpu_memory_gb'].notna().sum():,} / {len(df):,}")

    # 8.6 Extract number of offers
    print("8.6. Extracting _num_ofertas from Ofertas...")
    df['_num_ofertas'] = df['Ofertas'].apply(_extraer_num_ofertas)
    print(f"   -> Number of offers extracted: {df['_num_ofertas'].notna().sum():,} / {len(df):,}")

    # === ADDITIONAL FEATURES ===

    # 9. Extract weight in kg
    print("9. Extracting _peso_kg from Medidas y peso_Peso...")
    df['_peso_kg'] = df['Medidas y peso_Peso'].apply(_extraer_peso_kg)
    print(f"   -> Weight extracted: {df['_peso_kg'].notna().sum():,} / {len(df):,}")

    # 10. Extract total pixels from resolution
    print("10. Extracting _resolucion_pixeles from Pantalla_Resolución de pantalla...")
    df['_resolucion_pixeles'] = df['Pantalla_Resolución de pantalla'].apply(_extraer_resolucion_pixeles)
    print(f"   -> Resolution extracted: {df['_resolucion_pixeles'].notna().sum():,} / {len(df):,}")

    # 11. Extract refresh rate in Hz
    print("11. Extracting _tasa_refresco_hz from Pantalla_Tasa de actualización de imagen...")
    df['_tasa_refresco_hz'] = df['Pantalla_Tasa de actualización de imagen'].apply(_extraer_tasa_refresco)
    print(f"   -> Refresh rate extracted: {df['_tasa_refresco_hz'].notna().sum():,} / {len(df):,}")

    # 12. Binary flag: Has WiFi
    print("12. Creating _tiene_wifi from Comunicaciones_Conectividad...")
    df['_tiene_wifi'] = df['Comunicaciones_Conectividad'].apply(_tiene_wifi)
    print(f"   -> WiFi flag: {df['_tiene_wifi'].notna().sum():,} / {len(df):,}")

    # 13. Binary flag: Has Bluetooth
    print("13. Creating _tiene_bluetooth from Comunicaciones_Conectividad...")
    df['_tiene_bluetooth'] = df['Comunicaciones_Conectividad'].apply(_tiene_bluetooth)
    print(f"   -> Bluetooth flag: {df['_tiene_bluetooth'].notna().sum():,} / {len(df):,}")

    # 14. Binary flag: Has Webcam
    print("14. Creating _tiene_webcam from Cámara_Webcam...")
    df['_tiene_webcam'] = df['Cámara_Webcam'].apply(_tiene_webcam)
    print(f"   -> Webcam flag: {df['_tiene_webcam'].notna().sum():,} / {len(df):,}")

    # 15. Extract Bluetooth version
    print("15. Extracting _version_bluetooth from Comunicaciones_Versión Bluetooth...")
    df['_version_bluetooth'] = df['Comunicaciones_Versión Bluetooth'].apply(_extraer_version_bluetooth)
    print(f"   -> Bluetooth version extracted: {df['_version_bluetooth'].notna().sum():,} / {len(df):,}")

    print("\n=== Feature Engineering Complete ===\n")
    print(f"Total engineered features: 18")

    # Summary of key features based on correlation analysis
    print("\n=== Feature Summary (by importance from correlation analysis) ===")
    print("Strong predictors (correlation > 0.5):")
    print("  - _ram_gb: RAM size in GB")
    print("  - _gpu_memory_gb: GPU memory in GB")
    print("  - _cpu_cores: Number of CPU cores")
    print("\nModerate predictors (correlation 0.3-0.5):")
    print("  - _tasa_refresco_hz: Refresh rate")
    print("  - _ssd_gb: SSD capacity")
    print("  - _cpu_mark: CPU benchmark score")
    print("  - _gpu_mark: GPU benchmark score")
    print("\nAdditional features:")
    print("  - _tamano_pantalla_pulgadas: Screen size")
    print("  - _resolucion_pixeles: Total pixels")
    print("  - _num_ofertas: Number of offers")
    print("  - Brand, Series, and other categorical features")

    return df


# =============================================================================
# HELPER FUNCTIONS - Price Extraction
# =============================================================================

def _extraer_precio_medio(precio_rango: str) -> Optional[float]:
    """
    Extract the midpoint price from a range string.

    Example: "1.026,53 € – 2.287,17 €" -> 1656.85

    Handles Spanish number format:
    - Period (.) for thousands separator
    - Comma (,) for decimal separator
    """
    if pd.isna(precio_rango) or not isinstance(precio_rango, str):
        return np.nan

    # Pattern to match Spanish-formatted numbers
    # Example: "1.026,53" or "2.287,17"
    pattern = r'([\d.]+,\d{2})'
    matches = re.findall(pattern, precio_rango)

    if not matches:
        return np.nan

    # Convert Spanish format to float
    precios = []
    for match in matches:
        # Remove thousand separators (.), replace decimal comma with period
        num_str = match.replace('.', '').replace(',', '.')
        try:
            precios.append(float(num_str))
        except ValueError:
            continue

    if len(precios) == 0:
        return np.nan
    elif len(precios) == 1:
        # Single price (not a range)
        return precios[0]
    elif len(precios) == 2:
        # Price range - return midpoint
        return (precios[0] + precios[1]) / 2
    else:
        # Multiple prices - return mean (unlikely case)
        return np.mean(precios)


# =============================================================================
# HELPER FUNCTIONS - Brand and Series Extraction
# =============================================================================

def _extraer_brand(titulo: str) -> Optional[str]:
    """
    Extract brand from product title (usually first word).

    Common brands: Apple, ASUS, Lenovo, HP, Dell, Acer, MSI, Samsung, etc.
    """
    if pd.isna(titulo):
        return np.nan

    # Common computer brands (case insensitive matching)
    common_brands = {
        'apple': 'Apple',
        'asus': 'ASUS',
        'lenovo': 'Lenovo',
        'hp': 'HP',
        'dell': 'Dell',
        'acer': 'Acer',
        'msi': 'MSI',
        'samsung': 'Samsung',
        'microsoft': 'Microsoft',
        'razer': 'Razer',
        'alienware': 'Alienware',
        'lg': 'LG',
        'huawei': 'Huawei',
        'xiaomi': 'Xiaomi',
        'gigabyte': 'Gigabyte',
        'toshiba': 'Toshiba',
        'fujitsu': 'Fujitsu',
        'medion': 'Medion',
        'sony': 'Sony',
        'vaio': 'Vaio',
        'corsair': 'Corsair',
        'nzxt': 'NZXT',
    }

    # Get first word (usually the brand)
    first_word = str(titulo).split()[0] if titulo else ''
    first_word_lower = first_word.lower()

    # Check if first word matches a known brand
    if first_word_lower in common_brands:
        return common_brands[first_word_lower]

    # If not found in common brands, return the first word with capitalization
    return first_word.capitalize() if first_word else np.nan


def _extraer_serie(serie_original: str, titulo: str, brand: str) -> Optional[str]:
    """
    Extract product series from Serie column or infer from Título.

    Strategy:
    1. If Serie column has a value, use it
    2. Otherwise, try to extract from título based on brand patterns
    """
    # If Serie column has value, use it
    if pd.notna(serie_original) and str(serie_original).strip():
        return str(serie_original).strip()

    # Otherwise, try to extract from título
    if pd.isna(titulo) or pd.isna(brand):
        return np.nan

    titulo_lower = str(titulo).lower()

    # Known series patterns by brand
    series_patterns = {
        'Apple': ['MacBook Air', 'MacBook Pro', 'iMac', 'Mac Mini', 'Mac Pro', 'Mac Studio'],
        'ASUS': ['ROG Zephyrus', 'ROG Strix', 'ROG Flow', 'TUF Gaming', 'Republic of Gamers',
                 'Zenbook', 'Vivobook', 'ExpertBook', 'ProArt', 'StudioBook', 'Chromebook'],
        'Lenovo': ['ThinkPad', 'IdeaPad', 'Legion', 'LOQ', 'Yoga', 'ThinkBook'],
        'HP': ['Pavilion', 'Envy', 'Omen', 'EliteBook', 'ProBook', 'Spectre', 'ZBook'],
        'Dell': ['Inspiron', 'XPS', 'Alienware', 'Latitude', 'Precision', 'Vostro'],
        'MSI': ['Katana', 'Stealth', 'Raider', 'Cyborg', 'Prestige', 'Modern', 'Summit'],
        'Acer': ['Aspire', 'Swift', 'Nitro', 'Predator', 'TravelMate', 'ConceptD'],
        'Samsung': ['Galaxy Book'],
        'Microsoft': ['Surface'],
        'Gigabyte': ['Aero', 'Aorus'],
    }

    if brand not in series_patterns:
        return np.nan

    # Look for series keywords in title (order matters - longer matches first)
    for series in sorted(series_patterns[brand], key=len, reverse=True):
        if series.lower() in titulo_lower:
            return series

    return np.nan


# =============================================================================
# HELPER FUNCTIONS - Numeric Extractions
# =============================================================================

def _extraer_ram_gb(ram_str: str) -> Optional[float]:
    """
    Extract RAM size in GB from a string like "16 GB RAM" or "8 GB DDR4".

    Handles various formats and units.
    """
    if pd.isna(ram_str):
        return np.nan

    ram_str = str(ram_str).upper()

    # Look for pattern like "16 GB" or "32GB"
    # Match digits followed by optional space and GB
    match = re.search(r'(\d+)\s*GB', ram_str)

    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return np.nan

    return np.nan


def _extraer_ssd_gb(ssd_str: str) -> Optional[float]:
    """
    Extract SSD capacity in GB from strings like "512 GB", "1 TB", "1.000 GB".

    Converts TB to GB (1 TB = 1024 GB).
    Handles Spanish formatting with periods as thousand separators.
    """
    if pd.isna(ssd_str):
        return np.nan

    ssd_str = str(ssd_str).upper()

    # Check for TB first (convert to GB)
    tb_match = re.search(r'([\d.,]+)\s*TB', ssd_str)
    if tb_match:
        try:
            # Handle Spanish format: "1,5 TB" or "2.0 TB"
            tb_value = tb_match.group(1).replace('.', '').replace(',', '.')
            return float(tb_value) * 1024
        except ValueError:
            pass

    # Look for GB
    gb_match = re.search(r'([\d.,]+)\s*GB', ssd_str)
    if gb_match:
        try:
            # Handle Spanish format: "1.000 GB" -> 1000
            gb_value = gb_match.group(1).replace('.', '').replace(',', '.')
            return float(gb_value)
        except ValueError:
            pass

    return np.nan


def _extraer_pantalla_pulgadas(pulgadas_str: str, cm_str: str, titulo: str = None) -> Optional[float]:
    """
    Extract screen size in inches.

    Strategy:
    1. Try to extract from pulgadas_str (primary source)
    2. If missing, try to convert from cm_str (1 inch = 2.54 cm)
    3. If still missing, try to extract from título (e.g., '14"', '15.6 pulgadas')

    Examples:
    - "15,6 pulgadas" -> 15.6
    - "39,624 cm" -> 15.6 (39.624 / 2.54)
    - "MacBook Air 13.6\"" -> 13.6
    """
    # Try pulgadas first
    if pd.notna(pulgadas_str):
        pulgadas_str = str(pulgadas_str)
        # Match pattern like "15,6" or "13.3"
        match = re.search(r'([\d,\.]+)', pulgadas_str)
        if match:
            try:
                # Handle Spanish decimal comma
                num_str = match.group(1).replace(',', '.')
                return float(num_str)
            except ValueError:
                pass

    # Fallback to cm conversion
    if pd.notna(cm_str):
        cm_str = str(cm_str)
        # Match pattern like "39,624 cm"
        match = re.search(r'([\d,\.]+)', cm_str)
        if match:
            try:
                # Handle Spanish decimal comma
                num_str = match.group(1).replace(',', '.')
                cm_value = float(num_str)
                return cm_value / 2.54  # Convert to inches
            except ValueError:
                pass

    # Fallback to título extraction
    if pd.notna(titulo):
        titulo = str(titulo)
        # Match patterns like: 14", 15.6", 13.3 pulgadas, 15,6"
        match = re.search(r'(\d{2}(?:[.,]\d)?)\s*(?:"|\'\'|pulgadas|pulg|inch)', titulo, re.IGNORECASE)
        if match:
            try:
                num_str = match.group(1).replace(',', '.')
                return float(num_str)
            except ValueError:
                pass

    return np.nan


# =============================================================================
# HELPER FUNCTIONS - CPU/GPU Name Preprocessing for Better Matching
# =============================================================================

def _preprocess_cpu_name(nombre_cpu: str) -> str:
    """
    Preprocess CPU name to improve fuzzy matching success rate.

    Transformations:
    - Remove extra whitespace
    - Standardize common variations (e.g., "Core i7" patterns)
    - Keep generation numbers and model numbers intact
    """
    if pd.isna(nombre_cpu):
        return ""

    nombre = str(nombre_cpu).strip()

    # Common replacements for better matching
    replacements = {
        'Intel Core Ultra': 'Intel Core',
        'AMD Ryzen AI': 'AMD Ryzen',
    }

    for old, new in replacements.items():
        nombre = nombre.replace(old, new)

    return nombre


def _preprocess_gpu_name(nombre_gpu: str) -> str:
    """
    Preprocess GPU name to improve fuzzy matching success rate.

    Transformations:
    - Expand common abbreviations
    - Standardize NVIDIA naming (GeForce RTX/GTX)
    - Handle integrated graphics better
    - Remove generic descriptors that hurt matching
    """
    if pd.isna(nombre_gpu):
        return ""

    nombre = str(nombre_gpu).strip()

    # Skip generic integrated graphics that won't match
    generic_patterns = [
        'Intel Arc Graphics',
        'Intel Graphics',
        'Intel UHD Graphics',
        'Intel Iris Xe Graphics',
        'AMD Radeon Graphics',
        'Apple M[0-9]+ Graphics',
        'Apple M[0-9]+ Pro Graphics',
        'Apple M[0-9]+ Max Graphics',
        'Qualcomm Adreno'
    ]

    for pattern in generic_patterns:
        if re.match(pattern, nombre, re.IGNORECASE):
            return ""  # Return empty to signal skip

    # NVIDIA standardization
    nombre = nombre.replace('NVIDIA GeForce', 'GeForce')
    nombre = nombre.replace('NVIDIA', 'GeForce')

    # AMD standardization
    nombre = nombre.replace('AMD Radeon', 'Radeon')

    return nombre


# =============================================================================
# HELPER FUNCTIONS - Fuzzy Matching for Benchmarks
# =============================================================================

def _buscar_benchmark_cpu(nombre_cpu: str, df_cpu: pd.DataFrame,
                          num_cores: Optional[float] = None,
                          threshold: int = 70) -> Optional[float]:
    """
    Find the benchmark score for a CPU using intelligent fuzzy matching with fallback.

    Implements progressive matching strategy:
    1. Try exact/fuzzy match with full name
    2. For Apple processors: combine name + cores count
    3. Strip suffixes (H, HX, U, etc.) progressively and retry
    4. Try matching base model number only

    Parameters
    ----------
    nombre_cpu : str
        CPU name from the main dataset (e.g., "Intel Core i7-13700H")
    df_cpu : pd.DataFrame
        CPU benchmark dataframe with columns ['CPU Name', 'CPU Mark (higher is better)']
    num_cores : float, optional
        Number of CPU cores (used for Apple processor matching)
    threshold : int, default=70
        Minimum fuzzywuzzy score to consider a match (0-100)

    Returns
    -------
    float or None
        CPU Mark score if a good match is found, otherwise None
    """
    if pd.isna(nombre_cpu) or not FUZZY_AVAILABLE:
        return np.nan

    if df_cpu.empty or 'CPU Name' not in df_cpu.columns:
        return np.nan

    # Preprocess CPU name for better matching
    nombre_cpu_processed = _preprocess_cpu_name(nombre_cpu)

    if not nombre_cpu_processed:
        return np.nan

    # Get list of CPU names from benchmark data
    cpu_names = df_cpu['CPU Name'].dropna().astype(str).tolist()

    if not cpu_names:
        return np.nan

    # Strategy 1: Try exact fuzzy match first
    try:
        match = process.extractOne(
            nombre_cpu_processed,
            cpu_names,
            scorer=fuzz.token_sort_ratio
        )

        if match and match[1] >= threshold:
            matched_name = match[0]
            score = df_cpu.loc[df_cpu['CPU Name'] == matched_name, 'CPU Mark (higher is better)'].iloc[0]
            return float(score) if pd.notna(score) else np.nan
    except Exception:
        pass

    # Strategy 2: Handle Apple processors specially (combine with core count)
    if 'apple' in nombre_cpu_processed.lower() and pd.notna(num_cores):
        # Try patterns like "Apple M3 Pro 11-Core" or "Apple M3 8-Core"
        apple_patterns = [
            f"{nombre_cpu_processed} {int(num_cores)}-Core",
            f"{nombre_cpu_processed} {int(num_cores)} Core",
            f"Apple M{_extract_m_version(nombre_cpu_processed)} {int(num_cores)}-Core",
        ]

        for pattern in apple_patterns:
            try:
                match = process.extractOne(pattern, cpu_names, scorer=fuzz.token_sort_ratio)
                if match and match[1] >= threshold - 10:  # Slightly lower threshold
                    matched_name = match[0]
                    score = df_cpu.loc[df_cpu['CPU Name'] == matched_name, 'CPU Mark (higher is better)'].iloc[0]
                    return float(score) if pd.notna(score) else np.nan
            except Exception:
                continue

    # Strategy 3: Progressive suffix stripping for Intel/AMD processors
    # Common suffixes to strip: H, HX, U, P, K, KF, X, HS, HK, etc.
    suffixes_to_strip = ['HX', 'HS', 'HK', 'H', 'U', 'P', 'K', 'KF', 'X', 'T', 'S', 'F']

    for suffix in suffixes_to_strip:
        # Try removing suffix (with variations)
        for pattern in [f'-{suffix}', f' {suffix}', suffix]:
            if pattern in nombre_cpu_processed:
                stripped = nombre_cpu_processed.replace(pattern, '')
                try:
                    match = process.extractOne(stripped, cpu_names, scorer=fuzz.token_sort_ratio)
                    if match and match[1] >= threshold - 5:  # Slightly lower threshold
                        matched_name = match[0]
                        score = df_cpu.loc[df_cpu['CPU Name'] == matched_name, 'CPU Mark (higher is better)'].iloc[0]
                        return float(score) if pd.notna(score) else np.nan
                except Exception:
                    continue

    # Strategy 4: Try extracting just the core model number (e.g., "i7-13700" from "i7-13700H")
    model_match = re.search(r'(i[3579][-\s]?\d{4,5}|Ryzen\s+[3579]\s+\d{4})', nombre_cpu_processed, re.IGNORECASE)
    if model_match:
        base_model = model_match.group(1)
        try:
            match = process.extractOne(base_model, cpu_names, scorer=fuzz.partial_ratio)
            if match and match[1] >= threshold - 10:
                matched_name = match[0]
                score = df_cpu.loc[df_cpu['CPU Name'] == matched_name, 'CPU Mark (higher is better)'].iloc[0]
                return float(score) if pd.notna(score) else np.nan
        except Exception:
            pass

    return np.nan


def _extract_m_version(cpu_name: str) -> str:
    """Extract M-series version from Apple CPU name (e.g., 'M3', 'M2')."""
    match = re.search(r'M(\d+)', cpu_name, re.IGNORECASE)
    return match.group(1) if match else ''


def _buscar_benchmark_gpu(nombre_gpu: str, df_gpu: pd.DataFrame,
                          threshold: int = 70) -> Optional[float]:
    """
    Find the benchmark score for a GPU using intelligent fuzzy matching with fallback.

    Implements progressive matching strategy:
    1. Try exact/fuzzy match with full name
    2. Strip suffixes (Laptop, Mobile, Max-Q, etc.) and retry
    3. Try matching base model number only

    Parameters
    ----------
    nombre_gpu : str
        GPU name from the main dataset (e.g., "NVIDIA GeForce RTX 4060")
    df_gpu : pd.DataFrame
        GPU benchmark dataframe with columns ['Videocard Name', 'Passmark G3D Mark (higher is better)']
    threshold : int, default=70
        Minimum fuzzywuzzy score to consider a match (0-100)

    Returns
    -------
    float or None
        G3D Mark score if a good match is found, otherwise None
    """
    if pd.isna(nombre_gpu) or not FUZZY_AVAILABLE:
        return np.nan

    if df_gpu.empty or 'Videocard Name' not in df_gpu.columns:
        return np.nan

    # Preprocess GPU name for better matching (skip generic integrated graphics)
    nombre_gpu_processed = _preprocess_gpu_name(nombre_gpu)

    if not nombre_gpu_processed:
        # Skip generic integrated graphics that won't match benchmark data
        return np.nan

    # Get list of GPU names from benchmark data
    gpu_names = df_gpu['Videocard Name'].dropna().astype(str).tolist()

    if not gpu_names:
        return np.nan

    # Strategy 1: Try exact fuzzy match first
    try:
        match = process.extractOne(
            nombre_gpu_processed,
            gpu_names,
            scorer=fuzz.token_sort_ratio
        )

        if match and match[1] >= threshold:
            matched_name = match[0]
            score = df_gpu.loc[df_gpu['Videocard Name'] == matched_name, 'Passmark G3D Mark (higher is better)'].iloc[0]
            return float(score) if pd.notna(score) else np.nan
    except Exception:
        pass

    # Strategy 2: Try with "Laptop" or "Mobile" suffix added (laptop GPUs)
    laptop_variants = [
        f"{nombre_gpu_processed} Laptop",
        f"{nombre_gpu_processed} Mobile",
        f"{nombre_gpu_processed} Laptop GPU",
    ]

    for variant in laptop_variants:
        try:
            match = process.extractOne(variant, gpu_names, scorer=fuzz.token_sort_ratio)
            if match and match[1] >= threshold - 5:
                matched_name = match[0]
                score = df_gpu.loc[df_gpu['Videocard Name'] == matched_name, 'Passmark G3D Mark (higher is better)'].iloc[0]
                return float(score) if pd.notna(score) else np.nan
        except Exception:
            continue

    # Strategy 3: Strip common suffixes that might be missing in DB
    suffixes_to_strip = ['Laptop', 'Mobile', 'Max-Q', 'Ti', 'SUPER', 'XT', 'OEM']

    for suffix in suffixes_to_strip:
        if suffix.lower() in nombre_gpu_processed.lower():
            stripped = re.sub(rf'\s*{suffix}\s*', ' ', nombre_gpu_processed, flags=re.IGNORECASE).strip()
            try:
                match = process.extractOne(stripped, gpu_names, scorer=fuzz.token_sort_ratio)
                if match and match[1] >= threshold - 5:
                    matched_name = match[0]
                    score = df_gpu.loc[df_gpu['Videocard Name'] == matched_name, 'Passmark G3D Mark (higher is better)'].iloc[0]
                    return float(score) if pd.notna(score) else np.nan
            except Exception:
                continue

    # Strategy 4: Extract just the model number (e.g., "RTX 4060" or "RX 7600")
    model_match = re.search(r'(RTX\s*\d{4}|GTX\s*\d{4}|RX\s*\d{4})', nombre_gpu_processed, re.IGNORECASE)
    if model_match:
        base_model = model_match.group(1)
        try:
            match = process.extractOne(base_model, gpu_names, scorer=fuzz.partial_ratio)
            if match and match[1] >= threshold - 10:
                matched_name = match[0]
                score = df_gpu.loc[df_gpu['Videocard Name'] == matched_name, 'Passmark G3D Mark (higher is better)'].iloc[0]
                return float(score) if pd.notna(score) else np.nan
        except Exception:
            pass

    return np.nan


# =============================================================================
# HELPER FUNCTIONS - Additional Features
# =============================================================================

def _extraer_peso_kg(peso_str: str) -> Optional[float]:
    """
    Extract weight in kg from strings like "1,24 kg" or "1.5 kg".

    Handles Spanish decimal formatting (comma as decimal separator).
    """
    if pd.isna(peso_str):
        return np.nan

    peso_str = str(peso_str)

    # Match pattern like "1,24" or "1.5" before "kg"
    match = re.search(r'([\d,\.]+)\s*kg', peso_str, re.IGNORECASE)

    if match:
        try:
            # Handle Spanish decimal comma
            num_str = match.group(1).replace(',', '.')
            return float(num_str)
        except ValueError:
            pass

    return np.nan


def _extraer_resolucion_pixeles(resolucion_str: str) -> Optional[int]:
    """
    Extract total pixels from resolution string.

    Examples:
    - "1.920 x 1.080 píxeles" -> 2073600 (1920 * 1080)
    - "3.024 x 1.964 píxeles" -> 5939136

    Handles Spanish number formatting with periods as thousand separators.
    """
    if pd.isna(resolucion_str):
        return np.nan

    resolucion_str = str(resolucion_str)

    # Match pattern like "1.920 x 1.080" or "1920 x 1080"
    match = re.search(r'([\d.]+)\s*x\s*([\d.]+)', resolucion_str, re.IGNORECASE)

    if match:
        try:
            # Remove thousand separators (periods in Spanish format)
            width_str = match.group(1).replace('.', '').replace(',', '.')
            height_str = match.group(2).replace('.', '').replace(',', '.')

            width = int(float(width_str))
            height = int(float(height_str))

            return width * height
        except ValueError:
            pass

    return np.nan


def _extraer_tasa_refresco(tasa_str: str) -> Optional[float]:
    """
    Extract refresh rate in Hz from strings like "120 Hz" or "144 Hz".
    """
    if pd.isna(tasa_str):
        return np.nan

    tasa_str = str(tasa_str)

    # Match pattern like "120" before "Hz"
    match = re.search(r'(\d+)\s*Hz', tasa_str, re.IGNORECASE)

    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass

    return np.nan


def _tiene_wifi(conectividad_str: str) -> Optional[int]:
    """
    Binary flag: 1 if has WiFi, 0 if not, None if unknown.

    Looks for 'wifi' keyword in connectivity string.
    """
    if pd.isna(conectividad_str):
        return np.nan

    conectividad_str = str(conectividad_str).lower()

    return 1 if 'wifi' in conectividad_str else 0


def _tiene_bluetooth(conectividad_str: str) -> Optional[int]:
    """
    Binary flag: 1 if has Bluetooth, 0 if not, None if unknown.

    Looks for 'bluetooth' keyword in connectivity string.
    """
    if pd.isna(conectividad_str):
        return np.nan

    conectividad_str = str(conectividad_str).lower()

    return 1 if 'bluetooth' in conectividad_str else 0


def _tiene_webcam(webcam_str: str) -> Optional[int]:
    """
    Binary flag: 1 if has webcam, 0 if not, None if unknown.

    Looks for 'integrada' keyword or megapixel values.
    Returns 0 for 'ninguna' or 'no'.
    """
    if pd.isna(webcam_str):
        return np.nan

    webcam_str = str(webcam_str).lower()

    # Explicit no webcam
    if any(word in webcam_str for word in ['ninguna', 'ninguno', 'no']):
        return 0

    # Has webcam
    if any(word in webcam_str for word in ['integrada', 'megapixel', 'mp', 'hd', 'fhd']):
        return 1

    return np.nan


def _extraer_version_bluetooth(bluetooth_str: str) -> Optional[float]:
    """
    Extract Bluetooth version as numeric value.

    Examples:
    - "Bluetooth 5.3" -> 5.3
    - "Bluetooth 5.1" -> 5.1
    """
    if pd.isna(bluetooth_str):
        return np.nan

    bluetooth_str = str(bluetooth_str)

    # Match pattern like "5.3" or "5.1" after "Bluetooth"
    match = re.search(r'bluetooth\s*(\d+\.\d+)', bluetooth_str, re.IGNORECASE)

    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass

    return np.nan


def _extraer_cpu_cores(cores_str: str) -> Optional[float]:
    """
    Extract number of CPU cores as numeric value.

    Examples:
    - "8 (4P + 4E)" -> 8
    - "14" -> 14
    - "6 núcleos" -> 6
    """
    if pd.isna(cores_str):
        return np.nan

    cores_str = str(cores_str)

    # Look for first number (total cores)
    match = re.search(r'(\d+)', cores_str)

    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass

    return np.nan


def _extraer_gpu_memory_gb(gpu_mem_str: str) -> Optional[float]:
    """
    Extract GPU memory in GB.

    Examples:
    - "8 GB GDDR6" -> 8.0
    - "4096 MB" -> 4.0
    - "16 GB" -> 16.0
    """
    if pd.isna(gpu_mem_str):
        return np.nan

    gpu_mem_str = str(gpu_mem_str).upper()

    # Check for GB first
    match_gb = re.search(r'(\d+)\s*GB', gpu_mem_str)
    if match_gb:
        try:
            return float(match_gb.group(1))
        except ValueError:
            pass

    # Check for MB (convert to GB)
    match_mb = re.search(r'(\d+)\s*MB', gpu_mem_str)
    if match_mb:
        try:
            return float(match_mb.group(1)) / 1024
        except ValueError:
            pass

    return np.nan


def _extraer_num_ofertas(ofertas_str: str) -> Optional[int]:
    """
    Extract number of offers from string.

    Examples:
    - "200 ofertas:" -> 200
    - "50 ofertas" -> 50
    """
    if pd.isna(ofertas_str):
        return np.nan

    match = re.search(r'(\d+)\s+ofertas?', str(ofertas_str), re.IGNORECASE)

    if match:
        try:
            return int(match.group(1))
        except ValueError:
            pass

    return np.nan
