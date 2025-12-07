"""
features.py - Simplified Feature Engineering Module

This module contains functions for loading data, analyzing data quality,
and engineering features for the computer price prediction model.

CONVENTIONS:
- Original Spanish column names from CSV files are kept intact
- All NEW engineered features have names starting with underscore (_)
"""

import pandas as pd
import numpy as np
import re
import unicodedata
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from collections import defaultdict
from difflib import SequenceMatcher, get_close_matches


# =============================================================================
# DATA LOADING
# =============================================================================

def cargar_datos(ruta_computers: str, ruta_cpu: str, ruta_gpu: str) -> tuple:
    """Load the three main data files."""
    df_computers = pd.read_csv(
        ruta_computers, encoding='utf-8-sig', index_col=0, low_memory=False
    ).reset_index(drop=True)

    df_cpu = pd.read_csv(
        ruta_cpu, encoding='utf-8-sig', index_col=0
    ).reset_index(drop=True)

    df_gpu = pd.read_csv(
        ruta_gpu, encoding='utf-8-sig', index_col=0
    ).reset_index(drop=True)

    print(f"Loaded {len(df_computers):,} computer listings")
    print(f"Loaded {len(df_cpu):,} CPU benchmarks")
    print(f"Loaded {len(df_gpu):,} GPU benchmarks")

    return df_computers, df_cpu, df_gpu


# =============================================================================
# DATA QUALITY ANALYSIS
# =============================================================================

def detect_mixed_types(series: pd.Series) -> Tuple[bool, List[str]]:
    """Detect columns where numeric-looking values are mixed with text."""
    mixed = False
    text_samples = []

    for val in series.dropna().astype(str).head(200):
        cleaned = re.sub(r"[.,0-9\-+eE ]", "", val)
        if cleaned.strip():
            mixed = True
            text_samples.append(val)

    return mixed, text_samples[:5]


def detect_multilabel(series: pd.Series) -> int:
    """Detect fields with multiple values separated by commas, slashes, etc."""
    multilabel_separators = r",|/|;|\|"
    return series.astype(str).str.contains(multilabel_separators).sum()


def detect_combined_fields(series: pd.Series) -> int:
    """Detect fields containing multiple attributes (e.g., '16GB DDR4 3200MHz')."""
    pattern = r"\b\d+(\.\d+)?\s*(GB|TB|MHz|cm|W|%)\b"
    return series.astype(str).str.contains(pattern, flags=re.IGNORECASE).sum()


def analyze_format_issues(df: pd.DataFrame, df_name: str = "DataFrame") -> pd.DataFrame:
    """Analyze format issues in all columns of a dataframe."""
    print(f"\nAnalyzing: {df_name}")
    print("-" * 60)

    results = []
    for col in df.columns:
        series = df[col]
        mixed, samples = detect_mixed_types(series) if series.dtype == 'object' else (False, [])
        multilabel_count = detect_multilabel(series) if series.dtype == 'object' else 0
        combined_count = detect_combined_fields(series) if series.dtype == 'object' else 0

        results.append({
            "column": col,
            "mixed_numeric_text": mixed,
            "sample_text_values": samples,
            "multilabel_rows": multilabel_count,
            "combined_field_rows": combined_count,
        })

    return pd.DataFrame(results)


def group_columns_by_prefix(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Group columns by their prefix (before underscore)."""
    groups = defaultdict(list)
    for col in df.columns:
        prefix = col.split("_")[0]
        groups[prefix].append(col)
    return dict(groups)


def print_column_groups(df: pd.DataFrame):
    """Print column groups that might be merge candidates."""
    groups = group_columns_by_prefix(df)
    print("\n=== Column Groups by Prefix ===")
    for prefix, cols in sorted(groups.items()):
        if len(cols) > 1:
            print(f"\nGROUP: {prefix}")
            for c in cols:
                print(f"   - {c}")


# =============================================================================
# TEXT NORMALIZATION UTILITIES
# =============================================================================

def strip_accents(text: str) -> str:
    """Remove accents from text."""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def normalize_text(raw: Optional[str], noise_patterns: List[str] = None) -> str:
    """Normalize text: lowercase, remove accents, special chars, and noise patterns."""
    if not isinstance(raw, str):
        return ""

    text = strip_accents(raw).lower()
    text = text.replace("®", " ").replace("™", " ")
    text = re.sub(r"\(.*?\)", " ", text)  # Remove parentheses content
    text = re.sub(r"[\-/]", " ", text)

    if noise_patterns:
        for pattern in noise_patterns:
            text = re.sub(pattern, " ", text, flags=re.IGNORECASE)

    text = re.sub(r"[^a-z0-9 ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# =============================================================================
# CPU PARSING AND MATCHING
# =============================================================================

CPU_NOISE_PATTERNS = [
    r"procesador", r"processor", r"with.*graphics", r"grafica integrada",
    r"quad-?core", r"hexa-?core", r"octa-?core", r"dodeca-?core",
    r"\bup to\b", r"\bmax\.?\b",
]

CPU_BRAND_PATTERNS = {
    "intel": r"\bintel\b|core i[3579]|celeron|pentium|xeon|atom",
    "amd": r"\bamd\b|ryzen|athlon|threadripper|epyc",
    "apple": r"\bapple\b|\bm[1234]\b",
    "qualcomm": r"\bqualcomm\b|snapdragon",
}

INTEL_FAMILY_PATTERNS = [
    (re.compile(r"core\s*i\s*([3579])"), lambda m: f"core i{m.group(1)}"),
    (re.compile(r"core\s*ultra\s*(\d)"), lambda m: f"core ultra {m.group(1)}"),
    (re.compile(r"core\s*(duo|solo|m)"), lambda m: f"core {m.group(1)}"),
    (re.compile(r"xeon"), lambda _: "xeon"),
    (re.compile(r"celeron"), lambda _: "celeron"),
    (re.compile(r"pentium"), lambda _: "pentium"),
    (re.compile(r"atom"), lambda _: "atom"),
]

AMD_FAMILY_PATTERNS = [
    (re.compile(r"ryzen\s*(ai\s*)?(\d)"), lambda m: f"ryzen {m.group(2)}"),
    (re.compile(r"threadripper"), lambda _: "ryzen threadripper"),
    (re.compile(r"athlon"), lambda _: "athlon"),
    (re.compile(r"epyc"), lambda _: "epyc"),
]

APPLE_FAMILY_PATTERNS = [
    (re.compile(r"m(\d)\s*(pro|max|ultra)?"), lambda m: f"m{m.group(1)}" + (f" {m.group(2)}" if m.group(2) else "")),
]

QUALCOMM_FAMILY_PATTERNS = [
    (re.compile(r"snapdragon\s*x?\s*(\w+)"), lambda m: f"snapdragon {m.group(1)}"),
]

CPU_MODEL_PATTERN = re.compile(r"\b(\d{3,5})([a-z]{0,2})\b")
CPU_SUFFIX_CANDIDATES = {"k", "kf", "ks", "f", "g", "t", "p", "u", "h", "hx", "hk", "hs", "he", "x"}


def detect_cpu_brand(norm_text: str) -> Optional[str]:
    """Detect CPU brand from normalized text."""
    for brand, pattern in CPU_BRAND_PATTERNS.items():
        if re.search(pattern, norm_text):
            return brand
    return None


def detect_cpu_family(norm_text: str, brand: Optional[str]) -> Optional[str]:
    """Detect CPU family based on brand."""
    family_patterns = {
        "intel": INTEL_FAMILY_PATTERNS,
        "amd": AMD_FAMILY_PATTERNS,
        "apple": APPLE_FAMILY_PATTERNS,
        "qualcomm": QUALCOMM_FAMILY_PATTERNS,
    }.get(brand, INTEL_FAMILY_PATTERNS + AMD_FAMILY_PATTERNS + APPLE_FAMILY_PATTERNS)

    for pattern, formatter in family_patterns:
        match = pattern.search(norm_text)
        if match:
            return formatter(match)
    return None


def detect_cpu_model_and_suffix(norm_text: str) -> Dict[str, Optional[str]]:
    """Extract model code and suffix from normalized CPU text."""
    model_code = None
    suffix = None

    for match in CPU_MODEL_PATTERN.finditer(norm_text):
        candidate = match.group(1)
        letters = match.group(2)
        if len(candidate) >= 4:  # Prefer 4-5 digit model codes
            model_code = candidate + letters
            break
        if not model_code:
            model_code = candidate + letters

    if model_code:
        # Look for suffix after model code
        trailing = re.findall(r"\b([a-z]{1,2})\b", norm_text)
        for candidate in reversed(trailing):
            if candidate in CPU_SUFFIX_CANDIDATES:
                suffix = candidate
                break

    return {"model_code": model_code, "suffix": suffix}


def build_cpu_key(brand: Optional[str], family: Optional[str],
                  model_code: Optional[str], suffix: Optional[str]) -> Optional[str]:
    """Build normalized CPU key from components."""
    parts = [p for p in [brand, family, model_code] if p]
    key = " ".join(parts).strip()
    if suffix and suffix not in key.split():
        key = f"{key} {suffix}".strip()
    return key or None


def parse_cpu_name(raw: Optional[str]) -> Dict[str, Optional[str]]:
    """Parse CPU name into structured components."""
    norm_text = normalize_text(raw, CPU_NOISE_PATTERNS)

    if not norm_text:
        return {
            "cpu_name_clean": None,
            "cpu_brand": None,
            "cpu_family": None,
            "cpu_model_code": None,
            "cpu_suffix": None,
            "cpu_normalized_key": None,
            "cpu_parse_status": "empty",
        }

    brand = detect_cpu_brand(norm_text)
    family = detect_cpu_family(norm_text, brand)
    model_info = detect_cpu_model_and_suffix(norm_text)
    model_code = model_info["model_code"]
    suffix = model_info["suffix"]
    normalized_key = build_cpu_key(brand, family, model_code, suffix)

    return {
        "cpu_name_clean": norm_text,
        "cpu_brand": brand,
        "cpu_family": family,
        "cpu_model_code": model_code,
        "cpu_suffix": suffix,
        "cpu_normalized_key": normalized_key or norm_text,
        "cpu_parse_status": "ok" if normalized_key else "needs_review",
    }


def find_close_model_number(model_code: str, available_models: List[str], max_delta: int = 10) -> Optional[str]:
    """
    Find a close model number from available models.

    For example, if looking for '617' and we have ['615', '615e', '620'],
    this will return the closest one within max_delta.
    """
    if not model_code or not available_models:
        return None

    # Extract numeric part from model code
    base_match = re.match(r'(\d+)', model_code)
    if not base_match:
        return None

    base_num = int(base_match.group(1))

    # Find closest match
    best_match = None
    best_delta = float('inf')

    for candidate in available_models:
        cand_match = re.match(r'(\d+)', candidate)
        if cand_match:
            cand_num = int(cand_match.group(1))
            delta = abs(cand_num - base_num)
            if delta <= max_delta and delta < best_delta:
                best_delta = delta
                best_match = candidate

    return best_match


def build_cpu_lookup(df_cpu: pd.DataFrame) -> pd.DataFrame:
    """Build CPU benchmark lookup table with normalized keys."""
    # Parse CPU names in benchmark data
    cpu_parsed = df_cpu["CPU Name"].apply(parse_cpu_name).apply(pd.Series)
    df_cpu_enhanced = pd.concat([df_cpu, cpu_parsed], axis=1)

    # Rename benchmark columns
    bench_cols = {
        "CPU Name": "cpu_bench_name",
        "CPU Mark (higher is better)": "cpu_bench_mark",
        "Rank (lower is better)": "cpu_bench_rank",
        "CPU Value (higher is better)": "cpu_bench_value",
        "Price (USD)": "cpu_bench_price_usd",
    }

    # Keep best benchmark per normalized key
    df_lookup = (
        df_cpu_enhanced.rename(columns=bench_cols)
        .sort_values("cpu_bench_mark", ascending=False)
        .drop_duplicates(subset=["cpu_normalized_key"], keep="first")
    )

    return df_lookup[[
        "cpu_normalized_key", "cpu_bench_name", "cpu_bench_mark",
        "cpu_bench_rank", "cpu_bench_value", "cpu_bench_price_usd",
        "cpu_brand", "cpu_family", "cpu_model_code"
    ]]


def match_cpu_benchmarks(df_comp: pd.DataFrame, df_cpu: pd.DataFrame,
                         cpu_col: str = "Procesador_Procesador",
                         fuzzy_cutoff: float = 0.82) -> pd.DataFrame:
    """Match CPU benchmarks using exact + fuzzy + close neighbor + family mean."""
    df = df_comp.copy()

    # Parse CPU names in computer data
    print("Parsing CPU names...")
    cpu_parsed = df[cpu_col].apply(parse_cpu_name).apply(pd.Series)
    for col in cpu_parsed.columns:
        df[col] = cpu_parsed[col]

    # Build lookup table
    print("Building CPU lookup table...")
    cpu_lookup = build_cpu_lookup(df_cpu)

    # Exact matching
    print("Performing exact CPU matches...")
    df = df.merge(cpu_lookup, on="cpu_normalized_key", how="left", suffixes=("", "_bench"))
    df["cpu_match_strategy"] = np.where(df["cpu_bench_name"].notna(), "exact", "unmatched")
    df["cpu_match_score"] = np.where(df["cpu_bench_name"].notna(), 1.0, np.nan)

    exact_rate = (df["cpu_match_strategy"] == "exact").mean()
    print(f"  Exact matches: {exact_rate:.1%}")

    # Fuzzy matching for unmatched
    print("Performing fuzzy CPU matches...")
    lookup_keys = cpu_lookup["cpu_normalized_key"].dropna().unique().tolist()
    unmatched_keys = df.loc[
        df["cpu_match_strategy"] == "unmatched", "cpu_normalized_key"
    ].dropna().unique().tolist()

    fuzzy_matches = []
    for key in unmatched_keys:
        matches = get_close_matches(key, lookup_keys, n=1, cutoff=fuzzy_cutoff)
        if matches:
            score = SequenceMatcher(None, key, matches[0]).ratio()
            fuzzy_matches.append({
                "cpu_normalized_key": key,
                "cpu_bench_match_key": matches[0],
                "cpu_match_score_fuzzy": round(score, 4),
            })

    if fuzzy_matches:
        fuzzy_df = pd.DataFrame(fuzzy_matches)
        lookup_for_fuzzy = cpu_lookup.rename(columns={"cpu_normalized_key": "cpu_bench_match_key"})
        fuzzy_df = fuzzy_df.merge(lookup_for_fuzzy, on="cpu_bench_match_key", how="left")

        bench_cols = ["cpu_bench_name", "cpu_bench_mark", "cpu_bench_rank",
                      "cpu_bench_value", "cpu_bench_price_usd"]
        fuzzy_df = fuzzy_df.rename(columns={col: f"{col}_fuzzy" for col in bench_cols})
        df = df.merge(fuzzy_df, on="cpu_normalized_key", how="left")

        # Fill in fuzzy matches
        for col in bench_cols:
            df[col] = df[col].fillna(df.get(f"{col}_fuzzy"))
            if f"{col}_fuzzy" in df.columns:
                df.drop(columns=[f"{col}_fuzzy"], inplace=True)

        df["cpu_match_score"] = df["cpu_match_score"].fillna(df.get("cpu_match_score_fuzzy"))
        if "cpu_match_score_fuzzy" in df.columns:
            df.drop(columns=["cpu_match_score_fuzzy"], inplace=True)

        mask = df["cpu_match_strategy"] == "unmatched"
        df.loc[mask, "cpu_match_strategy"] = np.where(
            df.loc[mask, "cpu_bench_name"].notna(), "fuzzy", "unmatched"
        )

        if "cpu_bench_match_key" in df.columns:
            df.drop(columns=["cpu_bench_match_key"], inplace=True)

    # Close neighbor matching for remaining unmatched
    print("Performing close neighbor CPU matches...")
    still_unmatched = df["cpu_match_strategy"] == "unmatched"

    if still_unmatched.sum() > 0:
        neighbor_matches = []

        for idx in df[still_unmatched].index:
            brand = df.loc[idx, "cpu_brand"]
            family = df.loc[idx, "cpu_family"]
            model_code = df.loc[idx, "cpu_model_code"]

            if pd.isna(brand) or pd.isna(family) or pd.isna(model_code):
                continue

            # Find CPUs with same brand and family in lookup
            same_family = cpu_lookup[
                (cpu_lookup["cpu_brand"] == brand) &
                (cpu_lookup["cpu_family"] == family)
            ]

            if len(same_family) == 0:
                continue

            # Try to find close model number
            available_models = same_family["cpu_model_code"].dropna().tolist()
            close_model = find_close_model_number(model_code, available_models)

            if close_model:
                match_row = same_family[same_family["cpu_model_code"] == close_model].iloc[0]
                neighbor_matches.append({
                    "idx": idx,
                    "cpu_bench_name": match_row["cpu_bench_name"],
                    "cpu_bench_mark": match_row["cpu_bench_mark"],
                    "cpu_bench_rank": match_row["cpu_bench_rank"],
                    "cpu_bench_value": match_row["cpu_bench_value"],
                    "cpu_bench_price_usd": match_row["cpu_bench_price_usd"],
                    "strategy": "neighbor",
                })

        # Apply neighbor matches
        for match in neighbor_matches:
            idx = match["idx"]
            df.loc[idx, "cpu_bench_name"] = match["cpu_bench_name"]
            df.loc[idx, "cpu_bench_mark"] = match["cpu_bench_mark"]
            df.loc[idx, "cpu_bench_rank"] = match["cpu_bench_rank"]
            df.loc[idx, "cpu_bench_value"] = match["cpu_bench_value"]
            df.loc[idx, "cpu_bench_price_usd"] = match["cpu_bench_price_usd"]
            df.loc[idx, "cpu_match_strategy"] = "neighbor"
            df.loc[idx, "cpu_match_score"] = 0.9

        print(f"  Neighbor matches: {len(neighbor_matches)}")

    # Family mean imputation for remaining unmatched
    print("Computing family mean for remaining unmatched CPUs...")
    still_unmatched = df["cpu_match_strategy"] == "unmatched"

    if still_unmatched.sum() > 0:
        # Compute family means from lookup table
        family_means = cpu_lookup.groupby(["cpu_brand", "cpu_family"]).agg({
            "cpu_bench_mark": "mean",
            "cpu_bench_price_usd": "mean",
        }).reset_index()

        # Compute brand-level means as fallback
        brand_means = cpu_lookup.groupby("cpu_brand").agg({
            "cpu_bench_mark": "mean",
            "cpu_bench_price_usd": "mean",
        }).reset_index()

        family_imputed = 0
        brand_imputed = 0

        for idx in df[still_unmatched].index:
            brand = df.loc[idx, "cpu_brand"]
            family = df.loc[idx, "cpu_family"]

            if pd.isna(brand):
                continue

            # Try family mean first
            if pd.notna(family):
                family_match = family_means[
                    (family_means["cpu_brand"] == brand) &
                    (family_means["cpu_family"] == family)
                ]
                if len(family_match) > 0:
                    df.loc[idx, "cpu_bench_mark"] = family_match["cpu_bench_mark"].iloc[0]
                    df.loc[idx, "cpu_bench_price_usd"] = family_match["cpu_bench_price_usd"].iloc[0]
                    df.loc[idx, "cpu_match_strategy"] = "family_mean"
                    df.loc[idx, "cpu_match_score"] = 0.7
                    family_imputed += 1
                    continue

            # Fall back to brand mean
            brand_match = brand_means[brand_means["cpu_brand"] == brand]
            if len(brand_match) > 0:
                df.loc[idx, "cpu_bench_mark"] = brand_match["cpu_bench_mark"].iloc[0]
                df.loc[idx, "cpu_bench_price_usd"] = brand_match["cpu_bench_price_usd"].iloc[0]
                df.loc[idx, "cpu_match_strategy"] = "brand_mean"
                df.loc[idx, "cpu_match_score"] = 0.5
                brand_imputed += 1

        print(f"  Family mean imputations: {family_imputed}")
        print(f"  Brand mean imputations: {brand_imputed}")

    # Clean up extra columns from lookup merge
    extra_cols = ["cpu_brand_bench", "cpu_family_bench", "cpu_model_code_bench"]
    for col in extra_cols:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    # Report results
    print("\nCPU match strategy distribution:")
    print(df["cpu_match_strategy"].value_counts(dropna=False))
    coverage = df["cpu_bench_mark"].notna().mean()
    print(f"\nCPU benchmark coverage: {coverage:.1%}")

    return df


# =============================================================================
# GPU PARSING AND MATCHING
# =============================================================================

GPU_NOISE_PATTERNS = [
    r"tarjeta grafica", r"graphics card", r"graphic card",
    r"gddr\d", r"integrada", r"dedicada", r"video",
]

GPU_BRAND_PATTERNS = {
    "nvidia": r"nvidia|geforce|rtx|gtx|mx|tesla|quadro",
    "amd": r"\bamd\b|radeon|rx\s*\d|vega",
    "intel": r"intel|iris|uhd|arc\s*a",
    "apple": r"apple|m\d",
}

GPU_SERIES_PATTERNS = [
    (re.compile(r"rtx\s*(\d{3,4})"), lambda m: ("rtx", m.group(1))),
    (re.compile(r"gtx\s*(\d{3,4})"), lambda m: ("gtx", m.group(1))),
    (re.compile(r"mx\s*(\d{3,4})"), lambda m: ("mx", m.group(1))),
    (re.compile(r"\brx\s*(\d{3,4})"), lambda m: ("rx", m.group(1))),
    (re.compile(r"arc\s*(a\d{3})"), lambda m: ("arc", m.group(1))),
    (re.compile(r"iris\s*(xe|pro|plus)"), lambda m: ("iris", m.group(1))),
    (re.compile(r"uhd\s*(\d{3})"), lambda m: ("uhd", m.group(1))),
    (re.compile(r"vega\s*(\d{1,2})"), lambda m: ("vega", m.group(1))),
]

GPU_SUFFIX_MAP = {
    "ti": "ti", "super": "super", "max q": "max-q", "max-q": "max-q",
    "maxq": "max-q", "mobile": "mobile", "laptop": "mobile",
    "xt": "xt", "pro": "pro",
}


def detect_gpu_brand(norm_text: str) -> Optional[str]:
    """Detect GPU brand from normalized text."""
    for brand, pattern in GPU_BRAND_PATTERNS.items():
        if re.search(pattern, norm_text):
            return brand
    return None


def detect_gpu_series_and_model(norm_text: str) -> Dict[str, Optional[str]]:
    """Extract GPU series and model number."""
    for pattern, formatter in GPU_SERIES_PATTERNS:
        match = pattern.search(norm_text)
        if match:
            series, model = formatter(match)
            return {"series": series.strip(), "model": model.strip()}

    # Fallback: try to find any 3-4 digit model number
    model_match = re.search(r"\b(\d{3,4})\b", norm_text)
    if model_match:
        return {"series": None, "model": model_match.group(1)}

    return {"series": None, "model": None}


def detect_gpu_suffix(norm_text: str) -> Optional[str]:
    """Detect GPU suffix (ti, super, xt, etc.)."""
    for raw_suffix, normalized in GPU_SUFFIX_MAP.items():
        if re.search(rf"\b{re.escape(raw_suffix)}\b", norm_text):
            return normalized
    return None


def build_gpu_key(brand: Optional[str], series: Optional[str],
                  model: Optional[str], suffix: Optional[str]) -> Optional[str]:
    """Build normalized GPU key from components."""
    parts = [p for p in [brand, series, model] if p]
    key = " ".join(parts).strip()
    if suffix and suffix not in key.split():
        key = f"{key} {suffix}".strip()
    return key or None


def is_integrated_gpu(norm_text: str) -> bool:
    """Check if GPU is integrated (not discrete)."""
    integrated_patterns = [
        r"intel\s*(arc|uhd|iris|graphics)(?!\s*a\d{3})",  # Intel integrated (but not Arc A-series)
        r"amd\s*radeon\s*graphics$",  # AMD integrated
        r"apple\s*m\d",  # Apple integrated
        r"qualcomm\s*adreno",  # Qualcomm integrated
    ]
    for pattern in integrated_patterns:
        if re.search(pattern, norm_text):
            return True
    return False


def parse_gpu_name(raw: Optional[str]) -> Dict[str, Optional[str]]:
    """Parse GPU name into structured components."""
    norm_text = normalize_text(raw, GPU_NOISE_PATTERNS)

    if not norm_text:
        return {
            "gpu_name_clean": None,
            "gpu_brand": None,
            "gpu_series": None,
            "gpu_model_number": None,
            "gpu_suffix": None,
            "gpu_normalized_key": None,
            "gpu_is_integrated": None,
            "gpu_parse_status": "empty",
        }

    brand = detect_gpu_brand(norm_text)
    is_integrated = is_integrated_gpu(norm_text)
    series_model = detect_gpu_series_and_model(norm_text)
    suffix = detect_gpu_suffix(norm_text)
    normalized_key = build_gpu_key(brand, series_model["series"], series_model["model"], suffix)

    return {
        "gpu_name_clean": norm_text,
        "gpu_brand": brand,
        "gpu_series": series_model["series"],
        "gpu_model_number": series_model["model"],
        "gpu_suffix": suffix,
        "gpu_normalized_key": normalized_key or norm_text,
        "gpu_is_integrated": is_integrated,
        "gpu_parse_status": "ok" if normalized_key else "needs_review",
    }


def build_gpu_lookup(df_gpu: pd.DataFrame) -> pd.DataFrame:
    """Build GPU benchmark lookup table with normalized keys."""
    # Parse GPU names in benchmark data
    gpu_parsed = df_gpu["Videocard Name"].apply(parse_gpu_name).apply(pd.Series)
    df_gpu_enhanced = pd.concat([df_gpu, gpu_parsed], axis=1)

    # Rename benchmark columns
    bench_cols = {
        "Videocard Name": "gpu_bench_name",
        "Passmark G3D Mark (higher is better)": "gpu_bench_mark",
        "Rank (lower is better)": "gpu_bench_rank",
        "Videocard Value (higher is better)": "gpu_bench_value",
        "Price (USD)": "gpu_bench_price_usd",
    }

    # Keep best benchmark per normalized key
    df_lookup = (
        df_gpu_enhanced.rename(columns=bench_cols)
        .sort_values("gpu_bench_mark", ascending=False)
        .drop_duplicates(subset=["gpu_normalized_key"], keep="first")
    )

    return df_lookup[[
        "gpu_normalized_key", "gpu_bench_name", "gpu_bench_mark",
        "gpu_bench_rank", "gpu_bench_value", "gpu_bench_price_usd",
        "gpu_brand", "gpu_series", "gpu_model_number"
    ]]


def match_gpu_benchmarks(df_comp: pd.DataFrame, df_gpu: pd.DataFrame,
                         gpu_col: str = "Gráfica_Tarjeta gráfica",
                         fuzzy_cutoff: float = 0.80) -> pd.DataFrame:
    """Match GPU benchmarks using exact + fuzzy + close neighbor + series mean."""
    df = df_comp.copy()

    # Parse GPU names in computer data
    print("Parsing GPU names...")
    gpu_parsed = df[gpu_col].apply(parse_gpu_name).apply(pd.Series)
    for col in gpu_parsed.columns:
        df[col] = gpu_parsed[col]

    # Build lookup table
    print("Building GPU lookup table...")
    gpu_lookup = build_gpu_lookup(df_gpu)

    # Exact matching (only for discrete GPUs)
    print("Performing exact GPU matches...")
    df = df.merge(gpu_lookup, on="gpu_normalized_key", how="left", suffixes=("", "_bench"))

    # Mark integrated GPUs as "integrated" strategy, others as exact/unmatched
    df["gpu_match_strategy"] = np.where(
        df["gpu_is_integrated"] == True, "integrated",
        np.where(df["gpu_bench_name"].notna(), "exact", "unmatched")
    )
    df["gpu_match_score"] = np.where(df["gpu_bench_name"].notna(), 1.0, np.nan)

    exact_rate = (df["gpu_match_strategy"] == "exact").mean()
    integrated_rate = (df["gpu_match_strategy"] == "integrated").mean()
    print(f"  Exact matches: {exact_rate:.1%}")
    print(f"  Integrated GPUs (skipped): {integrated_rate:.1%}")

    # Fuzzy matching for unmatched discrete GPUs
    print("Performing fuzzy GPU matches...")
    lookup_keys = gpu_lookup["gpu_normalized_key"].dropna().unique().tolist()
    unmatched_mask = (df["gpu_match_strategy"] == "unmatched") & (df["gpu_is_integrated"] != True)
    unmatched_keys = df.loc[unmatched_mask, "gpu_normalized_key"].dropna().unique().tolist()

    fuzzy_matches = []
    for key in unmatched_keys:
        matches = get_close_matches(key, lookup_keys, n=1, cutoff=fuzzy_cutoff)
        if matches:
            score = SequenceMatcher(None, key, matches[0]).ratio()
            fuzzy_matches.append({
                "gpu_normalized_key": key,
                "gpu_bench_match_key": matches[0],
                "gpu_match_score_fuzzy": round(score, 4),
            })

    if fuzzy_matches:
        fuzzy_df = pd.DataFrame(fuzzy_matches)
        lookup_for_fuzzy = gpu_lookup.rename(columns={"gpu_normalized_key": "gpu_bench_match_key"})
        fuzzy_df = fuzzy_df.merge(lookup_for_fuzzy, on="gpu_bench_match_key", how="left")

        bench_cols = ["gpu_bench_name", "gpu_bench_mark", "gpu_bench_rank",
                      "gpu_bench_value", "gpu_bench_price_usd"]
        fuzzy_df = fuzzy_df.rename(columns={col: f"{col}_fuzzy" for col in bench_cols})
        df = df.merge(fuzzy_df, on="gpu_normalized_key", how="left")

        # Fill in fuzzy matches
        for col in bench_cols:
            df[col] = df[col].fillna(df.get(f"{col}_fuzzy"))
            if f"{col}_fuzzy" in df.columns:
                df.drop(columns=[f"{col}_fuzzy"], inplace=True)

        df["gpu_match_score"] = df["gpu_match_score"].fillna(df.get("gpu_match_score_fuzzy"))
        if "gpu_match_score_fuzzy" in df.columns:
            df.drop(columns=["gpu_match_score_fuzzy"], inplace=True)

        mask = df["gpu_match_strategy"] == "unmatched"
        df.loc[mask, "gpu_match_strategy"] = np.where(
            df.loc[mask, "gpu_bench_name"].notna(), "fuzzy", "unmatched"
        )

        if "gpu_bench_match_key" in df.columns:
            df.drop(columns=["gpu_bench_match_key"], inplace=True)

    # Close neighbor matching for remaining unmatched discrete GPUs
    print("Performing close neighbor GPU matches...")
    still_unmatched = (df["gpu_match_strategy"] == "unmatched") & (df["gpu_is_integrated"] != True)

    if still_unmatched.sum() > 0:
        neighbor_matches = []

        for idx in df[still_unmatched].index:
            brand = df.loc[idx, "gpu_brand"]
            series = df.loc[idx, "gpu_series"]
            model_number = df.loc[idx, "gpu_model_number"]

            if pd.isna(brand) or pd.isna(series) or pd.isna(model_number):
                continue

            # Find GPUs with same brand and series in lookup
            same_series = gpu_lookup[
                (gpu_lookup["gpu_brand"] == brand) &
                (gpu_lookup["gpu_series"] == series)
            ]

            if len(same_series) == 0:
                continue

            # Try to find close model number
            available_models = same_series["gpu_model_number"].dropna().tolist()
            close_model = find_close_model_number(model_number, available_models, max_delta=20)

            if close_model:
                match_row = same_series[same_series["gpu_model_number"] == close_model].iloc[0]
                neighbor_matches.append({
                    "idx": idx,
                    "gpu_bench_name": match_row["gpu_bench_name"],
                    "gpu_bench_mark": match_row["gpu_bench_mark"],
                    "gpu_bench_rank": match_row["gpu_bench_rank"],
                    "gpu_bench_value": match_row["gpu_bench_value"],
                    "gpu_bench_price_usd": match_row["gpu_bench_price_usd"],
                    "strategy": "neighbor",
                })

        # Apply neighbor matches
        for match in neighbor_matches:
            idx = match["idx"]
            df.loc[idx, "gpu_bench_name"] = match["gpu_bench_name"]
            df.loc[idx, "gpu_bench_mark"] = match["gpu_bench_mark"]
            df.loc[idx, "gpu_bench_rank"] = match["gpu_bench_rank"]
            df.loc[idx, "gpu_bench_value"] = match["gpu_bench_value"]
            df.loc[idx, "gpu_bench_price_usd"] = match["gpu_bench_price_usd"]
            df.loc[idx, "gpu_match_strategy"] = "neighbor"
            df.loc[idx, "gpu_match_score"] = 0.9

        print(f"  Neighbor matches: {len(neighbor_matches)}")

    # Series mean imputation for remaining unmatched discrete GPUs
    print("Computing series mean for remaining unmatched GPUs...")
    still_unmatched = (df["gpu_match_strategy"] == "unmatched") & (df["gpu_is_integrated"] != True)

    if still_unmatched.sum() > 0:
        # Compute series means from lookup table
        series_means = gpu_lookup.groupby(["gpu_brand", "gpu_series"]).agg({
            "gpu_bench_mark": "mean",
            "gpu_bench_price_usd": "mean",
        }).reset_index()

        # Compute brand-level means as fallback
        brand_means = gpu_lookup.groupby("gpu_brand").agg({
            "gpu_bench_mark": "mean",
            "gpu_bench_price_usd": "mean",
        }).reset_index()

        series_imputed = 0
        brand_imputed = 0

        for idx in df[still_unmatched].index:
            brand = df.loc[idx, "gpu_brand"]
            series = df.loc[idx, "gpu_series"]

            if pd.isna(brand):
                continue

            # Try series mean first
            if pd.notna(series):
                series_match = series_means[
                    (series_means["gpu_brand"] == brand) &
                    (series_means["gpu_series"] == series)
                ]
                if len(series_match) > 0:
                    df.loc[idx, "gpu_bench_mark"] = series_match["gpu_bench_mark"].iloc[0]
                    df.loc[idx, "gpu_bench_price_usd"] = series_match["gpu_bench_price_usd"].iloc[0]
                    df.loc[idx, "gpu_match_strategy"] = "series_mean"
                    df.loc[idx, "gpu_match_score"] = 0.7
                    series_imputed += 1
                    continue

            # Fall back to brand mean
            brand_match = brand_means[brand_means["gpu_brand"] == brand]
            if len(brand_match) > 0:
                df.loc[idx, "gpu_bench_mark"] = brand_match["gpu_bench_mark"].iloc[0]
                df.loc[idx, "gpu_bench_price_usd"] = brand_match["gpu_bench_price_usd"].iloc[0]
                df.loc[idx, "gpu_match_strategy"] = "brand_mean"
                df.loc[idx, "gpu_match_score"] = 0.5
                brand_imputed += 1

        print(f"  Series mean imputations: {series_imputed}")
        print(f"  Brand mean imputations: {brand_imputed}")

    # Clean up extra columns from lookup merge
    extra_cols = ["gpu_brand_bench", "gpu_series_bench", "gpu_model_number_bench"]
    for col in extra_cols:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    # Report results
    print("\nGPU match strategy distribution:")
    print(df["gpu_match_strategy"].value_counts(dropna=False))
    discrete_mask = df["gpu_is_integrated"] != True
    if discrete_mask.sum() > 0:
        coverage = df.loc[discrete_mask, "gpu_bench_mark"].notna().mean()
        print(f"\nGPU benchmark coverage (discrete only): {coverage:.1%}")

    return df


# =============================================================================
# FEATURE EXTRACTION HELPERS
# =============================================================================

def extraer_precio_medio(precio_rango: str) -> Optional[float]:
    """Extract midpoint price from Spanish format range string."""
    if pd.isna(precio_rango) or not isinstance(precio_rango, str):
        return np.nan

    pattern = r'([\d.]+,\d{2})'
    matches = re.findall(pattern, precio_rango)

    if not matches:
        return np.nan

    precios = []
    for match in matches:
        num_str = match.replace('.', '').replace(',', '.')
        try:
            precios.append(float(num_str))
        except ValueError:
            continue

    if len(precios) == 0:
        return np.nan
    elif len(precios) == 1:
        return precios[0]
    else:
        return (precios[0] + precios[1]) / 2


def extraer_brand(titulo: str) -> Optional[str]:
    """Extract brand from product title."""
    if pd.isna(titulo):
        return np.nan

    common_brands = {
        'apple': 'Apple', 'asus': 'ASUS', 'lenovo': 'Lenovo', 'hp': 'HP',
        'dell': 'Dell', 'acer': 'Acer', 'msi': 'MSI', 'samsung': 'Samsung',
        'microsoft': 'Microsoft', 'razer': 'Razer', 'alienware': 'Alienware',
        'lg': 'LG', 'huawei': 'Huawei', 'xiaomi': 'Xiaomi', 'gigabyte': 'Gigabyte',
        'toshiba': 'Toshiba', 'fujitsu': 'Fujitsu', 'medion': 'Medion',
    }

    first_word = str(titulo).split()[0].lower() if titulo else ''
    return common_brands.get(first_word, first_word.capitalize() if first_word else np.nan)


def extraer_serie(serie_original: str, titulo: str, brand: str) -> Optional[str]:
    """Extract product series from Serie column or título."""
    if pd.notna(serie_original) and str(serie_original).strip():
        return str(serie_original).strip()

    if pd.isna(titulo) or pd.isna(brand):
        return np.nan

    titulo_lower = str(titulo).lower()

    series_patterns = {
        'Apple': ['MacBook Air', 'MacBook Pro', 'iMac', 'Mac Mini', 'Mac Pro'],
        'ASUS': ['ROG Zephyrus', 'ROG Strix', 'TUF Gaming', 'Zenbook', 'Vivobook', 'ProArt'],
        'Lenovo': ['ThinkPad', 'IdeaPad', 'Legion', 'LOQ', 'Yoga'],
        'HP': ['Pavilion', 'Envy', 'Omen', 'EliteBook', 'ProBook', 'Spectre', 'ZBook'],
        'Dell': ['Inspiron', 'XPS', 'Alienware', 'Latitude', 'Precision'],
        'MSI': ['Katana', 'Stealth', 'Raider', 'Cyborg', 'Modern', 'Sword'],
        'Acer': ['Aspire', 'Swift', 'Nitro', 'Predator'],
    }

    for series in sorted(series_patterns.get(brand, []), key=len, reverse=True):
        if series.lower() in titulo_lower:
            return series

    return np.nan


def extraer_numero(texto: str, pattern: str = r'(\d+(?:[.,]\d+)?)',
                   multiplier: float = 1.0, handle_spanish: bool = True) -> Optional[float]:
    """Generic number extraction from text."""
    if pd.isna(texto):
        return np.nan

    texto = str(texto)
    match = re.search(pattern, texto)

    if match:
        try:
            num_str = match.group(1)
            if handle_spanish:
                # Spanish format: 1.000 = 1000, 1,5 = 1.5
                if ',' in num_str and '.' in num_str:
                    num_str = num_str.replace('.', '').replace(',', '.')
                elif ',' in num_str:
                    num_str = num_str.replace(',', '.')
            return float(num_str) * multiplier
        except ValueError:
            pass
    return np.nan


def extraer_ram_gb(ram_str: str) -> Optional[float]:
    """Extract RAM size in GB."""
    if pd.isna(ram_str):
        return np.nan
    match = re.search(r'(\d+)\s*GB', str(ram_str).upper())
    return float(match.group(1)) if match else np.nan


def extraer_ssd_gb(ssd_str: str) -> Optional[float]:
    """Extract SSD capacity in GB (converts TB to GB)."""
    if pd.isna(ssd_str):
        return np.nan

    ssd_str = str(ssd_str).upper()

    # Check TB first
    tb_match = re.search(r'([\d.,]+)\s*TB', ssd_str)
    if tb_match:
        val = tb_match.group(1).replace('.', '').replace(',', '.')
        try:
            return float(val) * 1024
        except ValueError:
            pass

    # Check GB
    gb_match = re.search(r'([\d.,]+)\s*GB', ssd_str)
    if gb_match:
        val = gb_match.group(1).replace('.', '').replace(',', '.')
        try:
            return float(val)
        except ValueError:
            pass

    return np.nan


def extraer_pantalla_pulgadas(pulgadas_str: str, cm_str: str, titulo: str = None) -> Optional[float]:
    """Extract screen size in inches from multiple sources."""
    # Try pulgadas first
    if pd.notna(pulgadas_str):
        match = re.search(r'([\d,\.]+)', str(pulgadas_str))
        if match:
            try:
                return float(match.group(1).replace(',', '.'))
            except ValueError:
                pass

    # Fallback to cm conversion
    if pd.notna(cm_str):
        match = re.search(r'([\d,\.]+)', str(cm_str))
        if match:
            try:
                return float(match.group(1).replace(',', '.')) / 2.54
            except ValueError:
                pass

    # Fallback to título extraction
    if pd.notna(titulo):
        match = re.search(r'(\d{2}(?:[.,]\d)?)\s*(?:"|pulgadas|pulg)', str(titulo), re.IGNORECASE)
        if match:
            try:
                return float(match.group(1).replace(',', '.'))
            except ValueError:
                pass

    return np.nan


def extraer_resolucion_pixeles(resolucion_str: str) -> Optional[int]:
    """Extract total pixels from resolution (width x height)."""
    if pd.isna(resolucion_str):
        return np.nan

    match = re.search(r'([\d.]+)\s*x\s*([\d.]+)', str(resolucion_str), re.IGNORECASE)
    if match:
        try:
            width = int(float(match.group(1).replace('.', '')))
            height = int(float(match.group(2).replace('.', '')))
            return width * height
        except ValueError:
            pass
    return np.nan


def extraer_tasa_refresco(tasa_str: str) -> Optional[float]:
    """Extract refresh rate in Hz."""
    if pd.isna(tasa_str):
        return np.nan
    match = re.search(r'(\d+)\s*Hz', str(tasa_str), re.IGNORECASE)
    return float(match.group(1)) if match else np.nan


def extraer_peso_kg(peso_str: str) -> Optional[float]:
    """Extract weight in kg."""
    if pd.isna(peso_str):
        return np.nan
    match = re.search(r'([\d,\.]+)\s*kg', str(peso_str), re.IGNORECASE)
    if match:
        try:
            return float(match.group(1).replace(',', '.'))
        except ValueError:
            pass
    return np.nan


def extraer_cpu_cores(cores_str: str) -> Optional[float]:
    """Extract number of CPU cores."""
    if pd.isna(cores_str):
        return np.nan
    match = re.search(r'(\d+)', str(cores_str))
    return float(match.group(1)) if match else np.nan


def extraer_gpu_memory_gb(gpu_mem_str: str) -> Optional[float]:
    """Extract GPU memory in GB."""
    if pd.isna(gpu_mem_str):
        return np.nan

    gpu_mem_str = str(gpu_mem_str).upper()

    gb_match = re.search(r'(\d+)\s*GB', gpu_mem_str)
    if gb_match:
        return float(gb_match.group(1))

    mb_match = re.search(r'(\d+)\s*MB', gpu_mem_str)
    if mb_match:
        return float(mb_match.group(1)) / 1024

    return np.nan




def tiene_wifi(conectividad_str: str) -> Optional[int]:
    """Binary flag: 1 if has WiFi, 0 if not."""
    if pd.isna(conectividad_str):
        return np.nan
    return 1 if 'wifi' in str(conectividad_str).lower() else 0


def tiene_bluetooth(conectividad_str: str) -> Optional[int]:
    """Binary flag: 1 if has Bluetooth, 0 if not."""
    if pd.isna(conectividad_str):
        return np.nan
    return 1 if 'bluetooth' in str(conectividad_str).lower() else 0


def tiene_webcam(webcam_str: str) -> Optional[int]:
    """Binary flag: 1 if has webcam, 0 if not."""
    if pd.isna(webcam_str):
        return np.nan

    webcam_str = str(webcam_str).lower()
    if any(word in webcam_str for word in ['ninguna', 'ninguno', 'no']):
        return 0
    if any(word in webcam_str for word in ['integrada', 'megapixel', 'mp', 'hd', 'fhd']):
        return 1
    return np.nan


def extraer_version_bluetooth(bluetooth_str: str) -> Optional[float]:
    """Extract Bluetooth version."""
    if pd.isna(bluetooth_str):
        return np.nan
    match = re.search(r'bluetooth\s*(\d+\.\d+)', str(bluetooth_str), re.IGNORECASE)
    return float(match.group(1)) if match else np.nan


# =============================================================================
# MAIN FEATURE ENGINEERING FUNCTION
# =============================================================================

def construir_features(df_computers: pd.DataFrame,
                       df_cpu: pd.DataFrame,
                       df_gpu: pd.DataFrame) -> pd.DataFrame:
    """Build all engineered features from raw computer data."""
    df = df_computers.copy()

    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)

    # 1. Target variable: Price
    print("\n1. Extracting _precio_num...")
    df['_precio_num'] = df['Precio_Rango'].apply(extraer_precio_medio)
    print(f"   Valid prices: {df['_precio_num'].notna().sum():,}")

    # 2. Brand
    print("\n2. Extracting _brand...")
    df['_brand'] = df['Título'].apply(extraer_brand)
    print(f"   Brands: {df['_brand'].nunique()} unique")

    # 3. Series
    print("\n3. Extracting _serie...")
    df['_serie'] = df.apply(
        lambda row: extraer_serie(row.get('Serie'), row.get('Título'), row.get('_brand')),
        axis=1
    )
    print(f"   Series: {df['_serie'].notna().sum():,}")

    # 4. RAM
    print("\n4. Extracting _ram_gb...")
    df['_ram_gb'] = df['RAM_Memoria RAM'].apply(extraer_ram_gb)
    print(f"   RAM: {df['_ram_gb'].notna().sum():,}")

    # 5. SSD
    print("\n5. Extracting _ssd_gb...")
    df['_ssd_gb'] = df['Disco duro_Capacidad de memoria SSD'].apply(extraer_ssd_gb)
    print(f"   SSD: {df['_ssd_gb'].notna().sum():,}")

    # 6. Screen size
    print("\n6. Extracting _tamano_pantalla_pulgadas...")
    df['_tamano_pantalla_pulgadas'] = df.apply(
        lambda row: extraer_pantalla_pulgadas(
            row.get('Pantalla_Tamaño de la pantalla'),
            row.get('Pantalla_Diagonal de la pantalla'),
            row.get('Título')
        ),
        axis=1
    )
    print(f"   Screen size: {df['_tamano_pantalla_pulgadas'].notna().sum():,}")

    # 7. CPU cores
    print("\n7. Extracting _cpu_cores...")
    df['_cpu_cores'] = df['Procesador_Número de núcleos del procesador'].apply(extraer_cpu_cores)
    print(f"   CPU cores: {df['_cpu_cores'].notna().sum():,}")

    # 8. CPU benchmarks (with parsing and matching)
    print("\n8. Matching CPU benchmarks...")
    df = match_cpu_benchmarks(df, df_cpu, "Procesador_Procesador")
    df['_cpu_mark'] = df['cpu_bench_mark']
    df['_cpu_price_usd'] = df['cpu_bench_price_usd']
    print(f"   CPU price: {df['_cpu_price_usd'].notna().sum():,}")

    # 9. GPU benchmarks (with parsing and matching)
    print("\n9. Matching GPU benchmarks...")
    df = match_gpu_benchmarks(df, df_gpu, "Gráfica_Tarjeta gráfica")
    df['_gpu_mark'] = df['gpu_bench_mark']
    df['_gpu_price_usd'] = df['gpu_bench_price_usd']
    print(f"   GPU price: {df['_gpu_price_usd'].notna().sum():,}")

    # 10. GPU memory
    print("\n10. Extracting _gpu_memory_gb...")
    df['_gpu_memory_gb'] = df['Gráfica_Memoria gráfica'].apply(extraer_gpu_memory_gb)
    print(f"   GPU memory: {df['_gpu_memory_gb'].notna().sum():,}")

    # 11. Weight
    print("\n11. Extracting _peso_kg...")
    df['_peso_kg'] = df['Medidas y peso_Peso'].apply(extraer_peso_kg)
    print(f"   Weight: {df['_peso_kg'].notna().sum():,}")

    # 12. Resolution
    print("\n12. Extracting _resolucion_pixeles...")
    df['_resolucion_pixeles'] = df['Pantalla_Resolución de pantalla'].apply(extraer_resolucion_pixeles)
    print(f"   Resolution: {df['_resolucion_pixeles'].notna().sum():,}")

    # 13. Refresh rate
    print("\n13. Extracting _tasa_refresco_hz...")
    df['_tasa_refresco_hz'] = df['Pantalla_Tasa de actualización de imagen'].apply(extraer_tasa_refresco)
    print(f"   Refresh rate: {df['_tasa_refresco_hz'].notna().sum():,}")

    # 14. WiFi
    print("\n14. Creating _tiene_wifi...")
    df['_tiene_wifi'] = df['Comunicaciones_Conectividad'].apply(tiene_wifi)
    print(f"   WiFi: {df['_tiene_wifi'].notna().sum():,}")

    # 15. Bluetooth
    print("\n15. Creating _tiene_bluetooth...")
    df['_tiene_bluetooth'] = df['Comunicaciones_Conectividad'].apply(tiene_bluetooth)
    print(f"   Bluetooth: {df['_tiene_bluetooth'].notna().sum():,}")

    # 16. Webcam
    print("\n16. Creating _tiene_webcam...")
    df['_tiene_webcam'] = df['Cámara_Webcam'].apply(tiene_webcam)
    print(f"   Webcam: {df['_tiene_webcam'].notna().sum():,}")

    # 17. Bluetooth version
    print("\n17. Extracting _version_bluetooth...")
    df['_version_bluetooth'] = df['Comunicaciones_Versión Bluetooth'].apply(extraer_version_bluetooth)
    print(f"   BT version: {df['_version_bluetooth'].notna().sum():,}")

    # Summary
    engineered = [c for c in df.columns if c.startswith('_')]
    print("\n" + "=" * 60)
    print(f"COMPLETE: {len(engineered)} engineered features")
    print("=" * 60)

    return df


# =============================================================================
# DATA PREPARATION FOR MODELING
# =============================================================================

def prepare_modeling_data(df: pd.DataFrame,
                          target_col: str = '_precio_num',
                          max_missing_pct: float = 0.60) -> pd.DataFrame:
    """
    Prepare data for modeling by:
    1. Removing rows without valid target
    2. Removing features with >max_missing_pct missing values
    3. Selecting only engineered features (starting with _) plus useful metadata

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with engineered features
    target_col : str
        Name of the target column
    max_missing_pct : float
        Maximum allowed missing value percentage (0.60 = 60%)

    Returns
    -------
    pd.DataFrame
        Cleaned and prepared DataFrame for modeling
    """
    print("\n" + "=" * 60)
    print("PREPARING DATA FOR MODELING")
    print("=" * 60)

    df_model = df.copy()

    # 1. Remove rows without valid target
    original_rows = len(df_model)
    df_model = df_model[df_model[target_col].notna()]
    rows_dropped = original_rows - len(df_model)
    print(f"\n1. Dropped {rows_dropped:,} rows with missing target")
    print(f"   Remaining rows: {len(df_model):,}")

    # 2. Identify engineered features
    engineered_cols = [c for c in df_model.columns if c.startswith('_')]
    print(f"\n2. Found {len(engineered_cols)} engineered features")

    # 3. Analyze and remove features with too many missing values
    print(f"\n3. Filtering features with >{max_missing_pct*100:.0f}% missing values...")

    missing_analysis = []
    for col in engineered_cols:
        missing_pct = df_model[col].isna().mean()
        missing_analysis.append({
            'column': col,
            'missing_pct': missing_pct,
            'keep': missing_pct <= max_missing_pct
        })

    missing_df = pd.DataFrame(missing_analysis).sort_values('missing_pct', ascending=False)

    # Show features being removed
    removed_features = missing_df[~missing_df['keep']]['column'].tolist()
    kept_features = missing_df[missing_df['keep']]['column'].tolist()

    if removed_features:
        print(f"\n   Removing {len(removed_features)} features with >{max_missing_pct*100:.0f}% missing:")
        for col in removed_features:
            pct = missing_df[missing_df['column'] == col]['missing_pct'].iloc[0]
            print(f"      - {col}: {pct*100:.1f}% missing")

    print(f"\n   Keeping {len(kept_features)} features with <={max_missing_pct*100:.0f}% missing")

    # 4. Select columns for modeling
    # Include kept engineered features + useful categorical columns for analysis
    metadata_cols = ['_brand', '_serie']
    benchmark_cols = ['cpu_match_strategy', 'gpu_match_strategy']

    final_cols = kept_features + [c for c in metadata_cols if c in kept_features]
    final_cols = list(dict.fromkeys(final_cols))  # Remove duplicates, preserve order

    # Also keep match strategy columns for analysis
    for col in benchmark_cols:
        if col in df_model.columns:
            final_cols.append(col)

    df_final = df_model[final_cols].copy()

    print(f"\n4. Final dataset shape: {df_final.shape}")
    print(f"   - Rows: {len(df_final):,}")
    print(f"   - Features: {len(df_final.columns)}")

    # 5. Show summary of remaining missing values
    remaining_missing = df_final[kept_features].isna().mean() * 100
    remaining_missing = remaining_missing[remaining_missing > 0].sort_values(ascending=False)

    if len(remaining_missing) > 0:
        print(f"\n5. Remaining missing values (to be imputed by model):")
        for col, pct in remaining_missing.head(10).items():
            print(f"      {col}: {pct:.1f}%")
        if len(remaining_missing) > 10:
            print(f"      ... and {len(remaining_missing) - 10} more")

    print("\n" + "=" * 60)
    print("DATA PREPARATION COMPLETE")
    print("=" * 60)

    return df_final


def get_feature_columns(df: pd.DataFrame, target_col: str = '_precio_num') -> Tuple[List[str], List[str], List[str]]:
    """
    Get feature columns for modeling, split by numeric and categorical.

    Returns
    -------
    Tuple[List[str], List[str], List[str]]
        - all_features: All feature column names
        - numeric_features: Numeric feature names
        - categorical_features: Categorical feature names
    """
    # Exclude target and metadata columns
    exclude_cols = [target_col, 'cpu_match_strategy', 'gpu_match_strategy']

    all_features = [c for c in df.columns if c.startswith('_') and c not in exclude_cols]

    numeric_features = []
    categorical_features = []

    for col in all_features:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_features.append(col)
        else:
            categorical_features.append(col)

    return all_features, numeric_features, categorical_features
