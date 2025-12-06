#!/usr/bin/env python3
"""
Quick test script to verify feature engineering works correctly.
Run this to confirm the improved matching is working.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

from features import cargar_datos, construir_features
import pandas as pd

def main():
    print("=" * 80)
    print("FEATURE ENGINEERING TEST")
    print("=" * 80)

    # Load data
    print("\n1. Loading data...")
    df_computers, df_cpu, df_gpu = cargar_datos(
        'data/db_computers_2025_raw.csv',
        'data/db_cpu_raw.csv',
        'data/db_gpu_raw.csv'
    )

    # Build features on full dataset
    print("\n2. Building features on FULL dataset...")
    print("   (This may take a few minutes)\n")
    df = construir_features(df_computers, df_cpu, df_gpu)

    # Verify features were created
    print("\n" + "=" * 80)
    print("VERIFICATION")
    print("=" * 80)

    eng_features = [c for c in df.columns if c.startswith('_')]
    print(f"\n✓ Total engineered features: {len(eng_features)}")

    # Check CPU matching
    cpu_with_names = df['Procesador_Procesador'].notna().sum()
    cpu_matched = df['_cpu_mark'].notna().sum()
    cpu_match_rate = (cpu_matched / cpu_with_names * 100) if cpu_with_names > 0 else 0

    print(f"\n✓ CPU Matching:")
    print(f"  Processors with names: {cpu_with_names:,}")
    print(f"  Successfully matched:  {cpu_matched:,}")
    print(f"  Match rate:            {cpu_match_rate:.1f}%")

    # Check GPU matching
    gpu_with_names = df['Gráfica_Tarjeta gráfica'].notna().sum()
    gpu_matched = df['_gpu_mark'].notna().sum()
    gpu_match_rate = (gpu_matched / gpu_with_names * 100) if gpu_with_names > 0 else 0

    print(f"\n✓ GPU Matching:")
    print(f"  GPUs with names:       {gpu_with_names:,}")
    print(f"  Successfully matched:  {gpu_matched:,}")
    print(f"  Match rate:            {gpu_match_rate:.1f}%")
    print(f"  (Note: Low rate is expected - most are integrated graphics)")

    # Show sample matches
    print("\n" + "-" * 80)
    print("Sample CPU Matches:")
    print("-" * 80)
    sample = df[df['_cpu_mark'].notna()][['Procesador_Procesador', '_cpu_cores', '_cpu_mark']].head(10)
    print(sample.to_string(index=False))

    # Save processed data
    print("\n3. Saving processed dataset...")
    output_path = Path('data/db_computers_processed.parquet')

    # Only keep rows with valid target
    df_clean = df[df['_precio_num'].notna()].copy()
    df_clean.to_parquet(output_path, index=False)

    print(f"\n✓ Saved to: {output_path}")
    print(f"  Rows: {len(df_clean):,}")
    print(f"  Features: {len([c for c in df_clean.columns if c.startswith('_')])}")
    print(f"  Size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

    print("\n" + "=" * 80)
    print("✅ FEATURE ENGINEERING COMPLETE!")
    print("=" * 80)
    print("\nNext step: Run the notebook or train your model!")

if __name__ == '__main__':
    main()
