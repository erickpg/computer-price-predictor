#!/usr/bin/env python3
"""
RUN THIS NOW to verify the matching works!
This will process the FULL dataset and show you the results.
"""

import sys
sys.path.insert(0, 'src')

# Force fresh import
for mod in list(sys.modules.keys()):
    if 'features' in mod:
        del sys.modules[mod]

from features import cargar_datos, construir_features

print("\n" + "=" * 80)
print("VERIFICATION TEST - FULL DATASET")
print("=" * 80)

# Load data
print("\nLoading data...")
df_c, df_cpu, df_gpu = cargar_datos(
    'data/db_computers_2025_raw.csv',
    'data/db_cpu_raw.csv',
    'data/db_gpu_raw.csv'
)

# Build features
print("\nBuilding features on FULL dataset...")
print("(This takes ~30-60 seconds)\n")

df = construir_features(df_c, df_cpu, df_gpu)

# Show results
print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)

cpu_with_names = df['Procesador_Procesador'].notna().sum()
cpu_matched = df['_cpu_mark'].notna().sum()
cpu_rate = (cpu_matched / cpu_with_names * 100) if cpu_with_names > 0 else 0

print(f"\n🎯 CPU MATCHING:")
print(f"   Processors with names: {cpu_with_names:,}")
print(f"   Successfully matched:  {cpu_matched:,}")
print(f"   Match rate:            {cpu_rate:.1f}%")

if cpu_rate > 90:
    print("   ✅ EXCELLENT! Matching is working!")
else:
    print("   ❌ Something is wrong")

# Show examples
print(f"\n📊 SAMPLE MATCHES:")
print("-" * 80)
sample = df[df['_cpu_mark'].notna()][['Procesador_Procesador', '_cpu_cores', '_cpu_mark']].head(15)
for idx, row in sample.iterrows():
    print(f"  {row['Procesador_Procesador']:40s} → {row['_cpu_mark']:,.0f}")

print("\n" + "=" * 80)
if cpu_rate > 90:
    print("✅ THE CODE WORKS! Your notebook just needs to reload the module.")
    print("   Follow the instructions to fix your notebook.")
else:
    print("❌ There's an issue. Contact support.")
print("=" * 80 + "\n")
