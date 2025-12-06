# Quick Start Guide

## Feature Engineering Pipeline

The improved feature engineering pipeline now includes **intelligent CPU/GPU matching** and **18 comprehensive features** based on correlation analysis.

### Running the Pipeline

```bash
# Option 1: Run the notebook
jupyter notebook notebooks/02_feature_engineering.ipynb

# Option 2: Run from Python
python3 << 'EOF'
import sys
sys.path.insert(0, 'src')
from features import cargar_datos, construir_features

# Load data
df_computers, df_cpu, df_gpu = cargar_datos(
    'data/db_computers_2025_raw.csv',
    'data/db_cpu_raw.csv',
    'data/db_gpu_raw.csv'
)

# Build features
df = construir_features(df_computers, df_cpu, df_gpu)

# Save
df.to_parquet('data/db_computers_processed.parquet', index=False)
print(f"\n✓ Saved {len(df):,} rows with {len([c for c in df.columns if c.startswith('_')])} features")
EOF
```

### What You Get

**18 Engineered Features:**

1. `_precio_num` - Target variable (price midpoint)
2. `_brand` - Brand from title
3. `_serie` - Product series
4. `_ram_gb` - RAM in GB ⭐ Strong predictor
5. `_ssd_gb` - SSD capacity
6. `_tamano_pantalla_pulgadas` - Screen size
7. `_cpu_cores` - Number of CPU cores ⭐ Strong predictor
8. `_cpu_mark` - CPU benchmark score (100% match rate!)
9. `_gpu_mark` - GPU benchmark score
10. `_gpu_memory_gb` - GPU VRAM ⭐ Strong predictor
11. `_num_ofertas` - Number of marketplace offers
12. `_peso_kg` - Weight in kg
13. `_resolucion_pixeles` - Total pixels
14. `_tasa_refresco_hz` - Refresh rate ⭐ Moderate predictor
15. `_tiene_wifi` - Has WiFi (binary)
16. `_tiene_bluetooth` - Has Bluetooth (binary)
17. `_tiene_webcam` - Has webcam (binary)
18. `_version_bluetooth` - Bluetooth version

### Key Improvements

✅ **100% CPU Match Rate**
- Intelligent suffix handling (H, HX, U, P, K, etc.)
- Apple processor matching with core count
- Progressive fallback strategies

✅ **Smart GPU Matching**
- Filters integrated graphics correctly
- Matches discrete GPUs (RTX, GTX, Radeon)
- Handles laptop/desktop variants

✅ **Based on Correlation Analysis**
- Strong predictors (r > 0.5): RAM, GPU memory, CPU cores
- Moderate predictors (0.3-0.5): Refresh rate, SSD, benchmarks

### File Structure

```
computer-price-predictor/
├── data/
│   ├── db_computers_2025_raw.csv      # Raw data
│   ├── db_cpu_raw.csv                 # CPU benchmarks
│   ├── db_gpu_raw.csv                 # GPU benchmarks
│   └── db_computers_processed.parquet # Output (created)
├── src/
│   └── features.py                    # Feature engineering
├── notebooks/
│   ├── 01_eda.ipynb                   # Exploratory analysis
│   └── 02_feature_engineering.ipynb   # Feature engineering ⭐
└── FEATURE_ENGINEERING_IMPROVEMENTS.md # Full documentation
```

### Next Steps

1. ✅ Feature engineering complete
2. → Run notebook 02 to process full dataset
3. → Train ML models with engineered features
4. → Evaluate and deploy

### Troubleshooting

**Issue:** Low GPU match rate (~40%)
- **Expected!** Most laptops have integrated graphics (Intel Arc, UHD, AMD Radeon Graphics)
- These are correctly filtered out (not in benchmark DB)
- Discrete GPUs (RTX, GTX) match successfully

**Issue:** Missing fuzzywuzzy
```bash
pip install fuzzywuzzy python-Levenshtein
```

### Documentation

- [FEATURE_ENGINEERING_IMPROVEMENTS.md](FEATURE_ENGINEERING_IMPROVEMENTS.md) - Detailed improvements
- [notebooks/01_eda.ipynb](notebooks/01_eda.ipynb) - Correlation analysis
- [notebooks/02_feature_engineering.ipynb](notebooks/02_feature_engineering.ipynb) - Feature engineering

---

**Status: READY FOR MODEL TRAINING** ✅
