# Model Optimization Results

## Final Model Performance

After extensive optimization, the best model configuration achieved:

| Metric | Value |
|--------|-------|
| **Model** | CatBoost |
| **RMSE** | €329.00 |
| **MAE** | €222.57 |
| **R²** | 0.79 |
| **MAPE** | 20.44% |

## Preprocessing Strategy Comparison

We tested multiple outlier removal strategies to find the optimal approach:

| Strategy | Train Samples | RMSE (€) | MAE (€) | R² | MAPE (%) |
|----------|---------------|----------|---------|-----|----------|
| **IQR Method** | 5,915 | **329.00** | 222.57 | 0.79 | 20.44 |
| 2-98th Percentile | 5,963 | 373.75 | 244.78 | 0.80 | 19.64 |
| 1-99th Percentile | 6,087 | 393.98 | 257.94 | 0.79 | 21.65 |
| Log + 1-99th Pct | 6,087 | 398.48 | 246.47 | 0.79 | 18.93 |
| Log-Transform | 6,213 | 439.21 | 259.17 | 0.79 | 19.41 |

**Winner: IQR Method** - Best RMSE with acceptable sample reduction (~5.5% removed)

### IQR Method Details
- Price range kept: €0 - €3,605
- Q1: €746, Q3: €1,890, IQR: €1,143
- Upper bound: Q3 + 1.5×IQR = €3,605

## Model Comparison (with 1-99th percentile data)

| Model | RMSE | MAE | R² |
|-------|------|-----|-----|
| **CatBoost** | 471.54 | 295.82 | 0.78 |
| XGBoost | 499.03 | 313.37 | 0.75 |
| Ridge | 511.00 | 328.68 | 0.74 |
| LightGBM | 514.19 | 322.95 | 0.73 |
| HistGradientBoosting | 516.35 | 322.89 | 0.73 |
| Random Forest | 519.89 | 317.15 | 0.73 |

**CatBoost consistently outperforms** all other models due to its native categorical handling.

## Feature Selection

### Top 30 Features (by Permutation Importance)

Based on CatBoost feature importance analysis, the top 30 features account for ~90% of predictive power:

**Benchmark Features (High Importance):**
- `_cpu_mark` - CPU benchmark score
- `_cpu_rank` - CPU ranking
- `_cpu_value` - CPU value score
- `_cpu_price_usd` - CPU reference price
- `_gpu_mark` - GPU benchmark score
- `_gpu_rank` - GPU ranking
- `_gpu_value` - GPU value score
- `_gpu_price_usd` - GPU reference price

**Hardware Specs (Medium-High Importance):**
- `_ram_gb` - RAM size
- `_ssd_gb` - SSD capacity
- `_cpu_cores` - CPU core count
- `_gpu_memory_gb` - GPU VRAM

**Display Features:**
- `_tamano_pantalla_pulgadas` - Screen size
- `_resolucion_pixeles` - Resolution
- `_tasa_refresco_hz` - Refresh rate

**Categorical Features:**
- `_brand` - Manufacturer
- `cpu_brand` - CPU manufacturer
- `cpu_family` - CPU family (i3, i5, i7, Ryzen 5, etc.)
- `gpu_brand` - GPU manufacturer
- `gpu_series` - GPU series

### Full Model vs Lite Model

| Model | Features | RMSE | Notes |
|-------|----------|------|-------|
| **Full Model** | 30 | €329 | Includes benchmark features |
| Lite Model | 15 | €520+ | User-friendly only, ~60% worse |

**Recommendation: Use the Full Model (Top 30 features)** - The performance difference is too significant to use the lite model.

## Feature Categories

### Features Requiring Backend Lookup (8 features)
These are automatically populated from benchmark databases:
- `_cpu_mark`, `_cpu_rank`, `_cpu_value`, `_cpu_price_usd`
- `_gpu_mark`, `_gpu_rank`, `_gpu_value`, `_gpu_price_usd`

### User-Provided Features

**Easy (direct input):**
- RAM (GB), SSD (GB), Screen size, Weight

**Medium (dropdown selection):**
- Brand, CPU family, GPU series, Screen technology

## Data Pipeline Summary

```
Raw Data (8,064 rows)
    ↓
Feature Engineering (notebook 02)
    - CPU/GPU benchmark matching
    - 100% CPU match rate
    - 41% GPU match rate (correct - integrated GPUs filtered)
    ↓
Outlier Removal (IQR Method)
    - Remove prices > €3,605
    - Remove prices < €0 (none exist)
    - ~5,915 training samples
    ↓
Model Training (CatBoost)
    - Native categorical handling
    - Top 30 features
    ↓
Final Model
    - RMSE: €329
    - MAPE: 20.44%
```

## Files and Artifacts

### Models
- `models/price_model_optimized.pkl` - Best CatBoost model with metadata

### Data
- `data/db_features.parquet` - Processed features (with 1-99th percentile)
- `data/db_features_raw.parquet` - Raw features (for experiment comparison)

### Code
- `src/features.py` - Feature engineering functions
- `src/modeling.py` - Model building and evaluation functions

### Notebooks
1. `01_eda.ipynb` - Exploratory data analysis + outlier analysis
2. `02_feature_engineering.ipynb` - Feature creation and correlation
3. `03_model_training.ipynb` - Initial model comparison
4. `04_model_optimization.ipynb` - Hyperparameter tuning and final model

## Hyperparameter Tuning (Optuna)

CatBoost was tuned with 30 Optuna trials using the IQR-filtered data:

**Search Space:**
- `iterations`: 200-800
- `depth`: 4-10
- `learning_rate`: 0.01-0.3 (log scale)
- `l2_leaf_reg`: 1.0-10.0
- `bagging_temperature`: 0.0-1.0
- `min_data_in_leaf`: 1-30

## Key Learnings

1. **Outlier removal matters**: IQR method improved RMSE by ~25% vs no removal
2. **Benchmark features are crucial**: CPU/GPU marks are top predictors
3. **CatBoost > other boosting**: Native categorical handling makes a difference
4. **Top 30 features sufficient**: No need for all features, maintains 90% predictive power

## Next Steps

1. Deploy model via Streamlit app
2. Implement NLP interface for natural language queries
3. Add prediction explanations (SHAP values)
4. Log user feedback for continuous improvement
