# Frontend Integration Notes

## For: Person 2 (Streamlit App + NLP Interface + Integration)

This document contains important considerations based on the ML pipeline analysis. These are observations from the model optimization phase that will impact your frontend and integration work.

---

## Model Selection

### Use the Full Model (Top 30 Features), Not the Lite Model

| Model | RMSE | Difference |
|-------|------|------------|
| Full Model (30 features) | €329 | baseline |
| Lite Model (15 features) | €520+ | **+58% worse** |

The lite model was designed for easier user input but the performance degradation is too significant. The full model with benchmark features is required for acceptable predictions.

**Implication**: You will need to implement backend lookups for CPU/GPU benchmarks.

---

## Feature Input Requirements

### Features That Need Backend Lookup (Critical)

These 8 features are the most important predictors but users cannot provide them directly:

| Feature | Description | Source |
|---------|-------------|--------|
| `_cpu_mark` | CPU benchmark score | Lookup from CPU name |
| `_cpu_rank` | CPU ranking | Lookup from CPU name |
| `_cpu_value` | CPU value score | Lookup from CPU name |
| `_cpu_price_usd` | CPU reference price | Lookup from CPU name |
| `_gpu_mark` | GPU benchmark score | Lookup from GPU name |
| `_gpu_rank` | GPU ranking | Lookup from GPU name |
| `_gpu_value` | GPU value score | Lookup from GPU name |
| `_gpu_price_usd` | GPU reference price | Lookup from GPU name |

**How it works**: User selects "Intel Core i7-13700H" → backend looks up benchmark data → returns all 4 CPU metrics automatically.

The benchmark lookup functions already exist in `src/features.py`:
- `match_cpu_benchmarks()`
- `match_gpu_benchmarks()`

### User Input Features

**Numeric (sliders or number inputs):**
- `_ram_gb`: 4-128 GB (common: 8, 16, 32)
- `_ssd_gb`: 128-4000 GB (common: 256, 512, 1000)
- `_tamano_pantalla_pulgadas`: 11-18 inches
- `_peso_kg`: 0.5-4.0 kg
- `_resolucion_pixeles`: calculated from resolution dropdown
- `_tasa_refresco_hz`: 60, 90, 120, 144, 165, 240 Hz
- `_gpu_memory_gb`: 0-16 GB (0 for integrated)
- `_cpu_cores`: 2-24 cores

**Categorical (dropdowns):**
- `_brand`: Apple, ASUS, Acer, Dell, HP, Lenovo, MSI, etc.
- `cpu_brand`: Intel, AMD, Apple
- `cpu_family`: Core i3, Core i5, Core i7, Core i9, Ryzen 5, Ryzen 7, M1, M2, M3, etc.
- `gpu_brand`: NVIDIA, AMD, Intel, Apple
- `gpu_series`: GeForce RTX 30xx, RTX 40xx, Radeon RX, etc.
- `gpu_is_integrated`: True/False

---

## Data Considerations

### Price Range

The model was trained on prices between **€0 - €3,605** (IQR method).

- Predictions outside this range may be unreliable
- Consider showing a warning if predicted price approaches bounds
- Very high-end workstations (>€4,000) were excluded from training

### Missing Values

The model handles missing values via imputation, but:
- Missing benchmark features significantly hurt predictions
- If user selects a CPU/GPU not in the benchmark database, predictions will be less accurate
- Consider showing confidence level based on data completeness

### Categorical Encoding

CatBoost handles categoricals natively. When loading the model:
- Categorical columns must be passed as strings
- The model expects specific column names (see metadata in pkl file)

---

## Model Loading

### Loading the Model

```python
import joblib

# Load model and metadata
saved = joblib.load('models/price_model_optimized.pkl')
model = saved['model']
metadata = saved['metadata']

# Get expected features
feature_cols = metadata['feature_cols']
numeric_cols = metadata['numeric_cols']
categorical_cols = metadata['categorical_cols']
```

### Making Predictions

```python
import pandas as pd

# Create input DataFrame with correct column order
input_data = pd.DataFrame([{
    '_ram_gb': 16,
    '_ssd_gb': 512,
    '_cpu_mark': 26452,  # from benchmark lookup
    # ... all 30 features
}])

# Ensure categorical columns are strings
for col in categorical_cols:
    if col in input_data.columns:
        input_data[col] = input_data[col].astype(str)

# Predict
predicted_price = model.predict(input_data[feature_cols])[0]
```

---

## Benchmark Database Integration

### CPU Benchmark Data

File: `data/cpu_benchmark.csv`

Columns:
- `cpuName`: Full CPU name (use for matching)
- `cpuMark`: Benchmark score → `_cpu_mark`
- `Rank`: Ranking → `_cpu_rank`
- `cpuValue`: Value score → `_cpu_value`
- `price`: USD price → `_cpu_price_usd`

### GPU Benchmark Data

File: `data/gpu_benchmark.csv`

Similar structure to CPU data.

### Matching Logic

The matching functions in `src/features.py` handle:
- Fuzzy matching for name variations
- Laptop/Mobile suffix handling
- Integrated graphics filtering (skip matching)

Consider pre-building a lookup dictionary for faster UI responses.

---

## Error Ranges and Uncertainty

### Expected Prediction Error

| Metric | Value | Interpretation |
|--------|-------|----------------|
| MAE | €222 | Average error magnitude |
| MAPE | 20% | Predictions typically within ±20% |

**Suggestion**: Show predictions as a range rather than a single number:
- "Estimated price: €1,200 (€960 - €1,440)"
- This is more honest and user-friendly

### Confidence Indicators

Consider showing lower confidence when:
- CPU/GPU not found in benchmark database
- Many features are missing
- Price prediction is near training data boundaries

---

## NLP Interface Considerations

### Entity Extraction

Users might say things like:
- "MacBook Pro with M3 chip, 16GB RAM, 512GB SSD"
- "Gaming laptop with RTX 4070 and i7 processor"
- "Budget laptop around 15 inches with 8GB RAM"

Key entities to extract:
1. Brand names
2. CPU/GPU model names
3. RAM size
4. Storage size
5. Screen size
6. Use case (gaming, work, budget)

### Handling Ambiguity

- "i7 processor" → Need to ask which generation (11th, 12th, 13th, 14th?)
- "RTX graphics" → Need to ask which model (3050, 4060, 4070?)
- "16GB RAM" is clear
- "Big screen" → Ambiguous, default to 15.6" or ask

### Fallback Strategy

If NLP cannot extract required features:
1. Show form with pre-filled known values
2. Highlight missing required fields
3. Allow user to complete manually

---

## Feedback Logging

### Minimum Fields to Log

```csv
timestamp,predicted_price,actual_features,user_satisfaction,user_comment
2024-01-15 10:30:00,1250.00,"{...json...}",4,""
```

### Satisfaction Scale

Suggest using 1-5 scale:
1. Very inaccurate
2. Somewhat inaccurate
3. Neutral
4. Somewhat accurate
5. Very accurate

### Privacy Note

Do not log any personally identifiable information.

---

## Performance Expectations

### Prediction Speed

- CatBoost predictions are fast (~1-10ms per prediction)
- Benchmark lookup may be slower if searching large CSV
- Consider caching common CPU/GPU lookups

### Memory Usage

- Model size: ~5-20 MB depending on tuning
- Benchmark CSVs: ~1-2 MB each
- Total app memory: Should be under 500 MB

---

## Common Edge Cases

1. **Integrated Graphics**: When `gpu_is_integrated=True`, GPU benchmark features should be None/NaN
2. **Apple Devices**: Apple M-series chips are both CPU and GPU - handle specially
3. **Unknown CPUs/GPUs**: New models not in benchmark DB - use family averages if available
4. **Extreme Values**: RAM > 64GB, SSD > 2TB - rare but valid

---

## Files You'll Need

```
models/
├── price_model_optimized.pkl   # Main model

data/
├── cpu_benchmark.csv           # CPU lookup data
├── gpu_benchmark.csv           # GPU lookup data

src/
├── features.py                 # Feature engineering + benchmark matching
├── modeling.py                 # Model utilities
```

---

## Testing Checklist

- [ ] Model loads correctly
- [ ] All 30 features can be populated
- [ ] CPU benchmark lookup works
- [ ] GPU benchmark lookup works
- [ ] Predictions are in reasonable range (€200-€4000)
- [ ] Missing values handled gracefully
- [ ] Categorical encoding is correct
- [ ] Feedback logging works
- [ ] App runs with `streamlit run app.py`
