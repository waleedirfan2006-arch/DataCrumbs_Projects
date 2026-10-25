# DATASET REPORT – HousePredict

**File**: `data/data.csv` (CSV, comma‑separated)  
**Rows**: **4 600** (excluding header)  
**Columns**: **18**

| Column | Detected type | Role |
|--------|---------------|------|
| `date` | datetime (string) | **Categorical** (timestamp – can be turned into numeric if needed) |
| `price` | float | **Target** |
| `bedrooms` | float (int values) | Numerical |
| `bathrooms` | float | Numerical |
| `sqft_living` | int | Numerical |
| `sqft_lot` | int | Numerical |
| `floors` | float | Numerical |
| `waterfront` | int (0/1) | **Categorical** (binary) |
| `view` | int (0‑4) | **Categorical** (ordinal) |
| `condition` | int (1‑5) | Numerical |
| `sqft_above` | int | Numerical |
| `sqft_basement` | int | **Categorical** (treated as categorical because of mixed types in source) |
| `yr_built` | int | Numerical |
| `yr_renovated` | int (0 = none) | Numerical |
| `street` | string | **Categorical** |
| `city` | string | **Categorical** |
| `statezip` | string (e.g., `WA 98133`) | **Categorical** |
| `country` | string (`USA`) | **Categorical** |

### Missing values
*No missing entries* – every column contains a value for all 4 600 rows.

### Duplicates
*Zero duplicate rows* – all records are unique.

### Numeric vs. Categorical
- **Numeric columns** (used directly or after scaling): `price, bedrooms, bathrooms, sqft_living, sqft_lot, floors, condition, sqft_above, yr_built, yr_renovated`  
- **Categorical columns** (to be one‑hot encoded): `date, waterfront, view, sqft_basement, street, city, statezip, country`

### Potential data leakage
- `date` records the exact sale date. When predicting future sales, keep only derived components (year, month) or drop the column to avoid leakage.

### Suitability for Multiple Linear Regression
The dataset is clean, well‑balanced, and sized appropriately for a linear model after the preprocessing steps listed below.

### Recommended preprocessing pipeline (Python / scikit‑learn)
```python
# 1️⃣ Parse date → numeric timestamp (or drop if leakage is a concern)
# 2️⃣ One‑hot encode all categorical columns listed above
# 3️⃣ Standard‑scale numeric columns
# 4️⃣ Train‑test split (e.g., 80/20, random_state=42)
# 5️⃣ Fit LinearRegression inside a Pipeline with ColumnTransformer
# 6️⃣ Persist the pipeline with joblib for inference
```

### Next steps
- Run `model/train_model.py` to create `model/antigravity_pipeline.joblib`.
- Launch the Node.js web UI (`npm start`) to predict house prices interactively.
