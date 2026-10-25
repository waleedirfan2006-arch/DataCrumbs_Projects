import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'data.csv')
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'antigravity_pipeline.joblib')

# Load dataset
df = pd.read_csv(DATA_PATH)

# Identify target and features
TARGET = 'price'
FEATURES = [col for col in df.columns if col != TARGET]

# Basic type inference
numeric_features = []
categorical_features = []
for col in FEATURES:
    if pd.api.types.is_numeric_dtype(df[col]):
        numeric_features.append(col)
    else:
        categorical_features.append(col)

# Preprocess pipeline
numeric_transformer = Pipeline(steps=[
    ('imputer',  __import__('sklearn.impute').impute.SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer',  __import__('sklearn.impute').impute.SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocess = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Full pipeline with Linear Regression
model = Pipeline(steps=[
    ('preprocess', preprocess),
    ('regressor', LinearRegression())
])

# Train‑test split (80/20) with a fixed random state for reproducibility
X = df[FEATURES]
y = df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Fit model
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.4f}")

# Ensure model directory exists
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
# Persist pipeline
joblib.dump(model, MODEL_PATH)
print(f"Model pipeline saved to {MODEL_PATH}")
