"""
test_pipeline.py
================
Unit and integration tests for the HousePredict ML pipeline:
1. Model loading verification
2. Valid prediction generation with finite numeric values
3. Graceful handling of edge-cases and unknown categories
4. Prediction reasonableness test
"""

import os
import joblib
import numpy as np
import pandas as pd
import pytest

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'house_price_pipeline.joblib')

@pytest.fixture
def model():
    """Fixture ensuring the model pipeline loads successfully."""
    assert os.path.exists(MODEL_PATH), f"Model file not found at {MODEL_PATH}"
    pipeline = joblib.load(MODEL_PATH)
    assert pipeline is not None
    return pipeline

def test_model_loading(model):
    """Verifies that the serialized pipeline can be loaded and has expected stages."""
    assert hasattr(model, 'predict'), "Loaded pipeline must implement a predict() method"
    assert 'preprocessor' in model.named_steps
    assert 'regressor' in model.named_steps

def test_valid_prediction(model):
    """Verifies prediction on typical input features returns a finite numeric value."""
    sample = pd.DataFrame([{
        'bedrooms': 3.0,
        'bathrooms': 2.0,
        'sqft_living': 2000,
        'sqft_lot': 6000,
        'floors': 1.5,
        'waterfront': 0,
        'view': 0,
        'condition': 3,
        'sqft_above': 1500,
        'sqft_basement': 500,
        'yr_built': 1980,
        'yr_renovated': 0,
        'city': 'Seattle'
    }])

    pred = model.predict(sample)
    assert len(pred) == 1
    assert np.isfinite(pred[0]), f"Prediction must be finite, got: {pred[0]}"
    assert pred[0] > 0, f"Predicted price should be positive, got: {pred[0]}"

def test_unknown_categorical_handling(model):
    """Verifies that OneHotEncoder handles unknown cities gracefully without throwing errors."""
    sample = pd.DataFrame([{
        'bedrooms': 4.0,
        'bathrooms': 2.5,
        'sqft_living': 2500,
        'sqft_lot': 8000,
        'floors': 2.0,
        'waterfront': 0,
        'view': 1,
        'condition': 4,
        'sqft_above': 2000,
        'sqft_basement': 500,
        'yr_built': 1995,
        'yr_renovated': 2005,
        'city': 'UnseenCity_Atlantis'
    }])

    pred = model.predict(sample)
    assert len(pred) == 1
    assert np.isfinite(pred[0]), f"Prediction for unknown category must be finite, got: {pred[0]}"

def test_missing_values_imputation(model):
    """Verifies that ColumnTransformer handles null/NaN in numeric inputs via SimpleImputer."""
    sample = pd.DataFrame([{
        'bedrooms': np.nan,
        'bathrooms': 2.0,
        'sqft_living': 1800,
        'sqft_lot': np.nan,
        'floors': 1.0,
        'waterfront': 0,
        'view': 0,
        'condition': 3,
        'sqft_above': 1800,
        'sqft_basement': 0,
        'yr_built': 1975,
        'yr_renovated': np.nan,
        'city': 'Bellevue'
    }])

    pred = model.predict(sample)
    assert len(pred) == 1
    assert np.isfinite(pred[0]), f"Prediction with missing values must be finite, got: {pred[0]}"

def test_batch_prediction_shape(model):
    """Verifies that multi-row batches yield predictions matching input batch size."""
    samples = pd.DataFrame([
        {
            'bedrooms': 2.0, 'bathrooms': 1.0, 'sqft_living': 900, 'sqft_lot': 4000,
            'floors': 1.0, 'waterfront': 0, 'view': 0, 'condition': 3,
            'sqft_above': 900, 'sqft_basement': 0, 'yr_built': 1945, 'yr_renovated': 0, 'city': 'Kent'
        },
        {
            'bedrooms': 5.0, 'bathrooms': 4.0, 'sqft_living': 4500, 'sqft_lot': 20000,
            'floors': 2.0, 'waterfront': 1, 'view': 4, 'condition': 5,
            'sqft_above': 3500, 'sqft_basement': 1000, 'yr_built': 2005, 'yr_renovated': 2010, 'city': 'Medina'
        }
    ])

    preds = model.predict(samples)
    assert len(preds) == 2
    assert all(np.isfinite(p) for p in preds)
    # Luxury waterfront Medina house should predict higher than smaller Kent house
    assert preds[1] > preds[0], f"Expected luxury waterfront house price ({preds[1]}) > modest house ({preds[0]})"
