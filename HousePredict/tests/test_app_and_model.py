"""
test_app_and_model.py
======================
Automated test suite verifying:
1. Model loading from joblib artifact
2. Streamlit app import and initialization logic
3. Form input validation and error detection
4. Instantaneous finite numeric prediction computation
5. Handling of missing/unseen categorical values
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
import joblib

# Ensure HousePredict directory is importable
APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

import app

@pytest.fixture
def loaded_pipeline():
    """Fixture ensuring pipeline artifact loads from disk."""
    pipeline = app.load_pipeline()
    assert pipeline is not None
    assert hasattr(pipeline, 'predict')
    return pipeline

def test_pipeline_loading(loaded_pipeline):
    """Verifies that the serialized pipeline contains expected preprocessing and regression steps."""
    assert 'preprocessor' in loaded_pipeline.named_steps
    assert 'regressor' in loaded_pipeline.named_steps

def test_instantaneous_valid_prediction(loaded_pipeline):
    """Verifies that a valid input payload yields a finite positive numeric valuation."""
    valid_payload = {
        'bedrooms': 3.0,
        'bathrooms': 2.0,
        'sqft_living': 1900,
        'sqft_lot': 6500,
        'floors': 1.0,
        'waterfront': 0,
        'view': 0,
        'condition': 3,
        'sqft_above': 1400,
        'sqft_basement': 500,
        'yr_built': 1980,
        'yr_renovated': 0,
        'city': 'Seattle'
    }

    price = app.predict_price(loaded_pipeline, valid_payload)
    assert isinstance(price, (int, float))
    assert np.isfinite(price)
    assert price > 10000.0, f"Expected realistic valuation > $10,000, got: {price}"

def test_input_validation_rules():
    """Verifies that validate_inputs catches invalid non-positive inputs."""
    # Invalid zero bedrooms and negative sqft
    invalid_payload = {
        'bedrooms': 0.0,
        'bathrooms': 0.0,
        'sqft_living': -100,
        'sqft_lot': 0,
        'floors': 1.0,
        'waterfront': 0,
        'view': 0,
        'condition': 3,
        'sqft_above': 1000,
        'sqft_basement': 0,
        'yr_built': 1990,
        'yr_renovated': 0,
        'city': 'Seattle'
    }

    is_valid, errors, warnings = app.validate_inputs(invalid_payload)
    assert not is_valid
    assert len(errors) >= 3

def test_input_validation_warnings_on_inconsistent_area():
    """Verifies that warning is issued if sqft_living differs from above + basement."""
    mismatched_payload = {
        'bedrooms': 3.0,
        'bathrooms': 2.0,
        'sqft_living': 2500,
        'sqft_lot': 5000,
        'floors': 1.5,
        'waterfront': 0,
        'view': 0,
        'condition': 3,
        'sqft_above': 1500,
        'sqft_basement': 500,  # Sum is 2000 != 2500
        'yr_built': 1980,
        'yr_renovated': 0,
        'city': 'Redmond'
    }

    is_valid, errors, warnings = app.validate_inputs(mismatched_payload)
    assert is_valid  # Still valid to predict, but emits warning
    assert any("differs from above-ground" in w for w in warnings)

def test_unseen_city_handling(loaded_pipeline):
    """Verifies that an unseen city does not cause inference crashes due to handle_unknown='ignore'."""
    unseen_payload = {
        'bedrooms': 4.0,
        'bathrooms': 2.5,
        'sqft_living': 2200,
        'sqft_lot': 7000,
        'floors': 2.0,
        'waterfront': 0,
        'view': 0,
        'condition': 4,
        'sqft_above': 2200,
        'sqft_basement': 0,
        'yr_built': 2000,
        'yr_renovated': 0,
        'city': 'Unknown_Metropolis_X'
    }

    price = app.predict_price(loaded_pipeline, unseen_payload)
    assert np.isfinite(price)
    assert price > 0

def test_relative_valuation_logic(loaded_pipeline):
    """Verifies that high-value luxury property in Medina predicts higher than modest house in Kent."""
    modest_house = {
        'bedrooms': 2.0, 'bathrooms': 1.0, 'sqft_living': 900, 'sqft_lot': 4000,
        'floors': 1.0, 'waterfront': 0, 'view': 0, 'condition': 3,
        'sqft_above': 900, 'sqft_basement': 0, 'yr_built': 1950, 'yr_renovated': 0,
        'city': 'Kent'
    }

    luxury_house = {
        'bedrooms': 5.0, 'bathrooms': 4.5, 'sqft_living': 5200, 'sqft_lot': 25000,
        'floors': 2.0, 'waterfront': 1, 'view': 4, 'condition': 5,
        'sqft_above': 3800, 'sqft_basement': 1400, 'yr_built': 2010, 'yr_renovated': 2012,
        'city': 'Medina'
    }

    p_modest = app.predict_price(loaded_pipeline, modest_house)
    p_luxury = app.predict_price(loaded_pipeline, luxury_house)

    assert p_luxury > p_modest, f"Luxury house ({p_luxury}) should exceed modest house ({p_modest})"
