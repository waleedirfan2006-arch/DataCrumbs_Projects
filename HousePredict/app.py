"""
app.py - HousePredict & Physics Pipeline Interactive Streamlit Web Application
Loads the trained .joblib pipeline, dynamically generates input forms,
provides real-time validation, and outputs instantaneous, finite numeric predictions.
"""

import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st

# Configure page layout and aesthetics
st.set_page_config(
    page_title="HousePredict — Intelligent Property Valuation",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for larger input labels and smoother UI
st.markdown(
    """
    <style>
        /* Increase the size of Streamlit number_input and selectbox labels */
        .stNumberInput label, .stSelectbox label {
            font-size: 1.3rem;
            font-weight: 600;
            color: #2e2e3a;
        }
        .stButton button {
            font-size: 1.2rem;
            padding: 0.6rem 1.2rem;
        }
        .prediction-card {
            background: linear-gradient(135deg, #1E3A8A, #2563EB, #38BDF8);
            color: #fff;
            padding: 1.8rem;
            border-radius: 14px;
            box-shadow: 0 8px 20px -5px rgba(37, 99, 235, 0.3);
            text-align: center;
            margin-top: 1.5rem;
        }
        .prediction-card h2 {font-size: 2.2rem; margin: 0.5rem 0;}
    </style>
    """,
    unsafe_allow_html=True,
)


# Supported King County cities from the authentic dataset
SUPPORTED_CITIES = [
    'Algona', 'Auburn', 'Beaux Arts Village', 'Bellevue', 'Black Diamond',
    'Bothell', 'Burien', 'Carnation', 'Clyde Hill', 'Covington',
    'Des Moines', 'Duvall', 'Enumclaw', 'Fall City', 'Federal Way',
    'Inglewood-Finn Hill', 'Issaquah', 'Kenmore', 'Kent', 'Kirkland',
    'Lake Forest Park', 'Maple Valley', 'Medina', 'Mercer Island', 'Milton',
    'Newcastle', 'Normandy Park', 'North Bend', 'Pacific', 'Preston',
    'Ravensdale', 'Redmond', 'Renton', 'Sammamish', 'SeaTac',
    'Seattle', 'Shoreline', 'Skykomish', 'Snoqualmie', 'Snoqualmie Pass',
    'Tukwila', 'Vashon', 'Woodinville', 'Yarrow Point'
]

# Model feature definitions
FEATURE_CONFIGS = {
    'bedrooms': {'type': 'float', 'label': '**Bedrooms**', 'min': 1.0, 'max': 12.0, 'default': 3.0, 'step': 1.0, 'help': 'Number of bedrooms (1 - 12)'},
    'bathrooms': {'type': 'float', 'label': '**Bathrooms**', 'min': 1.0, 'max': 8.0, 'default': 2.25, 'step': 0.25, 'help': 'Number of bathrooms (1 - 8)'},
    'sqft_living': {'type': 'int', 'label': '**Living space (sqft)**', 'min': 300, 'max': 15000, 'default': 2100, 'step': 50, 'help': 'Finished interior square footage'},
    'sqft_lot': {'type': 'int', 'label': '**Lot size (sqft)**', 'min': 400, 'max': 1200000, 'default': 7500, 'step': 100, 'help': 'Total land parcel area in sqft'},
    'floors': {'type': 'float', 'label': '**Floors / Levels**', 'min': 1.0, 'max': 3.5, 'default': 1.5, 'step': 0.5, 'help': 'Number of building levels'},
    'waterfront': {'type': 'binary', 'label': '**Waterfront Property**', 'options': [('No Waterfront', 0), ('Waterfront Property', 1)], 'default': 0, 'help': 'Direct water frontage'},
    'view': {'type': 'int', 'label': '**View quality (0‑4)**', 'min': 0, 'max': 4, 'default': 0, 'step': 1, 'help': 'View quality rating score (0 = None, 4 = Panoramic view)'},
    'condition': {'type': 'int', 'label': 'Overall Condition (1-5)', 'min': 1, 'max': 5, 'default': 3, 'step': 1, 'help': 'Property physical condition score'},
    'sqft_above': {'type': 'int', 'label': 'Above-Ground Area (sqft)', 'min': 300, 'max': 12000, 'default': 1700, 'step': 50, 'help': 'Interior living area above ground grade'},
    'sqft_basement': {'type': 'int', 'label': 'Basement Area (sqft)', 'min': 0, 'max': 5000, 'default': 400, 'step': 50, 'help': 'Basement area (0 if no basement)'},
    'yr_built': {'type': 'int', 'label': 'Year Built', 'min': 1900, 'max': 2026, 'default': 1985, 'step': 1, 'help': 'Original construction year'},
    'yr_renovated': {'type': 'int', 'label': 'Year Renovated', 'min': 0, 'max': 2026, 'default': 0, 'step': 1, 'help': 'Year of last structural remodel (0 = never renovated)'},
    'city': {'type': 'categorical', 'label': 'City / Locality', 'options': SUPPORTED_CITIES, 'default': 'Seattle', 'help': 'Municipality within King County, WA'}
}

@st.cache_resource
def load_pipeline():
    """
    Locates and deserializes the fitted scikit-learn pipeline artifact.
    Searches both primary and compatibility paths.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(base_dir, "model", "house_price_pipeline.joblib"),
        os.path.join(base_dir, "model", "antigravity_pipeline.joblib"),
        os.path.join("model", "house_price_pipeline.joblib"),
        os.path.join("HousePredict", "model", "house_price_pipeline.joblib"),
        os.path.join("HousePredict", "model", "antigravity_pipeline.joblib")
    ]
    for path in candidates:
        if os.path.exists(path):
            return joblib.load(path)
    raise FileNotFoundError(f"Model pipeline artifact not found in candidates: {candidates}")

def validate_inputs(form_values):
    """
    Performs real-time validation on user-provided field inputs.
    Returns (is_valid: bool, errors: list[str], warnings: list[str]).
    """
    errors = []
    warnings = []

    # Verify positive non-zero measurements
    if form_values['sqft_living'] <= 0:
        errors.append("Living area (sqft_living) must be strictly greater than 0.")
    if form_values['sqft_lot'] <= 0:
        errors.append("Lot size (sqft_lot) must be strictly greater than 0.")
    if form_values['bedrooms'] <= 0:
        errors.append("Number of bedrooms must be greater than 0.")
    if form_values['bathrooms'] <= 0:
        errors.append("Number of bathrooms must be greater than 0.")

    # Consistency check: sqft_living vs sqft_above + sqft_basement
    expected_living = form_values['sqft_above'] + form_values['sqft_basement']
    if form_values['sqft_living'] != expected_living:
        warnings.append(
            f"Living area ({form_values['sqft_living']:,} sqft) differs from above-ground + basement "
            f"({expected_living:,} sqft). Model will evaluate both dimensions."
        )

    # Renovation year validation
    if form_values['yr_renovated'] > 0 and form_values['yr_renovated'] < form_values['yr_built']:
        warnings.append("Year renovated is older than year built. Setting to year built.")
        form_values['yr_renovated'] = form_values['yr_built']

    return (len(errors) == 0), errors, warnings

def predict_price(pipeline, input_dict):
    """
    Executes pipeline inference on input data dictionary.
    Guarantees finite, non-null numerical output.
    """
    df_sample = pd.DataFrame([input_dict])
    raw_pred = pipeline.predict(df_sample)[0]

    if not np.isfinite(raw_pred):
        raise ValueError(f"Model output non-finite prediction: {raw_pred}")

    # Enforce realistic minimum property threshold
    clean_price = float(max(raw_pred, 15000.0))
    return clean_price

def main():
    # Modern styling
    st.markdown("""
        <style>
        .hero-title {
            font-size: 2.3rem;
            font-weight: 800;
            color: #0F172A;
            margin-bottom: 0.2rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .hero-desc {
            font-size: 1.05rem;
            color: #475569;
            margin-bottom: 1.5rem;
            line-height: 1.5;
        }
        .pred-container {
            background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 50%, #38BDF8 100%);
            color: #FFFFFF;
            padding: 1.8rem;
            border-radius: 14px;
            box-shadow: 0 10px 25px -5px rgba(37, 99, 235, 0.3);
            margin: 1.2rem 0;
            text-align: center;
        }
        .pred-label {
            font-size: 1.05rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
            opacity: 0.9;
        }
        .pred-amount {
            font-size: 2.9rem;
            font-weight: 900;
            letter-spacing: -0.5px;
            margin: 0.3rem 0;
        }
        .pred-sub {
            font-size: 0.95rem;
            opacity: 0.9;
        }
        .card-stat {
            background: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-radius: 10px;
            padding: 1rem;
            text-align: center;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="hero-title">🏡 HousePredict — Property Valuation Platform</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-desc">Multiple Linear Regression estimation engine trained on King County, WA real estate. '
        'Delivering instant, data-driven property valuation for buyers, sellers, and property professionals.</div>',
        unsafe_allow_html=True
    )

    # 1. Load Model Pipeline
    try:
        pipeline = load_pipeline()
    except Exception as e:
        st.error(f"❌ Could not load trained model pipeline: {e}")
        st.info("Run `python model/train_model.py` to train and export the pipeline artifact.")
        return

    # Sidebar: Model and Dataset Diagnostics
    with st.sidebar:
        st.header("⚙️ Model Metadata")
        st.markdown("**Algorithm**: Multiple Linear Regression")
        st.markdown("**Preprocessing**: SimpleImputer + StandardScaler + OneHotEncoder")
        st.markdown("**Target**: `price` (USD $)")
        st.markdown("**Region**: King County, Washington State")
        st.markdown("**Training Metrics**:")
        st.markdown("- **R² Score**: `0.6780`")
        st.markdown("- **MAE**: `$131,458.59`")
        st.markdown("- **RMSE**: `$218,879.58`")
        st.divider()
        st.markdown("💡 **Tip**: Enter structural characteristics on the right to receive an instantaneous fair market valuation.")

    # Main Input Form with dynamic inputs
    with st.form("valuation_form"):
        st.subheader("📋 Property Specification & Attributes")
        
        tab1, tab2, tab3 = st.columns(3)

        with tab1:
            st.markdown("##### 📍 Location & Space")
            city = st.selectbox(
                FEATURE_CONFIGS['city']['label'],
                options=FEATURE_CONFIGS['city']['options'],
                index=FEATURE_CONFIGS['city']['options'].index(FEATURE_CONFIGS['city']['default']),
                help=FEATURE_CONFIGS['city']['help']
            )
            sqft_living = st.number_input(
                FEATURE_CONFIGS['sqft_living']['label'],
                min_value=FEATURE_CONFIGS['sqft_living']['min'],
                max_value=FEATURE_CONFIGS['sqft_living']['max'],
                value=FEATURE_CONFIGS['sqft_living']['default'],
                step=FEATURE_CONFIGS['sqft_living']['step'],
                help=FEATURE_CONFIGS['sqft_living']['help']
            )
            sqft_lot = st.number_input(
                FEATURE_CONFIGS['sqft_lot']['label'],
                min_value=FEATURE_CONFIGS['sqft_lot']['min'],
                max_value=FEATURE_CONFIGS['sqft_lot']['max'],
                value=FEATURE_CONFIGS['sqft_lot']['default'],
                step=FEATURE_CONFIGS['sqft_lot']['step'],
                help=FEATURE_CONFIGS['sqft_lot']['help']
            )
            floors = st.number_input(
                FEATURE_CONFIGS['floors']['label'],
                min_value=FEATURE_CONFIGS['floors']['min'],
                max_value=FEATURE_CONFIGS['floors']['max'],
                value=FEATURE_CONFIGS['floors']['default'],
                step=FEATURE_CONFIGS['floors']['step'],
                help=FEATURE_CONFIGS['floors']['help']
            )

        with tab2:
            st.markdown("##### 🛏️ Rooms & Layout")
            bedrooms = st.number_input(
                FEATURE_CONFIGS['bedrooms']['label'],
                min_value=FEATURE_CONFIGS['bedrooms']['min'],
                max_value=FEATURE_CONFIGS['bedrooms']['max'],
                value=FEATURE_CONFIGS['bedrooms']['default'],
                step=FEATURE_CONFIGS['bedrooms']['step'],
                help=FEATURE_CONFIGS['bedrooms']['help']
            )
            bathrooms = st.number_input(
                FEATURE_CONFIGS['bathrooms']['label'],
                min_value=FEATURE_CONFIGS['bathrooms']['min'],
                max_value=FEATURE_CONFIGS['bathrooms']['max'],
                value=FEATURE_CONFIGS['bathrooms']['default'],
                step=FEATURE_CONFIGS['bathrooms']['step'],
                help=FEATURE_CONFIGS['bathrooms']['help']
            )
            sqft_above = st.number_input(
                FEATURE_CONFIGS['sqft_above']['label'],
                min_value=FEATURE_CONFIGS['sqft_above']['min'],
                max_value=FEATURE_CONFIGS['sqft_above']['max'],
                value=FEATURE_CONFIGS['sqft_above']['default'],
                step=FEATURE_CONFIGS['sqft_above']['step'],
                help=FEATURE_CONFIGS['sqft_above']['help']
            )
            sqft_basement = st.number_input(
                FEATURE_CONFIGS['sqft_basement']['label'],
                min_value=FEATURE_CONFIGS['sqft_basement']['min'],
                max_value=FEATURE_CONFIGS['sqft_basement']['max'],
                value=FEATURE_CONFIGS['sqft_basement']['default'],
                step=FEATURE_CONFIGS['sqft_basement']['step'],
                help=FEATURE_CONFIGS['sqft_basement']['help']
            )

        with tab3:
            st.markdown("##### 🌟 Features & History")
            waterfront_choice = st.selectbox(
                FEATURE_CONFIGS['waterfront']['label'],
                options=[opt[0] for opt in FEATURE_CONFIGS['waterfront']['options']],
                index=0,
                help=FEATURE_CONFIGS['waterfront']['help']
            )
            waterfront = 1 if "Waterfront Property" in waterfront_choice else 0

            view = st.slider(
                FEATURE_CONFIGS['view']['label'],
                min_value=FEATURE_CONFIGS['view']['min'],
                max_value=FEATURE_CONFIGS['view']['max'],
                value=FEATURE_CONFIGS['view']['default'],
                help=FEATURE_CONFIGS['view']['help']
            )
            condition = st.slider(
                FEATURE_CONFIGS['condition']['label'],
                min_value=FEATURE_CONFIGS['condition']['min'],
                max_value=FEATURE_CONFIGS['condition']['max'],
                value=FEATURE_CONFIGS['condition']['default'],
                help=FEATURE_CONFIGS['condition']['help']
            )
            yr_built = st.number_input(
                FEATURE_CONFIGS['yr_built']['label'],
                min_value=FEATURE_CONFIGS['yr_built']['min'],
                max_value=FEATURE_CONFIGS['yr_built']['max'],
                value=FEATURE_CONFIGS['yr_built']['default'],
                step=FEATURE_CONFIGS['yr_built']['step'],
                help=FEATURE_CONFIGS['yr_built']['help']
            )
            yr_renovated = st.number_input(
                FEATURE_CONFIGS['yr_renovated']['label'],
                min_value=FEATURE_CONFIGS['yr_renovated']['min'],
                max_value=FEATURE_CONFIGS['yr_renovated']['max'],
                value=FEATURE_CONFIGS['yr_renovated']['default'],
                step=FEATURE_CONFIGS['yr_renovated']['step'],
                help=FEATURE_CONFIGS['yr_renovated']['help']
            )

        submit_btn = st.form_submit_button("🚀 Compute Real-Time Valuation", use_container_width=True)

    if submit_btn:
        form_payload = {
            'bedrooms': float(bedrooms),
            'bathrooms': float(bathrooms),
            'sqft_living': int(sqft_living),
            'sqft_lot': int(sqft_lot),
            'floors': float(floors),
            'waterfront': int(waterfront),
            'view': int(view),
            'condition': int(condition),
            'sqft_above': int(sqft_above),
            'sqft_basement': int(sqft_basement),
            'yr_built': int(yr_built),
            'yr_renovated': int(yr_renovated),
            'city': str(city)
        }

        # Real‑time validation
        is_valid, errors, warnings = validate_inputs(form_payload)

        for w in warnings:
            st.warning(f"⚠️ {w}")

        if not is_valid:
            for e in errors:
                st.error(f"❌ {e}")
        else:
            try:
                predicted_val = predict_price(pipeline, form_payload)

                # Dynamic placeholder for the prediction card
                placeholder = st.empty()
                placeholder.markdown(f"""
                    <div class='prediction-card'>
                        <h2>💰 Estimated price</h2>
                        <p style='font-size:2rem; margin:0.5rem 0;'>${predicted_val:,.2f} USD</p>
                        <small>for {city}, WA • {sqft_living:,} sqft • {int(bedrooms)} beds, {bathrooms} baths</small>
                    </div>
                """, unsafe_allow_html=True)

                # Fun animation to emphasize result
                st.balloons()

                # Context breakdown cards (kept compact)
                m1, m2, m3, m4 = st.columns(4)
                price_per_sqft = predicted_val / max(sqft_living, 1)
                with m1:
                    st.metric("$ / sqft", f"${price_per_sqft:,.2f}")
                with m2:
                    st.metric("City", city)
                with m3:
                    st.metric("Waterfront", "Yes" if waterfront else "No")
                with m4:
                    st.metric("Age", f"{2026 - yr_built} yr")

            except Exception as exc:
                st.error(f"❌ Error during model prediction: {exc}")

if __name__ == '__main__':
    main()
