import streamlit as st
import joblib
import numpy as np
import os

# Function to load the model with error handling
@st.cache_resource  # Caches the model to avoid reloading on every run
def load_model(model_path):
    if not os.path.exists(model_path):
        st.error(f"Model file not found at path: {model_path}")
        return None
    try:
        model = joblib.load(model_path)
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

# Load the trained model
model = load_model('capsim_units_sold_model.pkl')

# Function to make predictions
def predict_units_sold(age_exp, price_lower_exp, price_upper_exp, pos_pfmn, pos_size, 
                       reliability_lower, reliability_upper, pfmn_actual, size_actual, 
                       price_actual, mtbf_actual, age_actual, awareness_actual, 
                       accessibility_actual, promo_budget, sales_budget, age_imp, price_imp, 
                       pos_imp, reliability_imp, total_demand):
    
    features = np.array([[age_exp, price_lower_exp, price_upper_exp, pos_pfmn, pos_size, 
                          reliability_lower, reliability_upper, pfmn_actual, size_actual, 
                          price_actual, mtbf_actual, age_actual, awareness_actual, 
                          accessibility_actual, promo_budget, sales_budget, age_imp, price_imp, 
                          pos_imp, reliability_imp, total_demand]])
    
    prediction = model.predict(features)
    return prediction

# Streamlit UI
st.title("Capsim Units Sold Predictor")

st.write("""
This model predicts units sold based on various product and market factors.
**Please note:** The estimate might be off by approximately **163 units**.
""")

# Buying Criteria Section
st.markdown("### Buying Criteria")

# Create columns for the table headers
col1, col2, col3, col4, col5 = st.columns([2, 2.5, 2, 2, 2])
col1.write("**Criteria**")
col2.write("**Ideal Value**")
col3.write("**Low End of Range**")
col4.write("**High End of Range**")
col5.write("**Importance (%)**")

# Price Criterion
col1, col2, col3, col4, col5 = st.columns([2, 2.5, 2, 2, 2])
col1.write("Price")
col2.write("N/A")
price_lower_exp = col3.number_input("Low End", key='price_lower_exp', min_value=0.0, step=0.1)
price_upper_exp = col4.number_input("High End", key='price_upper_exp', min_value=0.0, step=0.1)
price_imp = col5.number_input("Importance", key='price_imp', min_value=0.0, step=1.0)

# Age Criterion
col1, col2, col3, col4, col5 = st.columns([2, 2.5, 2, 2, 2])
col1.write("Age")
age_exp = col2.number_input("Ideal Age", key='age_exp', min_value=0.0, step=0.1)
col3.write("N/A")
col4.write("N/A")
age_imp = col5.number_input("Importance", key='age_imp', min_value=0.0, step=1.0)

# Ideal Position Criterion
col1, col2, col3, col4, col5 = st.columns([2, 2.5, 2, 2, 2])
col1.write("Ideal Position")
with col2:
    pos_col1, pos_col2 = st.columns(2)
    pos_pfmn = pos_col1.number_input("Pfmn", key='pos_pfmn', min_value=0.0, step=0.1)
    pos_size = pos_col2.number_input("Size", key='pos_size', min_value=0.0, step=0.1)
col3.write("N/A")
col4.write("N/A")
pos_imp = col5.number_input("Importance", key='pos_imp', min_value=0.0, step=1.0)

# MTBF Criterion
col1, col2, col3, col4, col5 = st.columns([2, 2.5, 2, 2, 2])
col1.write("MTBF")
col2.write("N/A")
reliability_lower = col3.number_input("Low End", key='reliability_lower', min_value=0.0, step=100.0)
reliability_upper = col4.number_input("High End", key='reliability_upper', min_value=0.0, step=100.0)
reliability_imp = col5.number_input("Importance", key='reliability_imp', min_value=0.0, step=1.0)

# Divider
st.markdown("---")

# Actual Product Values
st.markdown("### Actual Product Values")

col1, col2 = st.columns(2)

with col1:
    age_actual = st.number_input("Age (years)", min_value=0.0, step=0.1)
    price_actual = st.number_input("Price ($)", min_value=0.0, step=0.1)
    pfmn_actual = st.number_input("Performance", min_value=0.0, step=0.1)
    size_actual = st.number_input("Size", min_value=0.0, step=0.1)
    mtbf_actual = st.number_input("MTBF (hours)", min_value=0.0, step=100.0)

with col2:
    awareness_actual = st.number_input("Awareness (%)", min_value=0.0, max_value=100.0, step=1.0)
    accessibility_actual = st.number_input("Accessibility (%)", min_value=0.0, max_value=100.0, step=1.0)
    promo_budget = st.number_input("Promo Budget", min_value=0.0, step=1000.0)
    sales_budget = st.number_input("Sales Budget", min_value=0.0, step=1000.0)
    total_demand = st.number_input("Total Industry Unit Demand", min_value=0.0, step=100.0)

# Convert percentage inputs to decimals if needed
awareness_actual /= 100
accessibility_actual /= 100

# Predict button
if st.button("Predict Units Sold"):
    if model is not None:
        try:
            result = predict_units_sold(
                age_exp, price_lower_exp, price_upper_exp, pos_pfmn, pos_size, 
                reliability_lower, reliability_upper, pfmn_actual, size_actual, 
                price_actual, mtbf_actual, age_actual, awareness_actual, 
                accessibility_actual, promo_budget, sales_budget, age_imp, price_imp, 
                pos_imp, reliability_imp, total_demand
            )
            
            st.success(f"**Predicted Units Sold:** {result[0]:.2f}")
            st.info("Please note that the prediction may be off by approximately **163 units** in either direction.")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
    else:
        st.error("Model is not loaded. Please check the model file and dependencies.")
