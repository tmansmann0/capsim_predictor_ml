
import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load('capsim_units_sold_model.pkl')

# Function to make predictions
def predict_units_sold(age_exp, price_lower_exp, price_upper_exp, pos_pmft, pos_size, 
                       reliability_lower, reliability_upper, pmft_actual, size_actual, 
                       price_actual, mtbf_actual, age_actual, awareness_actual, 
                       accessibility_actual, promo_budget, sales_budget, age_imp, price_imp, 
                       pos_imp, reliability_imp, total_demand):
    
    features = np.array([[age_exp, price_lower_exp, price_upper_exp, pos_pmft, pos_size, 
                          reliability_lower, reliability_upper, pmft_actual, size_actual, 
                          price_actual, mtbf_actual, age_actual, awareness_actual, 
                          accessibility_actual, promo_budget, sales_budget, age_imp, price_imp, 
                          pos_imp, reliability_imp, total_demand]])
    
    prediction = model.predict(features)
    return prediction

# Streamlit UI
st.title("Capsim Units Sold Predictor")

st.write("This model predicts units sold based on various product and market factors. Please note, the estimate might be off by approximately 163 units.")

# Input fields for all features
age_exp = st.number_input("Age Expectation", min_value=0.0, step=0.1)
price_lower_exp = st.number_input("Price Lower Expectation", min_value=0.0, step=0.1)
price_upper_exp = st.number_input("Price Upper Expectation", min_value=0.0, step=0.1)
pos_pmft = st.number_input("Ideal Position PMFT", min_value=0.0, step=0.1)
pos_size = st.number_input("Ideal Position Size", min_value=0.0, step=0.1)
reliability_lower = st.number_input("Reliability MTBF Lower Limit", min_value=0.0, step=100)
reliability_upper = st.number_input("Reliability MTBF Upper Limit", min_value=0.0, step=100)
pmft_actual = st.number_input("PMFT Actual", min_value=0.0, step=0.1)
size_actual = st.number_input("Size Coordinate Actual", min_value=0.0, step=0.1)
price_actual = st.number_input("Price Actual", min_value=0.0, step=0.1)
mtbf_actual = st.number_input("MTBF Actual", min_value=0.0, step=100)
age_actual = st.number_input("Age Actual", min_value=0.0, step=0.1)
awareness_actual = st.number_input("Awareness Actual", min_value=0.0, step=0.1)
accessibility_actual = st.number_input("Accessibility Actual", min_value=0.0, step=0.1)
promo_budget = st.number_input("Promo Budget Actual", min_value=0.0, step=1000)
sales_budget = st.number_input("Sales Budget Actual", min_value=0.0, step=1000)
age_imp = st.number_input("Age Expectation Importance", min_value=0.0, step=1.0)
price_imp = st.number_input("Price Importance", min_value=0.0, step=1.0)
pos_imp = st.number_input("Ideal Position Importance", min_value=0.0, step=1.0)
reliability_imp = st.number_input("Reliability Importance", min_value=0.0, step=1.0)
total_demand = st.number_input("Total Industry Unit Demand", min_value=0.0, step=100)

# Predict button
if st.button("Predict Units Sold"):
    result = predict_units_sold(age_exp, price_lower_exp, price_upper_exp, pos_pmft, pos_size, 
                                reliability_lower, reliability_upper, pmft_actual, size_actual, 
                                price_actual, mtbf_actual, age_actual, awareness_actual, 
                                accessibility_actual, promo_budget, sales_budget, age_imp, price_imp, 
                                pos_imp, reliability_imp, total_demand)
    
    st.write(f"Predicted Units Sold: {result[0]:.2f}")
    st.write("Please note that the prediction may be off by approximately 163 units in either direction.")
