# Install necessary libraries (if not installed)
!pip install streamlit joblib pandas

# Import libraries
import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('trained_model.pkl')

# Define the Streamlit app
st.title("Capsim Inventory Demand Estimator")

# Create inputs for user to enter data
age_expectation = st.number_input('Age Expectation', min_value=0.0, max_value=10.0, step=0.1)
price_lower_expectation = st.number_input('Price Lower Expectation', min_value=0.0, max_value=100.0, step=0.1)
price_upper_expectation = st.number_input('Price Upper Expectation', min_value=0.0, max_value=100.0, step=0.1)
Ideal_Position_PMFT = st.number_input('Ideal Position PMFT', min_value=0.0, max_value=20.0, step=0.1)
Ideal_Position_Size = st.number_input('Ideal Position Size', min_value=0.0, max_value=20.0, step=0.1)
reliability_MTBF_lower_limit = st.number_input('Reliability MTBF Lower Limit', min_value=0, max_value=50000, step=1000)
reliability_MTBF_upper_limit = st.number_input('Reliability MTBF Upper Limit', min_value=0, max_value=50000, step=1000)
PMFT_actual = st.number_input('PMFT Actual', min_value=0.0, max_value=20.0, step=0.1)
size_coordinate_actual = st.number_input('Size Coordinate Actual', min_value=0.0, max_value=20.0, step=0.1)
price_actual = st.number_input('Price Actual', min_value=0.0, max_value=100.0, step=0.1)
MTBF_actual = st.number_input('MTBF Actual', min_value=0, max_value=50000, step=1000)
age_actual = st.number_input('Age Actual', min_value=0.0, max_value=10.0, step=0.1)
awareness_actual = st.number_input('Awareness Actual', min_value=0.0, max_value=100.0, step=1.0)
accessibility_actual = st.number_input('Accessibility Actual', min_value=0.0, max_value=100.0, step=1.0)
Promo_Budget_actual = st.number_input('Promo Budget Actual', min_value=0.0, max_value=5000.0, step=100.0)
Sales_Budget_actual = st.number_input('Sales Budget Actual', min_value=0.0, max_value=5000.0, step=100.0)
age_expectation_importance = st.number_input('Age Expectation Importance', min_value=0, max_value=100, step=1)
price_importance = st.number_input('Price Importance', min_value=0, max_value=100, step=1)
Ideal_Position_Importance = st.number_input('Ideal Position Importance', min_value=0, max_value=100, step=1)
reliability_importance = st.number_input('Reliability Importance', min_value=0, max_value=100, step=1)
Total_Industry_Unit_Demand = st.number_input('Total Industry Unit Demand', min_value=0, max_value=100000, step=1000)

# Prepare the input data for prediction
input_data = pd.DataFrame({
    'age expectation': [age_expectation],
    'price lower expectation': [price_lower_expectation],
    'price upper expectation': [price_upper_expectation],
    'Ideal Position PMFT': [Ideal_Position_PMFT],
    'Ideal Position Size': [Ideal_Position_Size],
    'reliability MTBF lower limit': [reliability_MTBF_lower_limit],
    'reliability MTBF upper limit': [reliability_MTBF_upper_limit],
    'PMFT actual': [PMFT_actual],
    'size coordinate actual': [size_coordinate_actual],
    'price actual': [price_actual],
    'MTBF actual': [MTBF_actual],
    'age actual': [age_actual],
    'awareness actual': [awareness_actual],
    'accessibility actual': [accessibility_actual],
    'Promo Budget actual': [Promo_Budget_actual],
    'Sales Budget actual': [Sales_Budget_actual],
    'age expectation importance': [age_expectation_importance],
    'price importance': [price_importance],
    'Ideal Position Importance': [Ideal_Position_Importance],
    'reliability importance': [reliability_importance],
    'Total Industry Unit Demand': [Total_Industry_Unit_Demand]
})

# Make a prediction when the user submits the data
if st.button('Predict Units Sold'):
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Units Sold: {prediction:.2f}")
