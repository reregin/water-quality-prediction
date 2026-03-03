# app/main.py
import streamlit as st
import sys
import os

# Add project root to path so we can import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.inference import make_prediction
import src.config as config

# --- UI Setup ---
st.set_page_config(page_title=f"{config.MODEL_NAME} Predictor", page_icon="💳")

st.title("💳 CreditGuard: Risk Analyzer")
st.markdown("Enter customer details to predict default risk.")

# --- Input Form ---
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        limit_bal = st.number_input("Credit Limit ($)", min_value=0, value=10000)
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
    
    with col2:
        bill_amt = st.number_input("Last Bill Amount ($)", value=0)
        pay_amt = st.number_input("Last Payment Amount ($)", value=0)
    
    # Add more inputs here matching dataset columns...
    
    submitted = st.form_submit_button("Analyze Risk")

# --- Logic ---
if submitted:
    # 1. Prepare Data
    input_data = {
        "limit_bal": limit_bal,
        "age": age,
        "bill_amt1": bill_amt,
        "pay_amt1": pay_amt
        # Ensure these keys match the column names in the training data!
    }
    
    # 2. Call the Engine
    with st.spinner("Analyzing..."):
        result = make_prediction(input_data)
        
    # 3. Display Result
    if result["status"] == "Success":
        prob = result["probability"]
        
        if prob > 0.5:
            st.error(f"⚠️ High Risk! (Probability: {prob:.2%})")
        else:
            st.success(f"✅ Low Risk. (Probability: {prob:.2%})")
            
    else:
        st.error(f"Error: {result['error']}")