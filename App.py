import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the trained model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# App title
st.title("💧Water Potability Prediction")
st.markdown("Provide the water quality parameters below to check if the water is safe to drink.")

# User input fields
ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=7.0)
hardness = st.number_input("Hardness", value=200.0)
tds = st.number_input("Total Dissolved Solids (TDS)", value=15000.0)
chloramines = st.number_input("Chloramines", value=7.0)
sulfate = st.number_input("Sulfate", value=350.0)
conductivity = st.number_input("Conductivity", value=400.0)
organic_carbon = st.number_input("Organic Carbon", value=13.0)
trihalomethanes = st.number_input("Trihalomethanes", value=60.0)
turbidity = st.number_input("Turbidity", value=4.0)

# Predict button
if st.button("Predict Water Potability"):
    # Prepare feature array
    features = np.array([[ph, hardness, tds, chloramines, sulfate, conductivity,
                          organic_carbon, trihalomethanes, turbidity]])
    
    # Scale the features
    features_scaled = scaler.transform(features)
    
    # Make prediction
    pred_prob = model.predict_proba(features_scaled)[0]
    pred = model.predict(features_scaled)[0]

    # Show prediction result
    st.subheader("🔍 Prediction Result")
    st.markdown(f"**Probability - Not Potable**: `{pred_prob[0]:.2f}`")
    st.markdown(f"**Probability - Potable**: `{pred_prob[1]:.2f}`")
    
    if pred == 1:
        st.success("✅ Prediction: Water is **Safe** to Drink.")
    else:
        st.error("❌ Prediction: Water is **Not Safe** to Drink.")

    # 🩺 Health Tips Based on User Input
    st.markdown("### 🩺 Health Tips Based on Your Inputs")
    
    if ph < 6.5 or ph > 8.5:
        st.warning("⚠️ pH is outside the ideal range (6.5 - 8.5). Can cause corrosion, leaching of metals, or bad taste.")
    if hardness > 300:
        st.warning("⚠️ Hardness is high (>300 mg/L). May cause scale buildup and taste issues.")
    if tds > 1000:
        st.warning("⚠️ TDS is high (>1000 mg/L). Can cause laxative effect and affect taste.")
    if chloramines > 4:
        st.warning("⚠️ Chloramines exceed safe level (>4 mg/L). May affect kidneys or cause respiratory issues.")
    if sulfate > 250:
        st.warning("⚠️ Sulfate is high (>250 mg/L). Can cause diarrhea and bitter taste.")
    if conductivity > 600:
        st.warning("⚠️ Conductivity is high (>600 µS/cm). Indicates presence of excess ions.")
    if trihalomethanes > 0.1:
        st.warning("⚠️ Trihalomethanes exceed safe limit (>0.1 mg/L). Long-term exposure may increase cancer risk.")
    if turbidity > 5:
        st.warning("⚠️ Turbidity is high (>5 NTU). May indicate microbial contamination.")
    if organic_carbon > 15:
        st.warning("⚠️ Organic Carbon is high (>15 mg/L). Promotes microbial growth in water.")

# WHO Guidelines
st.markdown("---")
st.markdown("### 🌍 WHO Guidelines for Safe Drinking Water")

who_data = {
    "Parameter": ["pH", "Hardness", "TDS", "Chloramines", "Sulfate", "Conductivity",
                  "Organic Carbon", "Trihalomethanes", "Turbidity"],
    "WHO Limit": ["6.5 – 8.5", "≤ 300 mg/L", "≤ 1000 mg/L", "≤ 4 mg/L", "≤ 250 mg/L", 
                  "≤ 400–600 µS/cm", "No fixed limit", "≤ 0.1 mg/L", "≤ 5 NTU"],
    "Health Risk if Exceeded": [
        "Corrosion, metal leaching", "Taste, scale buildup", "Taste, laxative effect",
        "Respiratory/kidney damage", "Diarrhea, bitter taste", "Indicator of high ions",
        "Supports microbial growth", "Carcinogenic byproduct", "Microbial contamination risk"
    ]
}

who_df = pd.DataFrame(who_data)
st.table(who_df.set_index("Parameter"))
