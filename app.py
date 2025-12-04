import streamlit as st
import joblib
import pandas as pd
from category_encoders import TargetEncoder


# Load model and threshold (not used for decision anymore, just display)
model = joblib.load("best_lead_model.pkl")
best_th = joblib.load("best_threshold.pkl")

st.title("🔮 Lead Category Prediction (High / Low Potential)")
st.write("This prediction uses a **strict 50% threshold** for better clarity.")
st.write(f"📌 Model original optimized threshold (for reference): **{best_th:.2f}**")

# ------------------- Input fields -------------------
source = st.text_input("Source")
agent = st.text_input("Sales Agent")
location = st.text_input("Location")

delivery_mode = st.selectbox("Delivery Mode", [
    "Mode-2", "Mode-3", "Mode-4", "Mode-5"
])

product = st.number_input("Product ID", value=1)
dow = st.number_input("Day of Week", 1, 7)
month = st.number_input("Month Number", 1, 12)
quarter = st.number_input("Quarter", 1, 4)
year = st.number_input("Year", 2018, 2030)

# ------------------- Prediction -------------------
if st.button("Predict Category"):
    df = pd.DataFrame([{
        "Source_Cleaned": source,
        "Sales_Agent": agent,
        "Location": location,
        "Product_ID": product,
        "day_of_week": dow,
        "month_num": month,
        "quarter": quarter,
        "year": year,
        # one-hot delivery modes
        "Delivery_Mode_Mode-2": 1 if delivery_mode=="Mode-2" else 0,
        "Delivery_Mode_Mode-3": 1 if delivery_mode=="Mode-3" else 0,
        "Delivery_Mode_Mode-4": 1 if delivery_mode=="Mode-4" else 0,
        "Delivery_Mode_Mode-5": 1 if delivery_mode=="Mode-5" else 0,
    }])

    prob = model.predict_proba(df)[0][1] * 100  # as %
    pred = "HIGH POTENTIAL" if prob >= 50 else "LOW POTENTIAL"

    st.subheader("🧾 Prediction Result")
    st.write(f"🎯 **Prediction:** `{pred}`")
    st.write(f"📊 **Confidence Score:** `{prob:.2f}%`")
    st.write("📌 Category rule: HIGH if confidence ≥ 50%, else LOW")
