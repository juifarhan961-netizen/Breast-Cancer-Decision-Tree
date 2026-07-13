import streamlit as st
import joblib
import pandas as pd

model = joblib.load("decision_tree_model.pkl")

st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🩺"
)

st.title("🩺 Breast Cancer Prediction")
st.write("Enter patient details below to predict the diagnosis.")

input_data = {}

with st.form("prediction_form"):

    for feature in model.feature_names_in_:
        input_data[feature] = st.number_input(
            feature.replace("_", " ").title(),
            min_value=0.0,
            value=0.0
        )

    predict_button = st.form_submit_button("Predict Diagnosis")

if predict_button:

    data = pd.DataFrame([input_data])

    prediction = model.predict(data)

    if prediction[0] == "M":
        st.error("⚠️ Prediction: Malignant")
    else:
        st.success("✅ Prediction: Benign")

    st.caption("This prediction is for educational purposes only.")
