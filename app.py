import streamlit as st
import joblib
import pandas as pd

model = joblib.load("decision_tree_model.pkl")

st.title("Breast Cancer Prediction")

input_data = {}

for feature in model.feature_names_in_:
    input_data[feature] = st.number_input(feature)

if st.button("Predict"):
    data = pd.DataFrame([input_data])
    prediction = model.predict(data)

    if prediction[0] == "M":
        st.error("Malignant")
    else:
        st.success("Benign")
