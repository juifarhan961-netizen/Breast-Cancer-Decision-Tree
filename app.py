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

mean_features = [
    feature for feature in model.feature_names_in_
    if feature.endswith("_mean")
]

se_features = [
    feature for feature in model.feature_names_in_
    if feature.endswith("_se")
]

worst_features = [
    feature for feature in model.feature_names_in_
    if feature.endswith("_worst")
]

with st.form("prediction_form"):

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Mean Features")

        for feature in mean_features:
            input_data[feature] = st.number_input(
                feature.replace("_", " ").title(),
                min_value=0.0
            )

    with col2:
        st.subheader("SE Features")

        for feature in se_features:
            input_data[feature] = st.number_input(
                feature.replace("_", " ").title(),
                min_value=0.0
            )

    with col3:
        st.subheader("Worst Features")

        for feature in worst_features:
            input_data[feature] = st.number_input(
                feature.replace("_", " ").title(),
                min_value=0.0
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
