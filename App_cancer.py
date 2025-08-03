import streamlit as st
import joblib
import pandas as pd
import os

# Absolute path to your model file
model_path = r'D:\Programming\Project\Python for EDA\Class related\Cancer\logistic_regression_model_cancer_detection.pkl'

# Load the model
model = joblib.load(model_path)

# Streamlit app for breast cancer prediction
st.title("Breast Cancer Prediction App")

# Feature names for user input
feature_names = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'compactness_mean',
    'concavity_mean', 'concave points_mean', 'radius_se', 'perimeter_se', 'area_se',
    'concavity_se', 'concave points_se', 'radius_worst', 'texture_worst', 'perimeter_worst',
    'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave points_worst'
]

# Create input fields for each feature
user_input = {}

# Loop through feature names and create input fields
for feature in feature_names:
    step_size = 0.01 if 'smoothness' in feature else 0.1
    label = feature.replace('_', ' ').title()
    user_input[feature] = st.number_input(label, min_value=0.0, step=step_size)

# Button to trigger prediction
if st.button('Predict'):
    input_df = pd.DataFrame([user_input])
    prediction = model.predict(input_df)[0]
    
    if prediction == 1:
        st.error("The tumor is predicted to be **Malignant**.")
    else:
        st.success("The tumor is predicted to be **Benign**.")
