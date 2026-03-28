import sys
import os
sys.path.insert(0, r"c:\Users\MANASA\OneDrive\Desktop\Illness-Prediction-using-SHAP\backend\app")

import app as my_app
my_app.load_artifacts()

print("Loaded Model:", my_app.model)

my_app.initialize_shap_explainer()

print("Explainer:", my_app.explainer)

import pandas as pd
import numpy as np

# Mock input
input_data = {
    'Age': [45],
    'Gender': [0], # 'Male'
    'BMI': [25.0],
    'Systolic_BP': [120.0],
    'Glucose': [90.0],
    'Body_Temp': [98.6]
}
df = pd.DataFrame(input_data)
df = df[my_app.feature_names]
df_scaled = my_app.scaler.transform(df)

base64_str = my_app.generate_shap_plot(df_scaled, 0.2, "Low Risk")
print("SHAP base64 length:", len(str(base64_str)) if base64_str else "None")
