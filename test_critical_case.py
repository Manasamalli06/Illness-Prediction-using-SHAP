import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import joblib
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

# Load model and artifacts
model = load_model('models/illness_risk_model.keras')
scaler = joblib.load('models/scaler.pkl')
label_encoder = joblib.load('models/label_encoder.pkl')
feature_names = joblib.load('models/feature_names.pkl')

print("=" * 60)
print("CRITICAL CASE TEST")
print("=" * 60)

# Test case from user: Age 68, BMI 34.5, SysBP 168, Glucose 215, Temp 102.4
test_case = {
    'Age': 68,
    'Gender': 'Female',  # arbitrary
    'BMI': 34.5,
    'Systolic_BP': 168,
    'Glucose': 215,
    'Body_Temp': 102.4
}

print(f"\nInput: {test_case}")

# Prepare data
input_data = {
    'Age': [test_case['Age']],
    'Gender': [test_case['Gender']],
    'BMI': [test_case['BMI']],
    'Systolic_BP': [test_case['Systolic_BP']],
    'Glucose': [test_case['Glucose']],
    'Body_Temp': [test_case['Body_Temp']]
}

df = pd.DataFrame(input_data)
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1, 'Transgender': 2})
df = df[feature_names]

print(f"\nFeatures (scaled order):\n{df}")

# Scale
df_scaled = scaler.transform(df)
print(f"\nScaled features:\n{df_scaled}")

# Predict
prediction_prob = model.predict(df_scaled, verbose=0)[0][0]
print(f"\nRaw model output (P(High Risk)): {prediction_prob:.4f}")

prob_high = prediction_prob
final_class = 1 if prob_high > 0.5 else 0
risk_label = 'High Risk' if final_class == 1 else 'Low Risk'

print(f"Threshold: 0.5")
print(f"Decision: {prob_high:.4f} > 0.5? {prob_high > 0.5}")
print(f"Predicted Class: {final_class}")
print(f"Predicted Label: {risk_label}")
print(f"\n{'✓ CORRECT' if risk_label == 'High Risk' else '✗ WRONG - Should be High Risk'}")

print("\n" + "=" * 60)
