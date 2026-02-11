import joblib
import os
import numpy as np
import pandas as pd

# Define robust paths relative to the script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '../../'))
MODEL_DIR = os.path.join(PROJECT_ROOT, 'backend', 'models')

# Load artifacts
MODEL_PATH = os.path.join(MODEL_DIR, 'illness_risk_model.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
ENCODER_PATH = os.path.join(MODEL_DIR, 'label_encoder.pkl')
FEATURE_NAMES_PATH = os.path.join(MODEL_DIR, 'feature_names.pkl')

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
le = joblib.load(ENCODER_PATH)
feature_names = joblib.load(FEATURE_NAMES_PATH)

def test_samples():
    # Low-risk sample (based on generate_perfect_data.py logic: low age, healthy BMI, low BP, low glucose, normal temp)
    low_risk_sample = {'Age':25,'Gender':'Male','BMI':22.0,'Systolic_BP':110.0,'Glucose':90.0,'Body_Temp':98.6}
    
    # High-risk sample (high age, high BMI, high BP, high glucose, high temp)
    high_risk_sample = {'Age':75,'Gender':'Female','BMI':150.0,'Systolic_BP':180.0,'Glucose':180.0,'Body_Temp':102.5}

    samples = [low_risk_sample, high_risk_sample]
    
    print("=" * 50)
    print("TESTING PREDICTIONS")
    print("=" * 50)

    for i, sample in enumerate(samples):
        # Create DataFrame to handle gender mapping and column ordering
        df = pd.DataFrame([sample])
        df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1, 'Transgender': 2})
        
        # Ensure correct order of columns
        df = df[feature_names]
        
        # Scale
        x_scaled = scaler.transform(df)
        
        # Predict
        prob = model.predict_proba(x_scaled)[0][1] # Probability of Class 1 (High Risk)
        pred_class = 1 if prob > 0.5 else 0
        label = le.inverse_transform([pred_class])[0]
        
        print(f"\nSample {i+1}: {'High' if i==1 else 'Low'} Risk Expected")
        print('Input:', sample)
        print('Prob (High Risk):', f"{prob:.4%}")
        print('Predicted Class:', pred_class, 'Label:', label)
        
        expected_label = 'High Risk' if i==1 else 'Low Risk'
        if label == expected_label:
            print(f"✓ Match: {label}")
        else:
            print(f"✗ Mismatch: Expected {expected_label}, got {label}")

if __name__ == "__main__":
    test_samples()
