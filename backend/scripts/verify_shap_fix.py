import pandas as pd
import numpy as np
import shap
import joblib
import os
import matplotlib.pyplot as plt
import base64
from io import BytesIO

# --- Path Setup ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '../../'))
MODEL_DIR = os.path.join(PROJECT_ROOT, 'backend', 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'illness_risk_model.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
FEATURE_NAMES_PATH = os.path.join(MODEL_DIR, 'feature_names.pkl')
DATA_FILE = os.path.join(PROJECT_ROOT, 'backend', 'data', 'augmented_medical_data.csv')

def verify_shap():
    print(f"Loading artifacts from {MODEL_DIR}...")
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_names = joblib.load(FEATURE_NAMES_PATH)
    
    # Load background data
    df_bg = pd.read_csv(DATA_FILE)
    df_bg['Gender'] = df_bg['Gender'].map({'Male': 0, 'Female': 1, 'Transgender': 2})
    X_bg = df_bg[feature_names].values
    X_bg_scaled = scaler.transform(X_bg)
    background_sample = X_bg_scaled[:50]
    
    # Create sample patient
    sample_data = {
        'Age': [45],
        'Gender': [0], # Male
        'BMI': [35.0],  # Highish
        'Systolic_BP': [155], # High
        'Glucose': [140], # High
        'Body_Temp': [98.6]
    }
    df_sample = pd.DataFrame(sample_data)
    X_sample_scaled = scaler.transform(df_sample[feature_names])
    
    # Run SHAP
    print("Initializing TreeExplainer...")
    explainer = shap.TreeExplainer(model)
    print("Computing SHAP values...")
    shap_values = explainer.shap_values(X_sample_scaled)
    
    if isinstance(shap_values, list): # Binary classification list output
        shap_vals_arr = shap_values[1]
    else:
        shap_vals_arr = shap_values
        
    print(f"SHAP calculation successful. Values shape: {np.array(shap_vals_arr).shape}")
    print(f"Sample prediction: {model.predict_proba(X_sample_scaled)[0][1]:.2%}")
    
    # Attempt plot
    plt.figure()
    shap.summary_plot(shap_vals_arr, X_sample_scaled, feature_names=feature_names, show=False)
    plt.close()
    print("✓ SHAP Visualization engine is working correctly.")

if __name__ == "__main__":
    verify_shap()
