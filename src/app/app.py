from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import tensorflow as tf
import shap
import joblib
import os

app = Flask(__name__)

# Paths
MODEL_DIR = '../../models'
MODEL_PATH = os.path.join(MODEL_DIR, 'illness_risk_model.keras')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
ENCODER_PATH = os.path.join(MODEL_DIR, 'label_encoder.pkl')
FEATURE_NAMES_PATH = os.path.join(MODEL_DIR, 'feature_names.pkl')

# Global variables to hold loaded artifacts
model = None
scaler = None
label_encoder = None
feature_names = None
explainer = None

def load_artifacts():
    global model, scaler, label_encoder, feature_names, explainer
    try:
        if os.path.exists(MODEL_PATH):
            model = tf.keras.models.load_model(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            label_encoder = joblib.load(ENCODER_PATH)
            feature_names = joblib.load(FEATURE_NAMES_PATH)
            
            # Initialize SHAP explainer (using a background dataset for reference ideally, 
            # here we might use a zero baseline or mean if data isn't available in memory)
            # For DeepExplainer, we need some background samples. 
            # We'll handle this dynamically or load a small subset if possible.
            print("Model and artifacts loaded successfully.")
        else:
            print("Model artifacts not found. Please run src/model/train.py first.")
    except Exception as e:
        print(f"Error loading artifacts: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return render_template('index.html', error='Model not loaded. Please train the model first.')

    try:
        # Use request.form for standard HTML form submission
        data = request.form
        
        # Prepare input dataframe
        # Expected keys: Age, Gender, BMI, Systolic_BP, Glucose, Body_Temp
        input_data = {
            'Age': [float(data['Age'])],
            'Gender': [data['Gender']], # 'Male', 'Female', 'Transgender'
            'BMI': [float(data['BMI'])],
            'Systolic_BP': [float(data['Systolic_BP'])],
            'Glucose': [float(data['Glucose'])],
            'Body_Temp': [float(data['Body_Temp'])]
        }
        
        df = pd.DataFrame(input_data)
        
        # Preprocessing
        df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1, 'Transgender': 2})
        
        # Ensure correct order of columns matches training
        df = df[feature_names]
        
        # Scale
        df_scaled = scaler.transform(df)
        
        # Predict
        prediction_prob = model.predict(df_scaled)[0][0]
        prediction_class = 1 if prediction_prob > 0.5 else 0
        risk_label = label_encoder.inverse_transform([prediction_class])[0]
        
        print(f"\n[PREDICTION] Input: {input_data}")
        print(f"[PREDICTION] Result: {risk_label} ({prediction_prob:.4f})")
        
        reasons = []
        if float(data['Systolic_BP']) > 140: reasons.append("High Blood Pressure")
        if float(data['Glucose']) > 140: reasons.append("High Glucose Levels")
        if float(data['Body_Temp']) > 100.4: reasons.append("Fever Detected")
        if float(data['BMI']) > 30: reasons.append("Obesity Range BMI")
        
        explanation_text = "Patient shows signs of " + ", ".join(reasons) if reasons else "Vital signs are within normal ranges."
        
        confidence_val = f"{prediction_prob if prediction_class == 1 else 1-prediction_prob:.2%}"
        
        return render_template('result.html', 
                             risk_level=risk_label,
                             confidence=confidence_val,
                             explanation=explanation_text)
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    load_artifacts()
    # Print a clear message that the server is ready
    print("\n" + "="*50)
    print("SERVER RUNNING! Open this URL in your browser:")
    print("http://127.0.0.1:5000/")
    print("="*50 + "\n")
    app.run(debug=True, port=5000)
