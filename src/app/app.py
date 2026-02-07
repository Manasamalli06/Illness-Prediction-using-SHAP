from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib
import base64
from io import BytesIO
import warnings
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import json
import shap

warnings.filterwarnings('ignore')
matplotlib.use('Agg')  # Use non-interactive backend

app = Flask(__name__)

# --- Path Logic (Robust and Absolute) ---
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(APP_DIR, '../../'))
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
DATA_FILE = os.path.join(PROJECT_ROOT, 'augmented_medical_data.csv')

MODEL_PATH = os.path.join(MODEL_DIR, 'illness_risk_model.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
ENCODER_PATH = os.path.join(MODEL_DIR, 'label_encoder.pkl')
FEATURE_NAMES_PATH = os.path.join(MODEL_DIR, 'feature_names.pkl')

# Global variables to hold loaded artifacts
model = None
scaler = None
label_encoder = None
feature_names = None
explainer = None
background_data = None

def load_artifacts():
    global model, scaler, label_encoder, feature_names, explainer, background_data
    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            label_encoder = joblib.load(ENCODER_PATH)
            feature_names = joblib.load(FEATURE_NAMES_PATH)

            # SHAP explainer will be initialized lazily on first prediction for faster startup
            print("XGBoost Model (Optimized) and artifacts loaded successfully.")
            print("SHAP TreeExplainer will be initialized on first prediction.")
        else:
            print("Model artifacts not found. Please run src/model/train.py first.")
    except Exception as e:
        print(f"Error loading artifacts: {e}")


def initialize_shap_explainer():
    """Initialize SHAP explainer lazily on first prediction"""
    global explainer, background_data

    if explainer is not None:
        return  # Already initialized

    try:
        print("[SHAP] Initializing SHAP explainer...")
        if os.path.exists(DATA_FILE):
            df = pd.read_csv(DATA_FILE)
            # Preprocess background data
            if 'Gender' in df.columns and df['Gender'].dtype == 'object':
                df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1, 'Transgender': 2})
            X_bg = df[feature_names].values
            X_bg_scaled = scaler.transform(X_bg)
            # Use smaller sample for faster SHAP computation (50 samples)
            background_data = X_bg_scaled[:min(50, len(X_bg_scaled))]

            # Initialize SHAP TreeExplainer for XGBoost
            explainer = shap.TreeExplainer(model)
            print("[SHAP] SHAP TreeExplainer initialized successfully.")
        else:
            print("[SHAP] Background data not found for SHAP explainer.")
    except Exception as e:
        print(f"[SHAP] Error initializing SHAP explainer: {e}")
        import traceback
        traceback.print_exc()

def generate_clinical_interpretation(age, gender_encoded, bmi, systolic_bp, glucose, body_temp, risk_class):
    """
    Generate feature-driven, patient-specific clinical interpretation.
    Analyzes exact input features and creates unique 2-line interpretation
    that differs based on abnormal patterns and interactions.
    
    Args:
        age: Patient age in years
        gender_encoded: 0=Male, 1=Female, 2=Transgender
        bmi: Body Mass Index
        systolic_bp: Systolic blood pressure
        glucose: Fasting glucose level
        body_temp: Body temperature in Fahrenheit
        risk_class: 0=Low Risk, 1=High Risk
    
    Returns:
        tuple: (line1, line2) - Two professional clinical lines
    """

    # Gender mapping for reference
    gender_map = {0: 'Male', 1: 'Female', 2: 'Transgender'}
    gender = gender_map.get(int(gender_encoded), 'Unknown')

    # Define clinical thresholds
    BP_ELEVATED = 130
    BP_HIGH_STAGE1 = 140
    BP_HIGH_STAGE2 = 160

    GLUCOSE_PREDIABETIC = 100
    GLUCOSE_DIABETIC = 126
    GLUCOSE_CRITICAL = 200

    TEMP_LOWGRADE_FEVER = 99.5
    TEMP_FEVER = 100.4
    TEMP_HIGH_FEVER = 102
    TEMP_CRITICAL = 103.5

    BMI_OVERWEIGHT = 25
    BMI_OBESE = 30
    BMI_OBESE_SEVERE = 35

    # Analyze feature deviations and severity
    features_analysis = {}

    # Blood Pressure Analysis
    if systolic_bp >= BP_HIGH_STAGE2:
        features_analysis['bp_severity'] = 'critical'
        features_analysis['bp_desc'] = f"Stage 2 hypertension ({systolic_bp} mmHg)"
    elif systolic_bp >= BP_HIGH_STAGE1:
        features_analysis['bp_severity'] = 'high'
        features_analysis['bp_desc'] = f"Stage 1 hypertension ({systolic_bp} mmHg)"
    elif systolic_bp >= BP_ELEVATED:
        features_analysis['bp_severity'] = 'elevated'
        features_analysis['bp_desc'] = f"elevated blood pressure ({systolic_bp} mmHg)"
    else:
        features_analysis['bp_severity'] = 'normal'
        features_analysis['bp_desc'] = None

    # Glucose Analysis
    if glucose >= GLUCOSE_CRITICAL:
        features_analysis['glucose_severity'] = 'critical'
        features_analysis['glucose_desc'] = f"critically elevated glucose ({glucose} mg/dL)"
    elif glucose >= GLUCOSE_DIABETIC:
        features_analysis['glucose_severity'] = 'high'
        features_analysis['glucose_desc'] = f"elevated fasting glucose ({glucose} mg/dL) consistent with diabetes"
    elif glucose >= GLUCOSE_PREDIABETIC:
        features_analysis['glucose_severity'] = 'moderate'
        features_analysis['glucose_desc'] = f"elevated glucose ({glucose} mg/dL) in prediabetic range"
    else:
        features_analysis['glucose_severity'] = 'normal'
        features_analysis['glucose_desc'] = None

    # Temperature Analysis
    if body_temp >= TEMP_CRITICAL:
        features_analysis['temp_severity'] = 'critical'
        features_analysis['temp_desc'] = f"critical hyperthermia ({body_temp}°F)"
    elif body_temp >= TEMP_HIGH_FEVER:
        features_analysis['temp_severity'] = 'high'
        features_analysis['temp_desc'] = f"significant fever ({body_temp}°F), suggesting acute infection"
    elif body_temp >= TEMP_FEVER:
        features_analysis['temp_severity'] = 'moderate'
        features_analysis['temp_desc'] = f"fever ({body_temp}°F) present"
    elif body_temp >= TEMP_LOWGRADE_FEVER:
        features_analysis['temp_severity'] = 'mild'
        features_analysis['temp_desc'] = f"low-grade fever ({body_temp}°F)"
    else:
        features_analysis['temp_severity'] = 'normal'
        features_analysis['temp_desc'] = None

    # BMI Analysis
    if bmi >= BMI_OBESE_SEVERE:
        features_analysis['bmi_severity'] = 'severe'
        features_analysis['bmi_desc'] = f"severe obesity (BMI {bmi:.1f})"
    elif bmi >= BMI_OBESE:
        features_analysis['bmi_severity'] = 'high'
        features_analysis['bmi_desc'] = f"obesity (BMI {bmi:.1f})"
    elif bmi >= BMI_OVERWEIGHT:
        features_analysis['bmi_severity'] = 'mild'
        features_analysis['bmi_desc'] = f"overweight (BMI {bmi:.1f})"
    else:
        features_analysis['bmi_severity'] = 'normal'
        features_analysis['bmi_desc'] = None

    # Age Risk Assessment
    age_risk = "advanced age" if age >= 65 else "mid-to-older age" if age >= 55 else "middle age" if age >= 40 else "younger age"
    is_elderly = age >= 65

    # Identify most critical abnormalities
    severity_ranking = []
    if features_analysis['bp_severity'] != 'normal':
        severity_ranking.append(('bp', features_analysis['bp_severity'], features_analysis['bp_desc']))
    if features_analysis['glucose_severity'] != 'normal':
        severity_ranking.append(('glucose', features_analysis['glucose_severity'], features_analysis['glucose_desc']))
    if features_analysis['temp_severity'] != 'normal':
        severity_ranking.append(('temp', features_analysis['temp_severity'], features_analysis['temp_desc']))
    if features_analysis['bmi_severity'] != 'normal':
        severity_ranking.append(('bmi', features_analysis['bmi_severity'], features_analysis['bmi_desc']))

    # Sort by severity priority (critical > high > moderate > mild)
    severity_order = {'critical': 0, 'high': 1, 'moderate': 2, 'mild': 3}
    severity_ranking.sort(key=lambda x: severity_order.get(x[1], 4))

    # ===================== INTERPRETATION GENERATION =====================

    line1 = ""
    line2 = ""

    if risk_class == 1:  # HIGH RISK

        # Case 1: Multiple critical/high abnormalities
        if len(severity_ranking) >= 2:
            top_abnormalities = [sr for sr in severity_ranking if sr[1] in ['critical', 'high']]

            if len(top_abnormalities) >= 2:
                # Two or more critical/high abnormalities
                abn_list = []
                if any(sr[0] == 'bp' for sr in top_abnormalities):
                    abn_list.append("significant hypertension")
                if any(sr[0] == 'glucose' for sr in top_abnormalities):
                    abn_list.append("hyperglycemia")
                if any(sr[0] == 'temp' for sr in top_abnormalities):
                    abn_list.append("fever")

                if len(abn_list) >= 2:
                    line1 = f"Patient demonstrates concurrent {' and '.join(abn_list)} with {'comorbid obesity' if features_analysis['bmi_severity'] in ['high', 'severe'] else 'elevated metabolic risk'}."

                    if features_analysis['temp_severity'] in ['high', 'critical'] and features_analysis['glucose_severity'] in ['high', 'critical']:
                        line2 = f"Combined thermoregulatory disruption and marked glucose dysregulation indicate acute systemic inflammatory response; immediate clinical evaluation warranted."
                    elif features_analysis['bp_severity'] in ['critical'] and features_analysis['glucose_severity'] in ['high', 'critical']:
                        line2 = f"Severe hypertension with uncontrolled glycemia presents substantial cardiovascular and metabolic compromise; urgent intervention necessary."
                    else:
                        line2 = f"Cumulative physiologic derangements suggest multisystem involvement requiring acute assessment and stabilization."

        # Case 2: Single dominant abnormality with age as modifier
        elif len(severity_ranking) == 1:
            abn_type, abn_severity, abn_desc = severity_ranking[0]

            if abn_type == 'bp' and abn_severity in ['high', 'critical']:
                if is_elderly:
                    line1 = f"Patient in {age_risk} with {abn_desc}, significantly amplifying cerebrovascular and cardiac event risk."
                else:
                    line1 = f"Marked elevation in systolic pressure ({systolic_bp} mmHg) despite {age_risk}, indicating resistant or secondary hypertensive process."

                if glucose >= GLUCOSE_PREDIABETIC:
                    line2 = f"Concurrent glucose dysregulation ({glucose} mg/dL) compounds vascular injury risk; metabolic syndrome phenotype evident."
                else:
                    line2 = f"Hypertensive crisis magnitude warrants expedited evaluation for end-organ damage and medication optimization."

            elif abn_type == 'glucose' and abn_severity in ['high', 'critical']:
                if systolic_bp >= BP_ELEVATED:
                    line1 = f"Poorly controlled diabetes ({glucose} mg/dL) with {features_analysis['bp_desc'].lower()} indicates accelerated diabetic microvascular disease."
                    line2 = f"Combined glycemic and blood pressure dyscontrol presents elevated risk for renal and retinal complications; therapy intensification indicated."
                else:
                    line1 = f"Markedly elevated fasting glucose ({glucose} mg/dL) in {age_risk} patient reflects severe beta-cell dysfunction or undiagnosed comorbidity."
                    line2 = f"Glycemic derangement of this magnitude necessitates urgent endocrinologic assessment and pharmacologic intervention."

            elif abn_type == 'temp' and abn_severity in ['high', 'critical']:
                line1 = f"High fever state ({body_temp}°F) with {'marked' if body_temp >= 102 else 'significant'} elevation suggests acute infectious or inflammatory process."
                if bmi >= BMI_OBESE or age >= 60:
                    line2 = f"Combined fever with {'obesity and ' if bmi >= BMI_OBESE else ''}{'advanced age' if age >= 60 else ''} substantially increases risk for complicated infection; comprehensive evaluation essential."
                else:
                    line2 = f"Fever magnitude and duration assessment critical to differentiate benign viral syndrome from serious bacterial infection."

            elif abn_type == 'bmi' and abn_severity in ['high', 'severe']:
                line1 = f"Patient exhibits {abn_desc} with documented {'hypertension' if features_analysis['bp_severity'] != 'normal' else 'elevated cardiovascular risk markers'}."
                line2 = f"Obesity-related metabolic dysfunction promotes insulin resistance and inflammatory cascade activation; structured weight reduction and metabolic monitoring imperative."

        # Case 3: No significant single abnormality but classified as high risk (model-detected pattern)
        else:
            line1 = f"Patient risk profile identified through integrated feature pattern analysis; baseline characteristics suggest enhanced disease susceptibility."
            line2 = f"Even with individually borderline findings, cumulative physiologic state warrants heightened clinical vigilance and preventative intervention optimization."

    else:  # LOW RISK (risk_class == 0)

        if len(severity_ranking) == 0:
            line1 = f"Vital signs and metabolic parameters within normal reference ranges; patient demonstrates favorable health profile."
            line2 = f"No acute illness indicators identified; maintenance of current preventative health practices recommended."

        elif len(severity_ranking) == 1:
            abn_type, abn_severity, abn_desc = severity_ranking[0]

            if abn_type == 'bmi' and abn_severity == 'mild':
                line1 = f"Patient in {age_risk} with {abn_desc}; otherwise metabolically and hemodynamically stable."
                line2 = f"Modest weight optimization may enhance long-term cardiovascular outcomes; current vital stability permits conservative management approach."

            elif abn_type == 'glucose' and abn_severity == 'moderate':
                line1 = f"Fasting glucose {glucose} mg/dL indicates prediabetic state within low-risk disease manifestation category."
                line2 = f"Lifestyle modification including dietary carbohydrate restriction and increased physical activity recommended for glycemic control optimization."

            elif abn_type == 'temp' and abn_severity == 'mild':
                line1 = f"Low-grade temperature elevation ({body_temp}°F) present but hemodynamically and metabolically stable."
                line2 = f"Likely self-limited process; supportive care and serial monitoring for symptom progression advisable."

            else:
                line1 = f"Patient demonstrates one mild parameter deviation; overall physiologic compensation remains intact."
                line2 = f"Continued monitoring recommended; escalation of care not presently indicated."

        else:
            line1 = f"Multiple mild abnormalities present but collectively indicate preserved physiologic resilience in {age_risk} patient."
            line2 = f"Low-risk categorization supported by absence of critical findings; standard preventative health screening intervals appropriate."

    # Ensure lines are not empty
    if not line1:
        line1 = "Clinical assessment indicates patient requires medical evaluation based on identified physiologic parameters."
    if not line2:
        line2 = "Continued monitoring and appropriate specialist referral recommended based on individual clinical context."

    return line1, line2


def extract_key_drivers(shap_values, feature_names, original_features, top_n=3):
    """
    Extract top key drivers (features) that influence the prediction.
    
    Args:
        shap_values: SHAP values for the sample
        feature_names: List of feature names
        original_features: Dictionary with original feature values
        top_n: Number of top drivers to return
    
    Returns:
        List of dicts with driver information
    """
    try:
        # Handle list of SHAP values (binary classification)
        if isinstance(shap_values, list):
            shap_vals = np.array(shap_values[1]).flatten()  # High Risk class
        else:
            shap_vals = np.array(shap_values).flatten()

        # Get absolute SHAP values
        abs_shap = np.abs(shap_vals)

        # Get top N indices
        top_indices = np.argsort(abs_shap)[::-1][:top_n]

        drivers = []
        for idx in top_indices:
            feature_name = list(feature_names)[idx]
            shap_val = shap_vals[idx]
            feature_val = original_features.get(feature_name, 'N/A')

            # Determine impact direction
            if shap_val > 0:
                impact = "raises risk"
                impact_color = "red"
            else:
                impact = "lowers risk"
                impact_color = "green"

            # For display purposes
            display_val = feature_val
            if feature_name == 'Gender':
                gender_map = {0: 'Male', 1: 'Female', 2: 'Transgender'}
                display_val = gender_map.get(int(feature_val), 'Unknown')
            elif isinstance(display_val, float):
                display_val = f"{display_val:.2f}"

            drivers.append({
                'name': feature_name,
                'value': display_val,
                'shap_contribution': float(abs_shap[idx]),
                'impact': impact,
                'impact_color': impact_color
            })

        return drivers
    except Exception as e:
        print(f"Error extracting key drivers: {e}")
        import traceback
        traceback.print_exc()
        return []


def generate_shap_plot(X_sample, predicted_prob, risk_label):
    """
    Generate SHAP feature importance visualization as PNG and return base64 encoded string.
    
    Args:
        X_sample: Scaled input features (1 x n_features)
        predicted_prob: Model prediction probability for High Risk
        risk_label: 'High Risk' or 'Low Risk' prediction string
    
    Returns:
        base64 string of the plot PNG or None if generation fails
    """
    try:
        if explainer is None or background_data is None:
            print("SHAP explainer not initialized.")
            return None

        # Compute SHAP values for this sample
        print(f"[SHAP] Computing SHAP values for prediction: {risk_label} ({predicted_prob:.2%})")
        # TreeExplainer usually returns a single array for binary classification
        shap_values = explainer.shap_values(X_sample)

        print(f"[SHAP] SHAP values type: {type(shap_values)}, is list: {isinstance(shap_values, list)}")

        # For binary classification, SHAP returns [class0_values, class1_values]
        # We want class 1 = High Risk
        if isinstance(shap_values, list):
            # shap_values is a list [low_risk_shap, high_risk_shap]
            shap_vals = shap_values[1]  # High Risk class SHAP values
        else:
            shap_vals = shap_values

        print(f"[SHAP] SHAP values shape: {np.array(shap_vals).shape}")

        # Extract single sample SHAP values
        if len(shap_vals.shape) > 1:
            shap_sample = shap_vals[0]  # First sample
        else:
            shap_sample = shap_vals

        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=(11, 6))

        # Get the feature importance (SHAP values - keep sign for direction)
        shap_vals_sample = np.array(shap_sample).flatten()
        feature_names_list = list(feature_names)

        # Sort by absolute importance
        abs_shap = np.abs(shap_vals_sample)
        sorted_idx = np.argsort(abs_shap)[::-1]

        # Create colors: Red for positive SHAP (increases High Risk), Blue for negative (decreases High Risk)
        colors_list = ['#EF553B' if shap_vals_sample[i] > 0 else '#636EFA' for i in sorted_idx]

        # Create horizontal bar plot
        y_pos = np.arange(len(sorted_idx))
        bar_values = shap_vals_sample[sorted_idx]

        ax.barh(y_pos, bar_values, color=colors_list, edgecolor='black', linewidth=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([feature_names_list[i] for i in sorted_idx], fontsize=10)
        ax.set_xlabel('SHAP Value Contribution', fontsize=11, fontweight='bold')
        ax.set_title(f'Feature Impact on Prediction: {risk_label}\n(Red→increases risk, Blue→decreases risk)',
                     fontsize=12, fontweight='bold', color='#2c3e50')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax.grid(axis='x', alpha=0.3, linestyle='--')

        # Add value labels on bars
        for i, (idx, val) in enumerate(zip(sorted_idx, bar_values)):
            label_x = val + (0.02 if val > 0 else -0.02)
            ax.text(label_x, i, f'{val:.3f}', va='center',
                   ha='left' if val > 0 else 'right', fontsize=8)

        plt.tight_layout()

        # Convert plot to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode()
        plt.close()

        print("[SHAP] SHAP plot generated successfully")
        return image_base64

    except Exception as e:
        print(f"[SHAP] Error generating SHAP plot: {e}")
        import traceback
        traceback.print_exc()
        return None

    except Exception as e:
        print(f"Error generating SHAP plot: {e}")
        return None


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

        # Store original gender value before encoding
        original_gender = data['Gender']

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

        # Predict (XGBoost model outputs probabilities)
        prediction_prob = model.predict_proba(df_scaled)[0][1]

        # The model was trained with 0='Low Risk', 1='High Risk'
        # So prediction_prob is P(High Risk)
        prob_high = prediction_prob

        # Final predicted class based on probability of High Risk
        # NOTE: If prob_high > 0.5, prediction is HIGH RISK, so final_class should be 1
        final_class = 1 if prob_high > 0.5 else 0
        risk_label = 'High Risk' if final_class == 1 else 'Low Risk'

        print(f"\n[PREDICTION] Input: {input_data}")
        print(f"[PREDICTION] Raw model prob(High Risk): {prob_high:.4f}, Label: {risk_label}")

        # Generate feature-driven clinical interpretation
        gender_code = df['Gender'].values[0]
        line1, line2 = generate_clinical_interpretation(
            age=float(data['Age']),
            gender_encoded=gender_code,
            bmi=float(data['BMI']),
            systolic_bp=float(data['Systolic_BP']),
            glucose=float(data['Glucose']),
            body_temp=float(data['Body_Temp']),
            risk_class=final_class
        )

        # Combine lines into final explanation
        explanation_text = f"{line1} {line2}"

        # Generate specific reason for the risk level
        if risk_label == 'High Risk':
            # Check specific conditions for high risk
            reasons = []
            if float(data['Body_Temp']) >= 100.4:
                reasons.append("high fever")
            if float(data['Systolic_BP']) >= 140:
                reasons.append("uncontrolled BP")
            if float(data['Glucose']) >= 126:
                reasons.append("high glucose")
            if float(data['BMI']) >= 30:
                reasons.append("high BMI")
            reason = ", ".join(reasons) if reasons else "multiple risk factors"
        else:
            reason = "normal vital signs and parameters"

        confidence_val = f"{prob_high:.2%}"

        # Create a simple, human-friendly 2-line interpretation
        simple_line1 = f"Prediction: {risk_label} — {confidence_val} confidence."
        if 'High' in risk_label:
            simple_line2 = "This indicates a high chance of significant illness — seek medical evaluation promptly."
            note_message = "This result suggests potential health risk. Please consult your healthcare provider promptly. This tool is not a diagnosis."
        else:
            simple_line2 = "This indicates low immediate risk — continue routine care and consult your doctor if symptoms develop."
            note_message = "Overall looks OK. Maintain healthy habits; consult your doctor if you have concerns. This tool is not a diagnosis."

        simple_interpretation = [simple_line1, simple_line2]
        # confidence_val = ... (rest of logic)

        # Create feature pairs with original gender value for display
        display_feature_pairs = []
        for i, feature_name in enumerate(feature_names):
            if feature_name == 'Gender':
                display_feature_pairs.append([feature_name, original_gender])
            else:
                display_feature_pairs.append([feature_name, df[feature_name].values[0]])

        # Initialize SHAP explainer on first prediction (lazy loading)
        initialize_shap_explainer()

        # Generate SHAP visualization
        shap_plot_base64 = generate_shap_plot(df_scaled, prob_high, risk_label)

        # Generate SHAP values for key drivers extraction
        shap_values_for_drivers = None
        if explainer is not None and background_data is not None:
            try:
                shap_values_for_drivers = explainer.shap_values(df_scaled)
            except:
                pass

        # Extract key drivers
        original_features_dict = {
            'Age': float(data['Age']),
            'Gender': gender_code,
            'BMI': float(data['BMI']),
            'Systolic_BP': float(data['Systolic_BP']),
            'Glucose': float(data['Glucose']),
            'Body_Temp': float(data['Body_Temp'])
        }

        key_drivers = extract_key_drivers(
            shap_values_for_drivers,
            feature_names,
            original_features_dict,
            top_n=3
        ) if shap_values_for_drivers is not None else []

        print(f"DEBUG: Risk Label: {risk_label}, Reason: {reason}")

        return render_template('result.html',
                     risk_level=risk_label,
                     confidence=confidence_val,
                     explanation=explanation_text,
                     simple_interpretation=simple_interpretation,
                     note_message=note_message,
                     reason=reason,
                     shap_plot=shap_plot_base64,
                     feature_names=feature_names,
                     feature_values=df[feature_names].values[0].tolist(),
                     feature_pairs=display_feature_pairs,
                     key_drivers=key_drivers)

    except Exception as e:
        print(f"Error during prediction: {e}")
        return render_template('index.html', error=str(e))

@app.route('/download_report', methods=['POST'])
def download_report():
    """Generate and download a PDF report of the analysis"""
    try:
        risk_level = request.form.get('risk_level', 'Unknown')
        confidence = request.form.get('confidence', 'N/A')
        explanation = request.form.get('explanation', 'N/A')
        feature_count = int(request.form.get('feature_count', 0))

        # Reconstruct feature pairs from form data
        feature_data = []
        for i in range(feature_count):
            feature_name = request.form.get(f'feature_name_{i}', '')
            feature_value_str = request.form.get(f'feature_value_{i}', '0')

            # Handle Gender specially (it's a string, not a number)
            if feature_name == 'Gender':
                feature_value = feature_value_str
            else:
                try:
                    feature_value = float(feature_value_str)
                except ValueError:
                    feature_value = feature_value_str

            feature_data.append([feature_name, feature_value])

        # Create PDF
        pdf_buffer = BytesIO()
        doc = SimpleDocTemplate(pdf_buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
        story = []

        # Styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#008080'),
            spaceAfter=30,
            alignment=1
        )

        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#008080'),
            spaceAfter=12,
            spaceBefore=12
        )

        body_style = ParagraphStyle(
            'CustomBody',
            parent=styles['BodyText'],
            fontSize=10,
            spaceAfter=10
        )

        # Title
        story.append(Paragraph("AI HealthGuard - Analysis Report", title_style))
        story.append(Spacer(1, 0.2*inch))

        # Report metadata
        metadata = [
            ['Report Generated:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            ['System:', 'AI HealthGuard v1.0'],
        ]

        metadata_table = Table(metadata, colWidths=[2*inch, 3*inch])
        metadata_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e0f2f1')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#2c3e50')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ]))
        story.append(metadata_table)
        story.append(Spacer(1, 0.3*inch))

        # Prediction Results Section
        story.append(Paragraph("Prediction Results", heading_style))

        results_data = [
            ['Risk Level:', risk_level],
            ['Confidence Score:', confidence],
        ]

        results_table = Table(results_data, colWidths=[2*inch, 3*inch])
        risk_color = colors.HexColor('#e74c3c') if 'High' in risk_level else colors.HexColor('#27ae60')
        results_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f0f0f0')),
            ('TEXTCOLOR', (1, 0), (1, 0), risk_color),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#2c3e50')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (1, 0), (1, 0), 12),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ]))
        story.append(results_table)
        story.append(Spacer(1, 0.2*inch))

        # Clinical Interpretation
        story.append(Paragraph("Clinical Interpretation", heading_style))
        story.append(Paragraph(explanation, body_style))
        story.append(Spacer(1, 0.2*inch))

        # Patient Features
        story.append(Paragraph("Patient Input Features", heading_style))

        feature_table_data = [['Feature Name', 'Value']]
        for feature_name, feature_value in feature_data:
            # Format value based on type
            if isinstance(feature_value, str):
                formatted_value = feature_value
            else:
                formatted_value = f"{feature_value:.2f}"
            feature_table_data.append([feature_name, formatted_value])

        feature_table = Table(feature_table_data, colWidths=[2.5*inch, 2.5*inch])
        feature_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#008080')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('TOPPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f0f0f0')),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.HexColor('#2c3e50')),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
            ('TOPPADDING', (0, 1), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#cccccc')),
        ]))
        story.append(feature_table)
        story.append(Spacer(1, 0.2*inch))

        # Disclaimer
        disclaimer_style = ParagraphStyle(
            'Disclaimer',
            parent=styles['Normal'],
            fontSize=8,
            textColor=colors.HexColor('#7f8c8d'),
            alignment=0
        )
        story.append(Paragraph(
            "<b>Note:</b> This is a decision-support tool using Optimized XGBoost and SHAP for explainability. "
            "It is not a replacement for professional medical diagnosis. Always consult with healthcare professionals for medical advice.",
            disclaimer_style
        ))

        # Build PDF
        doc.build(story)
        pdf_buffer.seek(0)

        # Return PDF file
        return send_file(
            pdf_buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f"HealthGuard_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        )

    except Exception as e:
        print(f"Error generating report: {e}")
        return render_template('index.html', error=f"Error generating report: {str(e)}")

if __name__ == '__main__':
    load_artifacts()
    # Print a clear message that the server is ready
    print("\n" + "="*50)
    print("SERVER RUNNING! Open this URL in your browser:")
    print("http://127.0.0.1:5000/")
    print("="*50 + "\n")
    app.run(debug=True, port=5000)
