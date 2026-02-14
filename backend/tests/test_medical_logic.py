import sys
import os
import pandas as pd
import numpy as np

# Add backend/app to path
sys.path.append(os.path.join(os.getcwd(), 'backend', 'app'))

try:
    from app import extract_key_drivers, generate_clinical_interpretation, CLINICAL_RANGES
    print("✓ Successfully imported app functions")
except ImportError as e:
    print(f"✗ Failed to import: {e}")
    sys.exit(1)

def test_key_drivers_low_values():
    print("\n--- Testing Key Drivers with Low Values ---")
    
    # Mock SHAP values (all zeros for simplicity, sign shouldn't matter for out-of-range)
    feature_names = ['Age', 'Gender', 'BMI', 'Systolic_BP', 'Glucose', 'Body_Temp']
    shap_values = [0] * len(feature_names) 
    
    test_cases = [
        {'feature': 'Body_Temp', 'value': 19.8, 'desc': 'Critical Hypothermia'},
        {'feature': 'Glucose', 'value': 17.0, 'desc': 'Severe Hypoglycemia'},
        {'feature': 'Systolic_BP', 'value': 18.0, 'desc': 'Circulatory Shock'}
    ]
    
    for case in test_cases:
        original_features = {case['feature']: case['value']}
        # Set all other features to normal
        for f in feature_names:
            if f not in original_features:
                if f in CLINICAL_RANGES:
                    original_features[f] = CLINICAL_RANGES[f]['min'] + 1
                else:
                    original_features[f] = 30 # default
        
        drivers = extract_key_drivers(shap_values, feature_names, original_features, top_n=6)
        
        # Find the driver for our target feature
        target_driver = next((d for d in drivers if d['name'] == case['feature']), None)
        
        if target_driver:
            print(f"Feature: {case['feature']}, Value: {case['value']} ({case['desc']})")
            print(f"  Impact: {target_driver['impact']}")
            if target_driver['impact'] == "raises risk":
                print("  ✓ CORRECT: Labeled as 'raises risk'")
            else:
                print(f"  ✗ WRONG: Labeled as '{target_driver['impact']}'")
        else:
            print(f"  ✗ Feature {case['feature']} not found in drivers")

def test_clinical_interpretation():
    print("\n--- Testing Clinical Interpretation ---")
    
    # Low values case
    line1, line2 = generate_clinical_interpretation(
        age=45, gender_encoded=0, bmi=22.0, 
        systolic_bp=18.0, glucose=17.0, body_temp=19.8, 
        risk_class=1
    )
    
    print(f"Interpretation Line 1: {line1}")
    print(f"Interpretation Line 2: {line2}")
    
    keywords = ['critical', 'shock', 'hypoglycemia', 'hypothermia', 'emergency']
    found = [k for k in keywords if k.lower() in line1.lower() or k.lower() in line2.lower()]
    
    if len(found) >= 3:
        print(f"  ✓ Found critical keywords: {found}")
    else:
        print(f"  ✗ Missing some critical keywords. Found: {found}")

if __name__ == "__main__":
    test_key_drivers_low_values()
    test_clinical_interpretation()
