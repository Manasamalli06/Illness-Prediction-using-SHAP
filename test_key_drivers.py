import requests
import json

# Test data for High Risk prediction
test_data_high_risk = {
    'Age': '70',
    'Gender': 'Male',
    'BMI': '150',
    'Systolic_BP': '180',
    'Glucose': '180',
    'Body_Temp': '99.5'
}

# Test data for Low Risk prediction
test_data_low_risk = {
    'Age': '30',
    'Gender': 'Female',
    'BMI': '22',
    'Systolic_BP': '110',
    'Glucose': '95',
    'Body_Temp': '98.6'
}

def test_prediction(test_data, risk_type):
    print(f"\n{'='*70}")
    print(f"Testing {risk_type} Prediction")
    print(f"{'='*70}")
    
    try:
        # Make POST request to predict endpoint
        response = requests.post(
            'http://127.0.0.1:5000/predict',
            data=test_data,
            timeout=30
        )
        
        if response.status_code == 200:
            print(f"✓ Request successful (Status: {response.status_code})")
            print(f"✓ HTML response received ({len(response.text)} bytes)")
            
            # Check if key drivers box is present in response
            if 'Key Drivers' in response.text and 'key-drivers-box' in response.text:
                print("✓ Key Drivers box found in HTML response")
                print("✓ Key drivers feature is working!")
            else:
                print("⚠ Key Drivers box not found in HTML response")
            
            # Check for other expected elements
            if 'Analysis Result' in response.text:
                print("✓ Analysis Result section found")
            if 'Clinical Interpretation' in response.text:
                print("✓ Clinical Interpretation section found")
            if 'SHAP' in response.text:
                print("✓ SHAP analysis section found")
                
        else:
            print(f"✗ Request failed (Status: {response.status_code})")
            print(f"Response: {response.text[:500]}")
    
    except Exception as e:
        print(f"✗ Error: {e}")

if __name__ == "__main__":
    print("\n" + "="*70)
    print("TESTING KEY DRIVERS FEATURE")
    print("="*70)
    
    test_prediction(test_data_high_risk, "HIGH RISK")
    test_prediction(test_data_low_risk, "LOW RISK")
    
    print("\n" + "="*70)
    print("Testing complete!")
    print("="*70)
