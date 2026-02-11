import requests
import time

# Wait for server to be ready
time.sleep(3)

# Test cases with clear patterns
test_cases = [
    {
        'name': 'HEALTHY LOW RISK',
        'data': {
            'Age': '25',
            'Gender': 'Female',
            'BMI': '22',
            'Systolic_BP': '115',
            'Glucose': '95',
            'Body_Temp': '98.6'
        },
        'expected': 'Low Risk'
    },
    {
        'name': 'HIGH RISK - ELDERLY WITH HIGH BP',
        'data': {
            'Age': '75',
            'Gender': 'Male',
            'BMI': '32',
            'Systolic_BP': '170',
            'Glucose': '140',
            'Body_Temp': '98.8'
        },
        'expected': 'High Risk'
    },
    {
        'name': 'HIGH RISK - OBESE WITH DIABETES',
        'data': {
            'Age': '55',
            'Gender': 'Female',
            'BMI': '95',
            'Systolic_BP': '155',
            'Glucose': '165',
            'Body_Temp': '98.9'
        },
        'expected': 'High Risk'
    },
    {
        'name': 'LOW RISK - YOUNG HEALTHY',
        'data': {
            'Age': '18',
            'Gender': 'Male',
            'BMI': '24',
            'Systolic_BP': '110',
            'Glucose': '92',
            'Body_Temp': '98.4'
        },
        'expected': 'Low Risk'
    },
    {
        'name': 'HIGH RISK - HIGH FEVER',
        'data': {
            'Age': '45',
            'Gender': 'Transgender',
            'BMI': '26',
            'Systolic_BP': '125',
            'Glucose': '105',
            'Body_Temp': '102.5'
        },
        'expected': 'High Risk'
    }
]

print("=" * 80)
print("TESTING MODEL PREDICTIONS FOR CORRECTNESS")
print("=" * 80)

correct = 0
total = 0

for i, test_case in enumerate(test_cases, 1):
    print(f"\nTest {i}: {test_case['name']}")
    print(f"Expected: {test_case['expected']}")
    
    try:
        response = requests.post(
            'http://127.0.0.1:5000/predict',
            data=test_case['data'],
            timeout=15
        )
        
        if response.status_code == 200:
            # Extract prediction from HTML
            if f"badge-high" in response.text and test_case['expected'] == 'High Risk':
                print(f"CORRECT: Predicted High Risk (as expected)")
                correct += 1
            elif f"badge-low" in response.text and test_case['expected'] == 'Low Risk':
                print(f"CORRECT: Predicted Low Risk (as expected)")
                correct += 1
            elif f"badge-high" in response.text:
                print(f"WRONG: Predicted High Risk (expected Low Risk)")
            else:
                print(f"WRONG: Predicted Low Risk (expected High Risk)")
            
            total += 1
            
            # Show confidence
            if 'Confidence Score' in response.text:
                print(f"  (Prediction result page loaded successfully)")
            else:
                print(f"FAILED: Request failed {response.status_code}")
    
    except Exception as e:
        print(f"FAILED: Error: {e}")

print("\n" + "=" * 80)
print(f"RESULTS: {correct}/{total} predictions correct ({100*correct/total:.1f}%)")
print("=" * 80)

if correct == total:
    print("ALL PREDICTIONS ARE CORRECT!")
elif correct >= total * 0.8:
    print("GOOD: Most predictions are correct")
else:
    print("PROBLEM: Some predictions are incorrect")
