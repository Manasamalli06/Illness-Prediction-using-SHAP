import urllib.request
import urllib.parse
import json

url = 'http://127.0.0.1:5000/predict'
data = urllib.parse.urlencode({
    'Age': 45,
    'Gender': 'Male',
    'BMI': 25.0,
    'Systolic_BP': 120.0,
    'Glucose': 90.0,
    'Body_Temp': 98.6
}).encode('utf-8')

try:
    req = urllib.request.Request(url, data=data)
    response = urllib.request.urlopen(req)
    html = response.read().decode('utf-8')
    if "SHAP visualization could not be generated" in html:
        print("FAIL: SHAP not generated")
    else:
        print("SUCCESS: SHAP generated")
    if "RuntimeError" in html:
        print("RuntimeError found in HTML.")
except Exception as e:
    print("Error:", e)
