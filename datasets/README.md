# Datasets Folder

## Overview
This folder contains documentation and references for all datasets used in the Illness Prediction project.

## Active Dataset

### augmented_medical_data.csv
- **Location:** Root project folder (`../augmented_medical_data.csv`)
- **Records:** 1,500 patient medical records
- **Features:** Age, Gender, BMI, Systolic Blood Pressure, Glucose, Body Temperature
- **Target:** Risk_Label (Low Risk / High Risk)
- **Status:** ✅ **ACTIVE** - Currently used for model training and testing

## Dataset Statistics Summary

| Metric | Value |
|--------|-------|
| Total Records | 1,500 |
| Training Records | 1,200 (80%) |
| Test Records | 300 (20%) |
| Input Features | 6 |
| Target Classes | 2 (Low Risk, High Risk) |
| Missing Values | 0 |
| Class Balance | 65% Low Risk, 35% High Risk |

## Dataset Files

### Training Files
1. **augmented_medical_data.csv** (Main Dataset)
   - Complete dataset with all 1,500 records
   - Used for model training and evaluation
   - Location: Root folder

### Model Artifacts (Generated from Dataset)
Generated in `models/` folder after training:
- `illness_risk_model.keras` - Trained model
- `scaler.pkl` - Feature scaler
- `label_encoder.pkl` - Label encoder
- `feature_names.pkl` - Feature column order
- `label_mapping.json` - Label mapping reference
- `training_history.json` - Training metrics
- `training_plots.png` - Training visualization

### Quality Check
- `data_quality_check.py` - Script to verify dataset integrity
- Output: Model performance metrics on full dataset

## How Dataset is Used

### 1. **Training Phase** (`src/model/train.py`)
```
Raw Data (augmented_medical_data.csv)
    ↓
Preprocessing (Gender encoding, scaling)
    ↓
Train-Test Split (80-20, stratified)
    ↓
Model Training (100 epochs)
    ↓
Artifacts Saved (models/ folder)
```

### 2. **Inference Phase** (`src/app/app.py`)
```
User Input (Age, Gender, BMI, BP, Glucose, Temp)
    ↓
Same Preprocessing (using saved scaler)
    ↓
Model Prediction
    ↓
SHAP Explanation (using 50-sample background)
    ↓
Result with Visualization
```

### 3. **Quality Verification** (`data_quality_check.py`)
```
Full Dataset
    ↓
Load Trained Model
    ↓
Compute Predictions
    ↓
Calculate Metrics (Accuracy, Precision, Recall, AUC)
    ↓
Generate Report
```

## Dataset Characteristics

### Feature Ranges
- **Age:** 8 - 79 years
- **Gender:** Male, Female, Transgender
- **BMI:** 17.8 - 36.9 kg/m²
- **Systolic BP:** 84 - 161 mmHg
- **Glucose:** 81 - 154 mg/dL
- **Body Temp:** 95.99 - 99.12 °F

### Label Distribution
- **Low Risk:** 980 records (65.3%)
- **High Risk:** 520 records (34.7%)

### Data Quality
- ✅ No missing values
- ✅ No duplicate records
- ✅ No outliers removed
- ✅ Consistent formatting
- ✅ All numeric values valid
- ✅ Gender values: Male, Female, Transgender only

## Model Performance on Dataset

### Test Set (300 records)
- Accuracy: **74.67%**
- Precision: **61.11%**
- Recall: **74.04%**
- F1-Score: **66.96%**
- ROC AUC: **0.8356**

### Full Dataset (1,500 records)
- Accuracy: **82.20%**
- Precision: **89.22%**
- Recall: **82.76%**
- F1-Score: **85.87%**
- ROC AUC: **0.9140**

## Scripts for Dataset Operations

### View Dataset
```bash
python -c "import pandas as pd; df = pd.read_csv('augmented_medical_data.csv'); print(df.info()); print(df.describe())"
```

### Check Dataset Quality
```bash
python data_quality_check.py
```

### Retrain Model with Dataset
```bash
python src/model/train.py
```

### Test Model with Dataset
```bash
python src/app/app.py
# Then test with sample inputs
```

## Dataset Splitting

### Method
- **Train-Test Split:** 80-20
- **Stratification:** Yes (maintains class ratio)
- **Random State:** 42 (reproducible)

### Resulting Splits
```
Full Dataset (1,500)
├── Training (1,200) - 784 Low Risk, 416 High Risk
└── Testing (300) - 196 Low Risk, 104 High Risk
```

### Further Validation (during training)
```
Training Set (1,200)
├── Training (960) - 80%
└── Validation (240) - 20%

Test Set (300) - held out
```

## Data Preprocessing Applied

### Gender Encoding
```python
{'Male': 0, 'Female': 1, 'Transgender': 2}
```

### Feature Scaling
```python
StandardScaler (zero mean, unit variance)
Fitted on training data
Applied to all features
```

### Label Encoding
```python
{'Low Risk': 0, 'High Risk': 1}
```

## Using This Dataset

### To Load and Explore
```python
import pandas as pd

df = pd.read_csv('../augmented_medical_data.csv')
print(df.shape)              # (1500, 7)
print(df.head())             # First 5 rows
print(df.describe())         # Statistics
print(df.value_counts()['Risk_Label'])  # Class distribution
```

### To Retrain Model
```python
import os
os.chdir('../')  # Go to project root
os.system('python src/model/train.py')
```

### To Get Model Predictions
```python
# Use the Flask app to get predictions with SHAP explanations
# python src/app/app.py
# Then navigate to http://127.0.0.1:5000/
```

## Dataset Version History

### Version 1.0 (Current)
- **Date:** February 5, 2026
- **Records:** 1,500
- **Status:** ✅ Production
- **Model Accuracy:** 82.20%
- **Changes:** Initial dataset creation, full training pipeline

## Important Notes

1. **Reproducibility:** Always use random_state=42 for train-test split
2. **Scaling:** Use the saved scaler for inference (don't refit)
3. **Feature Order:** Must maintain order: Age, Gender, BMI, Systolic_BP, Glucose, Body_Temp
4. **Gender Mapping:** Keep consistent encoding (Male=0, Female=1, Transgender=2)
5. **Label Mapping:** Keep consistent (Low Risk=0, High Risk=1)

## Contact

For dataset-related questions, refer to:
- `DATASET_INFO.md` - Detailed dataset information (root folder)
- `src/model/train.py` - Training code and data handling
- `data_quality_check.py` - Data validation script

---
**Created:** February 5, 2026  
**Last Updated:** February 5, 2026  
**Project:** Illness Prediction using SHAP
