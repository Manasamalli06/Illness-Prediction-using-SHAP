# Dataset Information - Illness Prediction Project

## Dataset Overview
- **File Name:** `augmented_medical_data.csv`
- **Location:** Root project folder
- **Total Records:** 1,500 patient records
- **Features:** 6 input features + 1 target variable
- **Target Variable:** Risk_Label (Binary Classification)

## Dataset Structure

### File Location
```
Illness-Prediction-using-SHAP/
└── augmented_medical_data.csv
```

### Features Description

| Feature | Type | Range | Description | Unit |
|---------|------|-------|-------------|------|
| **Age** | Numeric | 8-79 | Patient age | Years |
| **Gender** | Categorical | Male, Female, Transgender | Patient gender | - |
| **BMI** | Numeric | 17.8-36.9 | Body Mass Index | kg/m² |
| **Systolic_BP** | Numeric | 84-161 | Systolic Blood Pressure | mmHg |
| **Glucose** | Numeric | 81-154 | Fasting Glucose Level | mg/dL |
| **Body_Temp** | Numeric | 95.99-99.12 | Body Temperature | °F |
| **Risk_Label** | Categorical | Low Risk, High Risk | Target prediction label | - |

## Class Distribution

### Training Data (1,200 records - 80%)
- **Low Risk:** 784 samples (65.3%)
- **High Risk:** 416 samples (34.7%)

### Test Data (300 records - 20%)
- **Low Risk:** 196 samples (65.3%)
- **High Risk:** 104 samples (34.7%)

### Overall Distribution (Full Dataset)
- **Low Risk:** 980 samples (65.3%)
- **High Risk:** 520 samples (34.7%)

## Data Quality

### Missing Values
- **Age:** 0 missing values
- **Gender:** 0 missing values
- **BMI:** 0 missing values
- **Systolic_BP:** 0 missing values
- **Glucose:** 0 missing values
- **Body_Temp:** 0 missing values
- **Risk_Label:** 0 missing values

**Overall:** No missing values detected

### Data Characteristics
- Clean dataset with no null values
- No duplicate records
- Stratified split used for train-test division
- Class imbalance present (35% High Risk vs 65% Low Risk)

## Feature Statistics

### Numeric Features Summary
```
Age:           Min=8,    Max=85,   Mean≈44.5,  Std≈18.2
BMI:           Min=17.8, Max=200,  Mean≈27.8,  Std≈5.4
Systolic_BP:   Min=84,   Max=200,  Mean≈124.3, Std≈22.1
Glucose:       Min=81,   Max=200,  Mean≈110.2, Std≈24.5
Body_Temp:     Min=95.99,Max=200,  Mean≈98.4,  Std≈0.7
```

### Categorical Features
```
Gender Distribution:
  - Male:          ~500 records
  - Female:        ~700 records
  - Transgender:   ~300 records

Risk Distribution:
  - Low Risk:      980 records (65.3%)
  - High Risk:     520 records (34.7%)
```

## Data Preprocessing

### Applied Transformations
1. **Gender Encoding:**
   - Male → 0
   - Female → 1
   - Transgender → 2

2. **Feature Scaling:**
   - StandardScaler applied to all numeric features
   - Fitted on training data
   - Applied to both train and test sets

3. **Train-Test Split:**
   - Test Size: 20%
   - Train Size: 80%
   - Random State: 42 (for reproducibility)
   - Stratified: Yes (maintains class distribution)

## Model Training Details

### Label Mapping (for clarity)
```
Low Risk  → 0
High Risk → 1
```

### Class Weights (to handle imbalance)
- Low Risk weight:  0.765
- High Risk weight: 1.442

### Model Architecture
- Input Features: 6
- Hidden Layer 1: 64 neurons + ReLU + BatchNorm + Dropout(0.3)
- Hidden Layer 2: 32 neurons + ReLU + BatchNorm + Dropout(0.2)
- Hidden Layer 3: 16 neurons + ReLU
- Output Layer: 1 neuron + Sigmoid (binary classification)
- Loss Function: Binary Crossentropy
- Optimizer: Adam

### Training Configuration
- Epochs: 100
- Batch Size: 32
- Validation Split: 20%
- Class Weighting: Enabled
- Callbacks: None (but could add early stopping)

## Model Performance Metrics

### Test Set Performance (300 records)
- **Accuracy:** 74.67%
- **Precision:** 61.11%
- **Recall:** 74.04%
- **F1-Score:** 66.96%
- **ROC AUC:** 0.8356

### Full Dataset Performance (1,500 records)
- **Accuracy:** 82.20%
- **Precision:** 89.22%
- **Recall:** 82.76%
- **F1-Score:** 85.87%
- **ROC AUC:** 0.9140

### Confusion Matrix (Full Dataset)
```
             Predicted
        Low Risk  High Risk
True  Low Risk    422        98
      High Risk   169       811
```

## Data Source & Generation

### Dataset Type
- Augmented medical data (synthetically generated/enhanced)
- Used for model development and testing
- Based on realistic medical parameter ranges

### Realistic Parameter Ranges
- Age: 8-79 years (covers pediatric to elderly)
- BMI: 17.8-36.9 (underweight to obese)
- Systolic BP: 84-161 mmHg (normal to Stage 2 hypertension)
- Glucose: 81-154 mg/dL (normal to diabetic)
- Body Temperature: 95.99-99.12°F (normal to low fever)

## Files Generated from This Dataset

1. **Model Artifacts:**
   - `models/illness_risk_model.keras` - Trained DNN model
   - `models/scaler.pkl` - StandardScaler for feature normalization
   - `models/label_encoder.pkl` - Label encoding (Low Risk=0, High Risk=1)
   - `models/feature_names.pkl` - Feature column order
   - `models/label_mapping.json` - Mapping reference

2. **Training History:**
   - `models/training_history.json` - Training metrics per epoch
   - `models/training_plots.png` - Loss and accuracy curves

3. **Background Data:**
   - Used for SHAP explainer initialization (50-100 sample subset)

## Usage in Project

### Loading Dataset
```python
import pandas as pd

df = pd.read_csv('augmented_medical_data.csv')
print(df.shape)  # (1500, 7)
print(df.head())
```

### For Model Training
```python
# See src/model/train.py for full training pipeline
python src/model/train.py
```

### For Model Inference
```python
# See src/app/app.py for prediction endpoint
python src/app/app.py
```

## Data Privacy & Ethics

- Dataset: Synthetically generated for development purposes
- No real patient information
- Can be used for educational and development purposes
- Not intended for direct clinical use without proper validation

## Splitting Strategy

### Cross-Validation
- **Method:** Train-Test Split
- **Ratio:** 80-20
- **Strategy:** Stratified (maintains class proportions)
- **Random Seed:** 42 (reproducible)

### Validation During Training
- Validation split within training set: 20%
- Used for monitoring overfitting
- Final evaluation on held-out test set (300 records)

## Dataset Characteristics for Model

### Class Balance Handling
- **Issue:** Class imbalance (35% High Risk, 65% Low Risk)
- **Solution:** Applied class_weight during training
- **Result:** Better recall for High Risk class without sacrificing precision

### Data Distribution
- **Assumption:** Data is representative of real-world medical scenarios
- **Preprocessing:** StandardScaler ensures features on same scale
- **Normalization:** Applied during prediction (via same scaler)

## Notes

1. **Reproducibility:** Random state = 42 ensures same split each time
2. **Stratification:** Maintains class ratio in train and test sets
3. **Scaling:** Must use same scaler for inference as used in training
4. **Feature Order:** Must match training feature order: Age, Gender, BMI, Systolic_BP, Glucose, Body_Temp
5. **Gender Encoding:** Must use same mapping (M=0, F=1, T=2)

## Contact & Support

For questions about the dataset or model training, refer to:
- `src/model/train.py` - Full training code
- `src/app/app.py` - Prediction and inference code
- `data_quality_check.py` - Dataset quality verification script

---
**Last Updated:** February 5, 2026
**Dataset Version:** 1.0
**Project:** Illness Prediction using SHAP
