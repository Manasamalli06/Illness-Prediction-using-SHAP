"""
Optimization Script to Achieve 90% Accuracy
Combines feature engineering + aggressive hyperparameter tuning
"""

import sys
import os
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from sqlalchemy import create_engine

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Setup paths
sys.path.insert(0, os.path.abspath('.'))
project_root = Path(__file__).parent.parent.parent
models_dir = project_root / 'models'
models_dir.mkdir(exist_ok=True)

# Database connection
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'manasaDB@123',
    'database': 'medical_db'
}

def connect_db():
    """Connect to MySQL database"""
    try:
        engine = create_engine(
            f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}/{DB_CONFIG['database']}"
        )
        return engine
    except Exception as e:
        print(f"Database connection failed: {e}")
        return None

def load_data():
    """Load data from database with CSV fallback"""
    try:
        engine = connect_db()
        if engine is not None:
            query = "SELECT * FROM patients"
            df = pd.read_sql(query, engine)
            print("Loaded data from MySQL database")
            return df
    except Exception as e:
        print(f"Database error: {e}")
    
    print("Using CSV fallback...")
    csv_path = project_root / 'augmented_medical_data.csv'
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} records from CSV")
    return df

def engineer_features(X):
    """
    Create new features for better model performance
    
    New features:
    1. BMI Categories: underweight, normal, overweight, obese
    2. Age Groups: young, middle-aged, senior
    3. BP Categories: normal, elevated, stage1, stage2
    4. Glucose Categories: normal, prediabetic, diabetic
    5. Temperature risk: fever indicator
    6. Interaction features: Age×BMI, BP×Glucose, etc.
    7. Ratios: BP ratio, glucose to age ratio
    """
    X_eng = X.copy()
    
    # 1. BMI Categories
    X_eng['bmi_underweight'] = (X_eng['BMI'] < 18.5).astype(int)
    X_eng['bmi_normal'] = ((X_eng['BMI'] >= 18.5) & (X_eng['BMI'] < 25)).astype(int)
    X_eng['bmi_overweight'] = ((X_eng['BMI'] >= 25) & (X_eng['BMI'] < 30)).astype(int)
    X_eng['bmi_obese'] = (X_eng['BMI'] >= 30).astype(int)
    
    # 2. Age Groups
    X_eng['age_young'] = (X_eng['Age'] < 30).astype(int)
    X_eng['age_middle'] = ((X_eng['Age'] >= 30) & (X_eng['Age'] < 60)).astype(int)
    X_eng['age_senior'] = (X_eng['Age'] >= 60).astype(int)
    
    # 3. Blood Pressure Categories
    X_eng['bp_normal'] = (X_eng['Systolic_BP'] < 120).astype(int)
    X_eng['bp_elevated'] = ((X_eng['Systolic_BP'] >= 120) & (X_eng['Systolic_BP'] < 130)).astype(int)
    X_eng['bp_stage1'] = ((X_eng['Systolic_BP'] >= 130) & (X_eng['Systolic_BP'] < 140)).astype(int)
    X_eng['bp_stage2'] = (X_eng['Systolic_BP'] >= 140).astype(int)
    
    # 4. Glucose Categories
    X_eng['glucose_normal'] = (X_eng['Glucose'] < 100).astype(int)
    X_eng['glucose_prediabetic'] = ((X_eng['Glucose'] >= 100) & (X_eng['Glucose'] < 126)).astype(int)
    X_eng['glucose_diabetic'] = (X_eng['Glucose'] >= 126).astype(int)
    
    # 5. Temperature Risk
    X_eng['fever_risk'] = (X_eng['Body_Temp'] > 37.5).astype(int)
    X_eng['hypothermia_risk'] = (X_eng['Body_Temp'] < 36.5).astype(int)
    
    # 6. Interaction Features
    X_eng['age_bmi_interaction'] = X_eng['Age'] * X_eng['BMI']
    X_eng['bp_glucose_interaction'] = X_eng['Systolic_BP'] * X_eng['Glucose']
    X_eng['glucose_bmi_interaction'] = X_eng['Glucose'] * X_eng['BMI']
    X_eng['bp_temp_interaction'] = X_eng['Systolic_BP'] * X_eng['Body_Temp']
    
    # 7. Ratios
    X_eng['bp_age_ratio'] = X_eng['Systolic_BP'] / (X_eng['Age'] + 1)  # +1 to avoid division by zero
    X_eng['glucose_age_ratio'] = X_eng['Glucose'] / (X_eng['Age'] + 1)
    X_eng['bmi_age_ratio'] = X_eng['BMI'] / (X_eng['Age'] + 1)
    
    # 8. Risk score (combined indicators)
    X_eng['risk_score'] = (
        (X_eng['Systolic_BP'] > 130).astype(int) +
        (X_eng['Glucose'] > 126).astype(int) +
        (X_eng['BMI'] > 25).astype(int) +
        (X_eng['Body_Temp'] > 37).astype(int)
    )
    
    return X_eng

def load_and_preprocess_optimized(scaler_type='robust'):
    """Load and preprocess data with feature engineering"""
    df = load_data()
    
    # Separate features and target
    X = df[['Age', 'Gender', 'BMI', 'Systolic_BP', 'Glucose', 'Body_Temp']].copy()
    y = df['Risk_Label']
    
    # Encode gender
    gender_mapping = {'Male': 0, 'Female': 1, 'Transgender': 2}
    X['Gender'] = X['Gender'].map(gender_mapping)
    
    # Encode labels
    explicit_mapping = {'Low Risk': 0, 'High Risk': 1}
    y_encoded = np.array([explicit_mapping[label] for label in y])
    
    print(f"Original features: {X.shape[1]}")
    
    # ===== FEATURE ENGINEERING =====
    X = engineer_features(X)
    print(f"After feature engineering: {X.shape[1]} features")
    print(f"New features: {list(X.columns[6:])}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Scale features
    if scaler_type == 'robust':
        scaler = RobustScaler()
    elif scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
    else:
        scaler = RobustScaler()
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return (X_train_scaled, X_test_scaled, y_train, y_test, 
            scaler, None, X.columns.tolist())

def optimize_gradient_boosting():
    """
    Aggressive hyperparameter tuning for Gradient Boosting
    to reach 90% accuracy
    """
    print("\n" + "="*70)
    print("OPTIMIZING GRADIENT BOOSTING WITH FEATURE ENGINEERING")
    print("="*70)
    
    # Load data with feature engineering
    X_train, X_test, y_train, y_test, scaler, _, feature_names = \
        load_and_preprocess_optimized(scaler_type='robust')
    
    print(f"\nTraining set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
    print(f"Feature count: {X_train.shape[1]}")
    print(f"Class distribution (train): {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"Class distribution (test): {dict(zip(*np.unique(y_test, return_counts=True)))}")
    
    # AGGRESSIVE Grid Search for Gradient Boosting
    print("\n" + "-"*70)
    print("Running aggressive grid search (this will take 10-15 minutes)...")
    print("-"*70)
    
    param_grid = {
        'n_estimators': [250, 300, 350],      # More trees
        'learning_rate': [0.005, 0.01, 0.015, 0.02],  # Finer learning rates
        'max_depth': [4, 5, 6, 7],            # Deeper trees
        'min_samples_split': [2, 3, 5],       # More conservative splits
        'min_samples_leaf': [1, 2],           # More conservative leafs
        'subsample': [0.7, 0.8, 0.9, 1.0],   # More sampling variations
    }
    
    gb = GradientBoostingClassifier(
        random_state=42,
        loss='log_loss',          # Better for binary classification
        validation_fraction=0.1,  # 10% for validation
        n_iter_no_change=10       # Early stopping
    )
    
    # Use stratified K-fold for better cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    grid_search = GridSearchCV(
        gb, 
        param_grid, 
        cv=cv,
        scoring='f1_weighted',  # F1 is better for imbalanced data
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    print(f"\n✓ Best parameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    # Evaluate on test set
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
    
    print("\n" + "="*70)
    print("OPTIMIZED GRADIENT BOOSTING RESULTS (Test Set)")
    print("="*70)
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC AUC:   {roc_auc:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"  True Negatives:  {cm[0, 0]}")
    print(f"  False Positives: {cm[0, 1]}")
    print(f"  False Negatives: {cm[1, 0]}")
    print(f"  True Positives:  {cm[1, 1]}")
    
    # Cross-validation on training set
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"\nCross-validation scores (train): {cv_scores}")
    print(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    # Evaluate on full dataset (train + test)
    X_full = np.vstack([X_train, X_test])
    y_full = np.concatenate([y_train, y_test])
    y_pred_full = best_model.predict(X_full)
    accuracy_full = accuracy_score(y_full, y_pred_full)
    
    print(f"\nFull dataset accuracy: {accuracy_full:.4f} ({accuracy_full*100:.2f}%)")
    
    # Save model
    model_path = models_dir / 'illness_risk_model_optimized.pkl'
    joblib.dump(best_model, model_path)
    print(f"\n✓ Model saved to {model_path}")
    
    # Save scaler and metadata
    scaler_path = models_dir / 'scaler_optimized.pkl'
    joblib.dump(scaler, scaler_path)
    
    metadata = {
        'algorithm': 'GradientBoosting (Optimized)',
        'scaler': 'RobustScaler',
        'feature_engineering': True,
        'original_features': 6,
        'engineered_features': len(feature_names),
        'feature_names': feature_names,
        'test_accuracy': float(accuracy),
        'test_precision': float(precision),
        'test_recall': float(recall),
        'test_f1_score': float(f1),
        'test_roc_auc': float(roc_auc),
        'full_dataset_accuracy': float(accuracy_full),
        'cv_accuracy_mean': float(cv_scores.mean()),
        'cv_accuracy_std': float(cv_scores.std()),
        'best_params': best_params,
    }
    
    metadata_path = models_dir / 'metadata_optimized.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Metadata saved to {metadata_path}")
    
    # Summary
    print("\n" + "="*70)
    if accuracy >= 0.90:
        print("✓ SUCCESS! 90% ACCURACY ACHIEVED!")
    else:
        gap = 0.90 - accuracy
        print(f"⚠ Gap to 90%: {gap*100:.2f}% ({accuracy*100:.2f}% achieved)")
    print("="*70)
    
    return {
        'accuracy': accuracy,
        'best_params': best_params,
        'cv_scores': cv_scores,
        'full_accuracy': accuracy_full
    }

if __name__ == '__main__':
    result = optimize_gradient_boosting()
    
    if result['accuracy'] >= 0.90:
        print("\n✅ Model ready for production!")
    else:
        print("\n⚠ Model achieving good accuracy but below 90% target.")
        print("Consider further tuning or ensemble methods.")
