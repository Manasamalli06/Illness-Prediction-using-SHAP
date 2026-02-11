import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, accuracy_score, precision_score, 
    recall_score, f1_score, roc_auc_score, confusion_matrix
)
from xgboost import XGBClassifier
import joblib
import os
import sys
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine

# Import from sibling directories
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data.db_config import get_db_connection_string

# Constants
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, '../models')
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

def load_data():
    """Load data from CSV (prioritized for perfect training consistency)"""
    csv_path = os.path.join(SCRIPT_DIR, '../data/augmented_medical_data.csv')
    if os.path.exists(csv_path):
        print(f"Loading data from CSV: {csv_path}")
        return pd.read_csv(csv_path)
    
    try:
        engine = create_engine(get_db_connection_string())
        query = "SELECT * FROM patient_records"
        df = pd.read_sql(query, engine)
        print(f"Loaded {len(df)} records from MySQL.")
        return df
    except Exception as e:
        print(f"Database access failed: {e}")
        raise FileNotFoundError("No training data found (CSV missing and DB failed).")

def preprocess_data(df):
    """Clean and encode data for training"""
    if df.empty:
        raise ValueError("Dataset is empty.")

    # Drop non-feature columns if they exist (except Risk_Label)
    # Based on generate_perfect_training_data.py, features are:
    # Age, Gender, BMI, Systolic_BP, Glucose, Body_Temp
    required_features = ['Age', 'Gender', 'BMI', 'Systolic_BP', 'Glucose', 'Body_Temp', 'Risk_Label']
    df = df[required_features]

    # Explicit mapping for target
    label_mapping = {'Low Risk': 0, 'High Risk': 1}
    y = df['Risk_Label'].map(label_mapping)

    # Encode Gender
    # 0=Male, 1=Female, 2=Transgender (matching app.py mapping)
    X = df.drop('Risk_Label', axis=1)
    if X['Gender'].dtype == 'object':
        X['Gender'] = X['Gender'].map({'Male': 0, 'Female': 1, 'Transgender': 2})

    return X, y, X.columns

def train():
    print("=" * 80)
    print("PERFECTING MODEL TRAINING: XGBOOST OPTIMIZATION")
    print("=" * 80 + "\n")

    # 1. Load and Preprocess
    df = load_data()
    X, y, feature_names = preprocess_data(df)
    
    # 2. Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save Scaler
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))
    
    # Save LabelEncoder for app.py compatibility
    le = LabelEncoder()
    # Explicitly set classes to ensure 'Low Risk' is 0 and 'High Risk' is 1
    # Alphabetical order: High Risk (0), Low Risk (1). 
    le.classes_ = np.array(['Low Risk', 'High Risk'])
    joblib.dump(le, os.path.join(MODEL_DIR, 'label_encoder.pkl'))

    # 4. Model Training (Optimized for Perfect Data)
    print("Training XGBoost Model...")
    
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )

    model.fit(X_train_scaled, y_train)
    
    best_model = model

    # 5. Final Evaluation
    y_pred = best_model.predict(X_test_scaled)
    y_prob = best_model.predict_proba(X_test_scaled)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print("\n" + "=" * 80)
    print("FINAL TEST SET PERFORMANCE")
    print("=" * 80)
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC AUC:   {auc:.4f}")
    print("-" * 40)
    print(classification_report(y_test, y_pred, target_names=['Low Risk', 'High Risk']))

    # 6. Save Model and Artifacts
    model_path = os.path.join(MODEL_DIR, 'illness_risk_model.pkl')
    joblib.dump(best_model, model_path)
    print(f"\nModel saved to: {model_path}")

    # Save feature names for SHAP
    joblib.dump(feature_names, os.path.join(MODEL_DIR, 'feature_names.pkl'))

    # Save training history / metrics
    history = {
        'algorithm': 'XGBoost (Optimized)',
        'test_metrics': {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'auc': float(auc)
        }
    }
    
    with open(os.path.join(MODEL_DIR, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=4)

    # 7. Visualization
    plt.figure(figsize=(10, 6))
    feat_importances = pd.Series(best_model.feature_importances_, index=feature_names)
    feat_importances.nlargest(len(feature_names)).plot(kind='barh')
    plt.title('XGBoost Feature Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, 'training_plots.png'))
    plt.close()

    print("\n" + "=" * 80)
    if accuracy >= 0.99:
        print("✓✓✓ PERFECT: Model achieved near 100% accuracy on test set!")
    elif accuracy >= 0.95:
        print("✓ EXCELLENT: Model performance is outstanding.")
    else:
        print("⚠ Model performance is good, but check data for more patterns.")
    print("=" * 80)

if __name__ == "__main__":
    train()

