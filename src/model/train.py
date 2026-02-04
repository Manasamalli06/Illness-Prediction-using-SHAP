import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report
import joblib
import os
import sys
from sqlalchemy import create_engine

# Import from sibling directories
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from model.dnn_model import create_model
from data.db_config import get_db_connection_string

# Constants
MODEL_DIR = '../../models'
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

def load_data_from_db():
    try:
        engine = create_engine(get_db_connection_string())
        query = "SELECT * FROM patient_records"
        df = pd.read_sql(query, engine)
        print(f"Loaded {len(df)} records from MySQL.")
        return df
    except Exception as e:
        print(f"Error loading from DB: {e}")
        # Fallback to CSV if DB fails
        csv_path = '../../augmented_medical_data.csv'
        if os.path.exists(csv_path):
            print(f"Falling back to CSV: {csv_path}")
            return pd.read_csv(csv_path)
        raise e

def load_and_preprocess_data():
    """
    Loads data, handles preprocessing (encoding, scaling), and splitting.
    """
    df = load_data_from_db()

    if df.empty:
        raise ValueError("Dataset is empty.")

    # Separate features and target
    X = df.drop('Risk_Label', axis=1)
    y = df['Risk_Label']

    # Encode Target (Low Risk -> 0, High Risk -> 1)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Encode Categorical Features (Gender)
    # Using simple binary encoding for Gender
    # Check if Gender is string or already encoded
    if X['Gender'].dtype == 'object':
        X['Gender'] = X['Gender'].map({'Male': 0, 'Female': 1, 'Transgender': 2})
    
    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    # Scale Numerical Features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save Scaler and LabelEncoder for inference
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))
    joblib.dump(le, os.path.join(MODEL_DIR, 'label_encoder.pkl'))
    
    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns

def train():
    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test, feature_names = load_and_preprocess_data()
    
    print(f"Input dimensions: {X_train.shape[1]}")
    model = create_model(X_train.shape[1])
    
    # Train
    print("Starting training...")
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    # Evaluate
    print("\nEvaluating model...")
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy:.4f}")
    
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save Model
    model.save(os.path.join(MODEL_DIR, 'illness_risk_model.keras'))
    print(f"Model saved to {MODEL_DIR}/illness_risk_model.keras")
    
    # Save feature names for SHAP
    joblib.dump(feature_names, os.path.join(MODEL_DIR, 'feature_names.pkl'))

if __name__ == "__main__":
    train()
