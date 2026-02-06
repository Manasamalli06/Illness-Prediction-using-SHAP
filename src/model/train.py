import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
import joblib
import os
import sys
import json
import matplotlib.pyplot as plt
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
    Uses explicit label mapping: Low Risk=0, High Risk=1
    """
    df = load_data_from_db()

    if df.empty:
        raise ValueError("Dataset is empty.")

    # Separate features and target
    X = df.drop('Risk_Label', axis=1)
    y = df['Risk_Label']

    # EXPLICIT Label Encoding: Low Risk=0, High Risk=1
    # This ensures consistent mapping across training and inference
    label_mapping = {'Low Risk': 0, 'High Risk': 1}
    y_encoded = y.map(label_mapping)

    # Verify all labels were mapped
    if y_encoded.isnull().any():
        unmapped = y[y_encoded.isnull()].unique()
        raise ValueError(f"Unmapped labels found: {unmapped}")

    # Encode Categorical Features (Gender)
    if X['Gender'].dtype == 'object':
        X['Gender'] = X['Gender'].map({'Male': 0, 'Female': 1, 'Transgender': 2})

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

    # Scale Numerical Features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create LabelEncoder for compatibility with inference (classes in explicit order)
    le = LabelEncoder()
    le.fit(['Low Risk', 'High Risk'])  # Explicit order: 0=Low Risk, 1=High Risk

    # Save Scaler and LabelEncoder for inference
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))
    joblib.dump(le, os.path.join(MODEL_DIR, 'label_encoder.pkl'))

    # Save label mapping as JSON for reference
    with open(os.path.join(MODEL_DIR, 'label_mapping.json'), 'w') as f:
        json.dump(label_mapping, f)

    print(f"Label mapping: {label_mapping}")
    print(f"LabelEncoder classes: {le.classes_}")
    print(f"Y train distribution: {pd.Series(y_train).value_counts().to_dict()}")
    print(f"Y test distribution: {pd.Series(y_test).value_counts().to_dict()}")

    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns

def train():
    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test, feature_names = load_and_preprocess_data()

    print(f"Input dimensions: {X_train.shape[1]}")
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")

    # Compute class weights to handle imbalance
    unique, counts = np.unique(y_train, return_counts=True)
    class_weight_dict = {}
    total = len(y_train)
    for u, c in zip(unique, counts):
        class_weight_dict[u] = total / (2.0 * c)

    print(f"\nClass weights: {class_weight_dict}")

    # XGBoost with hyperparameter tuning for high accuracy
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'scale_pos_weight': [class_weight_dict[1]]  # Handle class imbalance
    }

    xgb = XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42,
        use_label_encoder=False
    )

    print("Starting XGBoost hyperparameter tuning...")
    grid_search = GridSearchCV(
        xgb,
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=2
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print(f"\nBest parameters: {grid_search.best_params_}")

    # Evaluate on test set
    print("\nEvaluating XGBoost on test set...")
    y_pred = best_model.predict(X_test)
    y_pred_prob = best_model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}")

    # Compute detailed metrics
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Low Risk', 'High Risk']))

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_prob)

    print(f"\nPrecision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"ROC AUC: {auc:.4f}")

    # Save Model
    joblib.dump(best_model, os.path.join(MODEL_DIR, 'illness_risk_model.pkl'))
    print(f"\nModel saved to {MODEL_DIR}/illness_risk_model.pkl")

    # Save feature names for SHAP
    joblib.dump(feature_names, os.path.join(MODEL_DIR, 'feature_names.pkl'))

    # Save training history (simplified for XGBoost)
    history_dict = {
        'algorithm': 'XGBoost',
        'best_params': grid_search.best_params_,
        'cv_results': grid_search.cv_results_,
        'test_accuracy': float(accuracy),
        'test_precision': float(precision),
        'test_recall': float(recall),
        'test_f1': float(f1),
        'test_auc': float(auc),
        'class_weights': class_weight_dict
    }

    with open(os.path.join(MODEL_DIR, 'training_history.json'), 'w') as f:
        json.dump(history_dict, f, indent=2)
    print(f"Training history saved to {MODEL_DIR}/training_history.json")

    # Generate feature importance plot
    print("\nGenerating feature importance plot...")
    plt.figure(figsize=(10, 6))
    feature_importance = best_model.feature_importances_
    sorted_idx = np.argsort(feature_importance)[::-1]
    plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx])
    plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
    plt.xlabel('Feature Importance')
    plt.title('XGBoost Feature Importance')
    plt.tight_layout()
    plot_path = os.path.join(MODEL_DIR, 'training_plots.png')
    plt.savefig(plot_path, dpi=100)
    print(f"Feature importance plot saved to {plot_path}")
    plt.close()

if __name__ == "__main__":
    train()
