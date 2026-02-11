import pandas as pd
import numpy as np
import os
import sys
import json
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report, roc_auc_score, 
    confusion_matrix, roc_curve, precision_score, recall_score, f1_score
)
from xgboost import XGBClassifier

# Add backend to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
sys.path.append(BACKEND_DIR)

# Constants
MODEL_DIR = os.path.join(SCRIPT_DIR, '../models')
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

def load_and_preprocess_data():
    """Load and preprocess data from the perfect dataset"""
    print("Loading data...")
    csv_path = os.path.join(SCRIPT_DIR, '../data/augmented_medical_data.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Data file not found at {csv_path}. Run data generation first.")
        
    df = pd.read_csv(csv_path)

    # Label Encoding
    label_mapping = {'Low Risk': 0, 'High Risk': 1}
    y_encoded = df['Risk_Label'].map(label_mapping)

    # Encode Categorical Features
    X = df.drop('Risk_Label', axis=1)
    # Ensure Gender is numeric
    gender_map = {'Male': 0, 'Female': 1, 'Transgender': 2}
    X['Gender'] = X['Gender'].map(gender_map)

    # Split Data (80-20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # Scale Numerical Features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save Scaler and Encoding for App Use
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))
    le = LabelEncoder()
    # Explicitly set classes to ensure 'Low Risk' is 0 and 'High Risk' is 1
    # Alphabetical order: High Risk (0), Low Risk (1). 
    # To match manual mapping {'Low Risk': 0, 'High Risk': 1}, we need them in that specific order.
    le.classes_ = np.array(['Low Risk', 'High Risk'])
    joblib.dump(le, os.path.join(MODEL_DIR, 'label_encoder.pkl'))

    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns

def train():
    print("=" * 80)
    print("EXECUTING PERFECT MODEL TRAINING (XGBOOST)")
    print("=" * 80 + "\n")

    try:
        X_train, X_test, y_train, y_test, feature_names = load_and_preprocess_data()
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return

    # Using XGBoost for perfect capture of non-overlapping rules
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )

    print("Training model...")
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_prob)
    
    print("\n" + "=" * 80)
    print("PERFORMANCE RESULTS")
    print("=" * 80)
    print(f"Accuracy: {accuracy:.4%}")
    print(f"ROC AUC:  {auc:.4%}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save model artifacts in multiple formats for compatibility
    joblib.dump(model, os.path.join(MODEL_DIR, 'illness_risk_model.pkl'))
    joblib.dump(feature_names, os.path.join(MODEL_DIR, 'feature_names.pkl'))

    # Save training history for the UI
    history_dict = {
        'algorithm': 'XGBoost (Perfect Version)',
        'test_metrics': {
            'accuracy': float(accuracy),
            'precision': float(precision_score(y_test, y_pred)),
            'recall': float(recall_score(y_test, y_pred)),
            'f1_score': float(f1_score(y_test, y_pred)),
            'auc': float(auc)
        }
    }
    with open(os.path.join(MODEL_DIR, 'training_history.json'), 'w') as f:
        json.dump(history_dict, f, indent=2)
    print(f"✓ Metrics saved to training_history.json")

    # Generate Performance Plots
    print("\nGenerating performance visualizations...")
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # 1. Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        im = axes[0].imshow(cm_norm, cmap='Blues')
        axes[0].set_title(f'Confusion Matrix (Acc: {accuracy:.1%})')
        axes[0].set_xticks([0, 1])
        axes[0].set_yticks([0, 1])
        axes[0].set_xticklabels(['Low', 'High'])
        axes[0].set_yticklabels(['Low', 'High'])
        for i in range(2):
            for j in range(2):
                axes[0].text(j, i, f'{cm[i, j]}', ha='center', va='center', 
                             color='white' if cm_norm[i, j] > 0.5 else 'black')

        # 2. ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        axes[1].plot(fpr, tpr, label=f'ROC (AUC={auc:.3f})')
        axes[1].plot([0, 1], [0, 1], 'k--')
        axes[1].set_title('ROC Curve')
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].legend()

        plt.tight_layout()
        plot_path = os.path.join(MODEL_DIR, 'training_plots.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"✓ Plots saved to {plot_path}")
    except Exception as e:
        print(f"Visualization error (skipping): {e}")

    print("\n" + "=" * 80)
    print("PERFECT TRAINING COMPLETE!")
    print("=" * 80)

if __name__ == "__main__":
    train()
