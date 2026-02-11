import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import joblib
import os
import json
import matplotlib.pyplot as plt
from sqlalchemy import create_engine

# Import from sibling directories
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
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
        csv_path = '../../augmented_medical_data.csv'
        if os.path.exists(csv_path):
            print(f"Falling back to CSV: {csv_path}")
            return pd.read_csv(csv_path)
        raise e

def load_and_preprocess_data(scaler_type='standard'):
    """
    Load data and preprocess with specified scaler.
    
    Args:
        scaler_type: 'standard', 'minmax', 'robust', or 'power'
    """
    df = load_data_from_db()

    if df.empty:
        raise ValueError("Dataset is empty.")

    # Separate features and target
    X = df.drop('Risk_Label', axis=1)
    y = df['Risk_Label']

    # EXPLICIT Label Encoding
    label_mapping = {'Low Risk': 0, 'High Risk': 1}
    y_encoded = y.map(label_mapping)
    
    if y_encoded.isnull().any():
        unmapped = y[y_encoded.isnull()].unique()
        raise ValueError(f"Unmapped labels found: {unmapped}")

    # Encode Categorical Features (Gender)
    if X['Gender'].dtype == 'object':
        X['Gender'] = X['Gender'].map({'Male': 0, 'Female': 1, 'Transgender': 2})
    
    # Split Data (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, 
        test_size=0.2, 
        random_state=42, 
        stratify=y_encoded
    )
    
    # Select Scaler
    if scaler_type == 'minmax':
        scaler = MinMaxScaler()
        scaler_name = 'MinMaxScaler'
    elif scaler_type == 'robust':
        scaler = RobustScaler()
        scaler_name = 'RobustScaler'
    elif scaler_type == 'power':
        scaler = PowerTransformer(method='yeo-johnson')
        scaler_name = 'PowerTransformer'
    else:  # default
        scaler = StandardScaler()
        scaler_name = 'StandardScaler'
    
    # Scale Numerical Features
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save Scaler
    joblib.dump(scaler, os.path.join(MODEL_DIR, f'xgb_scaler_{scaler_type}.pkl'))
    
    print(f"\nUsing Scaler: {scaler_name}")
    print(f"Y train distribution: {pd.Series(y_train).value_counts().to_dict()}")
    print(f"Y test distribution: {pd.Series(y_test).value_counts().to_dict()}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns, scaler_name

def train_xgboost(scaler_type='standard'):
    print("="*60)
    print(f"Training XGBoost with {scaler_type} Scaler")
    print("="*60)
    
    X_train, X_test, y_train, y_test, feature_names, scaler_name = load_and_preprocess_data(scaler_type)
    
    print(f"\nTraining set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Calculate class weights for imbalance
    scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
    print(f"Class weight (scale_pos_weight): {scale_pos_weight:.2f}")
    
    # XGBoost Model
    print("\nTraining XGBoost model...")
    model = xgb.XGBClassifier(
        n_estimators=200,           # More trees
        max_depth=5,                # Tree depth
        learning_rate=0.05,         # Learning rate
        subsample=0.8,              # Row sampling
        colsample_bytree=0.8,       # Feature sampling
        scale_pos_weight=scale_pos_weight,  # Handle class imbalance
        random_state=42,
        eval_metric='logloss',
        verbosity=1
    )
    
    # Train with early stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        early_stopping_rounds=20,
        verbose=True
    )
    
    # Predictions
    print("\nEvaluating model...")
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_prob)
    
    print("\n" + "="*60)
    print(f"XGBoost Performance with {scaler_name}:")
    print("="*60)
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC AUC:   {auc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Low Risk', 'High Risk']))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Full dataset evaluation
    print("\n" + "-"*60)
    print("Full Dataset Evaluation:")
    print("-"*60)
    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])
    y_pred_all = model.predict(X_all)
    y_pred_prob_all = model.predict_proba(X_all)[:, 1]
    
    full_accuracy = accuracy_score(y_all, y_pred_all)
    full_precision = precision_score(y_all, y_pred_all)
    full_recall = recall_score(y_all, y_pred_all)
    full_f1 = f1_score(y_all, y_pred_all)
    full_auc = roc_auc_score(y_all, y_pred_prob_all)
    
    print(f"Full Dataset Accuracy:  {full_accuracy:.4f} ({full_accuracy*100:.2f}%)")
    print(f"Full Dataset Precision: {full_precision:.4f}")
    print(f"Full Dataset Recall:    {full_recall:.4f}")
    print(f"Full Dataset F1-Score:  {full_f1:.4f}")
    print(f"Full Dataset ROC AUC:   {full_auc:.4f}")
    
    # Save Model
    model.save_model(os.path.join(MODEL_DIR, f'xgb_model_{scaler_type}.json'))
    joblib.dump(model, os.path.join(MODEL_DIR, f'xgb_model_{scaler_type}.pkl'))
    print(f"\nModel saved to {MODEL_DIR}/xgb_model_{scaler_type}.pkl")
    
    # Save Feature Names
    joblib.dump(feature_names, os.path.join(MODEL_DIR, 'xgb_feature_names.pkl'))
    
    # Save Metrics
    metrics = {
        'scaler': scaler_name,
        'test_accuracy': float(accuracy),
        'test_precision': float(precision),
        'test_recall': float(recall),
        'test_f1': float(f1),
        'test_auc': float(auc),
        'full_accuracy': float(full_accuracy),
        'full_precision': float(full_precision),
        'full_recall': float(full_recall),
        'full_f1': float(full_f1),
        'full_auc': float(full_auc),
        'feature_importance': model.get_booster().get_score(importance_type='weight')
    }
    
    with open(os.path.join(MODEL_DIR, f'xgb_metrics_{scaler_type}.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return accuracy, full_accuracy

if __name__ == "__main__":
    print("\n" + "="*60)
    print("XGBoost Training with Multiple Scalers")
    print("="*60)
    
    results = {}
    scalers = ['standard', 'minmax', 'robust', 'power']
    
    for scaler_type in scalers:
        try:
            print(f"\n\nTesting with {scaler_type} scaler...")
            test_acc, full_acc = train_xgboost(scaler_type)
            results[scaler_type] = {
                'test_accuracy': test_acc,
                'full_accuracy': full_acc
            }
        except Exception as e:
            print(f"Error with {scaler_type} scaler: {e}")
            results[scaler_type] = {'error': str(e)}
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY - XGBoost Performance with Different Scalers")
    print("="*80)
    for scaler_type, acc in results.items():
        if 'error' not in acc:
            print(f"{scaler_type:12} | Test: {acc['test_accuracy']*100:6.2f}% | Full: {acc['full_accuracy']*100:6.2f}%")
        else:
            print(f"{scaler_type:12} | Error: {acc['error']}")
    
    # Find best
    valid_results = {k: v for k, v in results.items() if 'error' not in v}
    if valid_results:
        best_scaler = max(valid_results.items(), key=lambda x: x[1]['full_accuracy'])
        print(f"\n✅ Best Scaler: {best_scaler[0]} with {best_scaler[1]['full_accuracy']*100:.2f}% accuracy")
