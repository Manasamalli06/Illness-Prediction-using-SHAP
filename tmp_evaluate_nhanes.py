
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import os

# Set paths
DATA_PATH = r'c:\Users\MANASA\OneDrive\Desktop\Illness-Prediction-using-SHAP\backend\data\nhanes_dataset.csv'

def evaluate_models():
    # Load data
    df = pd.read_csv(DATA_PATH)
    
    # Preprocess
    X = df.drop('Risk_Label', axis=1)
    y = df['Risk_Label'].map({'Low Risk': 0, 'High Risk': 1})
    
    # Encode Gender
    X['Gender'] = X['Gender'].map({'Male': 0, 'Female': 1, 'Transgender': 2})
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = []
    
    # 1. Logistic Regression
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train_scaled, y_train)
    y_pred = lr.predict(X_test_scaled)
    y_prob = lr.predict_proba(X_test_scaled)[:, 1]
    
    results.append({
        'Model': 'Logistic Regression',
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_prob)
    })
    
    # 2. Random Forest
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train_scaled, y_train)
    y_pred = rf.predict(X_test_scaled)
    y_prob = rf.predict_proba(X_test_scaled)[:, 1]
    
    results.append({
        'Model': 'Random Forest',
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_prob)
    })
    
    # 3. XGBoost
    # Note: scale_pos_weight for imbalance
    scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
    xgb_model = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, 
                                  scale_pos_weight=scale_pos_weight, random_state=42, eval_metric='logloss')
    xgb_model.fit(X_train_scaled, y_train)
    y_pred = xgb_model.predict(X_test_scaled)
    y_prob = xgb_model.predict_proba(X_test_scaled)[:, 1]
    
    results.append({
        'Model': 'XGBoost (Proposed)',
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_prob)
    })
    

    # Format and print
    print(f"{'Model':<25} | {'Acc (%)':<8} | {'Prec (%)':<9} | {'Rec (%)':<8} | {'F1 (%)':<7} | {'ROC-AUC':<8}")
    print("-" * 80)
    for res in results:
        print(f"{res['Model']:<25} | {res['Accuracy']*100:<8.2f} | {res['Precision']*100:<9.2f} | {res['Recall']*100:<8.2f} | {res['F1-Score']*100:<7.2f} | {res['ROC-AUC']:<8.4f}")

    # Save to JSON
    import json
    with open('nhanes_results.json', 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    evaluate_models()
