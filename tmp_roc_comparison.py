
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
import xgboost as xgb
import os

# Set paths
DATA_PATH = r'c:\Users\MANASA\OneDrive\Desktop\Illness-Prediction-using-SHAP\backend\data\nhanes_dataset.csv'
OUTPUT_IMAGE = r'c:\Users\MANASA\OneDrive\Desktop\Illness-Prediction-using-SHAP\roc_comparison.png'

def generate_roc_comparison():
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
    
    plt.figure(figsize=(10, 8))
    
    # 1. XGBoost
    scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
    xgb_model = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, 
                                  scale_pos_weight=scale_pos_weight, random_state=42, eval_metric='logloss')
    xgb_model.fit(X_train_scaled, y_train)
    y_prob_xgb = xgb_model.predict_proba(X_test_scaled)[:, 1]
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_prob_xgb)
    roc_auc_xgb = auc(fpr_xgb, tpr_xgb)
    plt.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC = {roc_auc_xgb:.4f})', color='blue', linewidth=2)
    
    # 2. Random Forest
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train_scaled, y_train)
    y_prob_rf = rf.predict_proba(X_test_scaled)[:, 1]
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
    roc_auc_rf = auc(fpr_rf, tpr_rf)
    plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {roc_auc_rf:.4f})', color='green', linewidth=2)
    
    # 3. SVM
    svm = SVC(probability=True, kernel='rbf', random_state=42)
    svm.fit(X_train_scaled, y_train)
    y_prob_svm = svm.predict_proba(X_test_scaled)[:, 1]
    fpr_svm, tpr_svm, _ = roc_curve(y_test, y_prob_svm)
    roc_auc_svm = auc(fpr_svm, tpr_svm)
    plt.plot(fpr_svm, tpr_svm, label=f'SVM (AUC = {roc_auc_svm:.4f})', color='red', linewidth=2)
    
    # Plot chance line
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Chance')
    
    # Aesthetics
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve Comparison - NHANES Dataset', fontsize=14)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE, dpi=300)
    print(f"ROC comparison image saved to: {OUTPUT_IMAGE}")

if __name__ == "__main__":
    generate_roc_comparison()
