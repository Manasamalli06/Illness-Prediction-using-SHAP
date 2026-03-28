
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import xgboost as xgb
import os

# Set paths
DATA_PATH = r'c:\Users\MANASA\OneDrive\Desktop\Illness-Prediction-using-SHAP\backend\data\nhanes_dataset.csv'
OUTPUT_PATH = r'c:\Users\MANASA\OneDrive\Desktop\Illness-Prediction-using-SHAP\confusion_matrix.png'

def generate_plot():
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
    
    # XGBoost
    scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
    xgb_model = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, 
                                  scale_pos_weight=scale_pos_weight, random_state=42, eval_metric='logloss')
    xgb_model.fit(X_train_scaled, y_train)
    y_pred = xgb_model.predict(X_test_scaled)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Plotting
    fig, ax = plt.subplots(figsize=(8, 6))
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Low Risk', 'High Risk'])
    disp.plot(cmap='Blues', ax=ax, values_format='d', colorbar=True)
    
    plt.title('Fig. 4. Confusion Matrix - Proposed XGBoost Model', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    generate_plot()

if __name__ == "__main__":
    generate_plot()
