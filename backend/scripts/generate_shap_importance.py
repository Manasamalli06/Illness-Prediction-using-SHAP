import pandas as pd
import numpy as np
import os
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns

# Robust Path Handling
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '../../'))
MODEL_DIR = os.path.join(PROJECT_ROOT, 'backend', 'models')
DATA_PATH = os.path.join(PROJECT_ROOT, 'backend', 'data', 'augmented_medical_data.csv')

def generate_shap_importance():
    print("Loading model and data for SHAP analysis...")
    
    # Load Model and Data
    model_path = os.path.join(MODEL_DIR, 'illness_risk_model.pkl')
    scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')
    
    if not os.path.exists(model_path) or not os.path.exists(DATA_PATH):
        print("Error: Model or data not found. Please run training first.")
        return

    model = joblib.dump = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    df = pd.read_csv(DATA_PATH)
    
    # Preprocess (sync with train_perfect_final.py)
    # 1. Feature selection (exclude label)
    X = df.drop('Risk_Label', axis=1)
    
    # 2. Encode Gender
    gender_map = {'Male': 0, 'Female': 1, 'Transgender': 2}
    X['Gender'] = X['Gender'].map(gender_map)
    
    # 3. Scale Features
    feature_names = X.columns.tolist()
    X_scaled = scaler.transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names)

    # Calculate SHAP values
    print("Calculating SHAP values (this may take a moment)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled_df)

    # For XGBoost, shap_values is often just an array of the same shape as X
    # If it's a list (for multiclass or some versions), we take the positive class
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # Create the Feature Importance Bar Chart manually for better styling
    # mean(|SHAP value|)
    feature_importance = np.abs(shap_values).mean(0)
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    }).sort_values(by='Importance', ascending=False)

    # Plotting
    plt.figure(figsize=(10, 6))
    
    # Premium styling
    sns.set_theme(style="whitegrid")
    palette = sns.color_palette("viridis", len(importance_df))
    
    ax = sns.barplot(
        x='Importance', 
        y='Feature', 
        data=importance_df, 
        palette=palette,
        hue='Feature',
        legend=False
    )
    
    plt.title('SHAP Feature Importance (Global Impact)', fontsize=16, pad=20, fontweight='bold')
    plt.xlabel('mean(|SHAP value|) (average impact on model output magnitude)', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    
    # Add values on bars
    for i, p in enumerate(ax.patches):
        width = p.get_width()
        ax.text(width + 0.01, p.get_y() + p.get_height()/2, f'{width:.3f}', 
                va='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    output_path = os.path.join(PROJECT_ROOT, "shap_feature_importance.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"SHAP importance chart saved to {output_path}")

if __name__ == "__main__":
    generate_shap_importance()
