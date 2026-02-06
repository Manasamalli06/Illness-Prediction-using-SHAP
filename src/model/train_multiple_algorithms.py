import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, accuracy_score, precision_score, 
                             recall_score, f1_score, roc_auc_score, confusion_matrix)
import joblib
import os
import json
from sqlalchemy import create_engine
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data.db_config import get_db_connection_string

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

def load_and_preprocess_data(scaler_type='robust'):
    """
    Load, preprocess, and scale data for ML models
    
    Args:
        scaler_type: 'standard', 'robust', or 'minmax'
    """
    df = load_data_from_db()
    
    if df.empty:
        raise ValueError("Dataset is empty.")
    
    # Separate features and target
    X = df.drop('Risk_Label', axis=1)
    y = df['Risk_Label']
    
    # Explicit label mapping
    label_mapping = {'Low Risk': 0, 'High Risk': 1}
    y_encoded = y.map(label_mapping)
    
    if y_encoded.isnull().any():
        unmapped = y[y_encoded.isnull()].unique()
        raise ValueError(f"Unmapped labels found: {unmapped}")
    
    # Encode Gender
    if X['Gender'].dtype == 'object':
        X['Gender'] = X['Gender'].map({'Male': 0, 'Female': 1, 'Transgender': 2})
    
    # Split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Apply appropriate scaler
    if scaler_type == 'robust':
        scaler = RobustScaler()  # Better for data with outliers
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()  # Scales to [0, 1]
    else:
        scaler = StandardScaler()  # Default: zero mean, unit variance
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler and metadata
    joblib.dump(scaler, os.path.join(MODEL_DIR, f'scaler_{scaler_type}.pkl'))
    joblib.dump(X.columns, os.path.join(MODEL_DIR, 'feature_names.pkl'))
    
    print(f"Scaler: {scaler_type}")
    print(f"Y train: {pd.Series(y_train).value_counts().to_dict()}")
    print(f"Y test:  {pd.Series(y_test).value_counts().to_dict()}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns, scaler_type


def train_random_forest(X_train, X_test, y_train, y_test, scaler_type):
    """Train RandomForest with hyperparameter tuning"""
    print("\n" + "="*60)
    print("TRAINING RANDOM FOREST")
    print("="*60)
    
    # Hyperparameter tuning
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'class_weight': ['balanced', 'balanced_subsample']
    }
    
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
    
    print("Performing grid search (this may take a moment)...")
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    print(f"Best params: {grid_search.best_params_}")
    
    # Evaluate
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'algorithm': 'RandomForest',
        'scaler': scaler_type,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
    }
    
    print(f"\nAccuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    print(f"ROC AUC:   {metrics['roc_auc']:.4f}")
    
    return best_model, metrics


def train_gradient_boosting(X_train, X_test, y_train, y_test, scaler_type):
    """Train Gradient Boosting with hyperparameter tuning"""
    print("\n" + "="*60)
    print("TRAINING GRADIENT BOOSTING")
    print("="*60)
    
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5],
        'subsample': [0.8, 1.0]
    }
    
    gb = GradientBoostingClassifier(random_state=42)
    grid_search = GridSearchCV(gb, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
    
    print("Performing grid search (this may take a moment)...")
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    print(f"Best params: {grid_search.best_params_}")
    
    # Evaluate
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'algorithm': 'GradientBoosting',
        'scaler': scaler_type,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
    }
    
    print(f"\nAccuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    print(f"ROC AUC:   {metrics['roc_auc']:.4f}")
    
    return best_model, metrics


def train_decision_tree(X_train, X_test, y_train, y_test, scaler_type):
    """Train Decision Tree with hyperparameter tuning"""
    print("\n" + "="*60)
    print("TRAINING DECISION TREE")
    print("="*60)
    
    param_grid = {
        'max_depth': [5, 10, 15, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'criterion': ['gini', 'entropy'],
        'class_weight': ['balanced', None]
    }
    
    dt = DecisionTreeClassifier(random_state=42)
    grid_search = GridSearchCV(dt, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
    
    print("Performing grid search...")
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    print(f"Best params: {grid_search.best_params_}")
    
    # Evaluate
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'algorithm': 'DecisionTree',
        'scaler': scaler_type,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
    }
    
    print(f"\nAccuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    print(f"ROC AUC:   {metrics['roc_auc']:.4f}")
    
    return best_model, metrics


def train_svm(X_train, X_test, y_train, y_test, scaler_type):
    """Train SVM with hyperparameter tuning"""
    print("\n" + "="*60)
    print("TRAINING SUPPORT VECTOR MACHINE (SVM)")
    print("="*60)
    
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['rbf', 'poly'],
        'gamma': ['scale', 'auto'],
        'class_weight': ['balanced', None]
    }
    
    svm = SVC(probability=True, random_state=42)
    grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
    
    print("Performing grid search (this may take longer)...")
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    print(f"Best params: {grid_search.best_params_}")
    
    # Evaluate
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'algorithm': 'SVM',
        'scaler': scaler_type,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
    }
    
    print(f"\nAccuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    print(f"ROC AUC:   {metrics['roc_auc']:.4f}")
    
    return best_model, metrics


def train_all_models():
    """Train all models and compare"""
    print("\n" + "="*70)
    print("ILLNESS PREDICTION - MULTI-ALGORITHM TRAINING")
    print("="*70)
    
    # Use RobustScaler as it's generally better for mixed data
    X_train, X_test, y_train, y_test, feature_names, scaler_type = load_and_preprocess_data('robust')
    
    results = []
    models = {}
    
    # Train all models
    try:
        rf, rf_metrics = train_random_forest(X_train, X_test, y_train, y_test, scaler_type)
        results.append(rf_metrics)
        models['RandomForest'] = rf
    except Exception as e:
        print(f"RandomForest training failed: {e}")
    
    try:
        gb, gb_metrics = train_gradient_boosting(X_train, X_test, y_train, y_test, scaler_type)
        results.append(gb_metrics)
        models['GradientBoosting'] = gb
    except Exception as e:
        print(f"Gradient Boosting training failed: {e}")
    
    try:
        dt, dt_metrics = train_decision_tree(X_train, X_test, y_train, y_test, scaler_type)
        results.append(dt_metrics)
        models['DecisionTree'] = dt
    except Exception as e:
        print(f"Decision Tree training failed: {e}")
    
    try:
        svm, svm_metrics = train_svm(X_train, X_test, y_train, y_test, scaler_type)
        results.append(svm_metrics)
        models['SVM'] = svm
    except Exception as e:
        print(f"SVM training failed: {e}")
    
    # Compare results
    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('accuracy', ascending=False)
    print(results_df.to_string(index=False))
    
    # Save comparison
    with open(os.path.join(MODEL_DIR, 'model_comparison.json'), 'w') as f:
        json.dump(results_df.to_dict(orient='records'), f, indent=2)
    
    # Save best model
    best_result = results_df.iloc[0]
    best_algorithm = best_result['algorithm']
    best_model = models[best_algorithm]
    
    print(f"\n{'='*70}")
    print(f"BEST MODEL: {best_algorithm}")
    print(f"Accuracy: {best_result['accuracy']:.4f}")
    print(f"Precision: {best_result['precision']:.4f}")
    print(f"Recall: {best_result['recall']:.4f}")
    print(f"F1-Score: {best_result['f1_score']:.4f}")
    print(f"ROC AUC: {best_result['roc_auc']:.4f}")
    print(f"{'='*70}")
    
    # Save best model
    model_file = os.path.join(MODEL_DIR, f'illness_risk_model_{best_algorithm.lower()}.pkl')
    joblib.dump(best_model, model_file)
    print(f"\nBest model saved to: {model_file}")
    
    # Save metadata
    metadata = {
        'best_algorithm': best_algorithm,
        'scaler': scaler_type,
        'accuracy': float(best_result['accuracy']),
        'precision': float(best_result['precision']),
        'recall': float(best_result['recall']),
        'f1_score': float(best_result['f1_score']),
        'roc_auc': float(best_result['roc_auc']),
        'feature_names': list(feature_names)
    }
    
    with open(os.path.join(MODEL_DIR, 'best_model_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return best_model, best_algorithm, metadata


if __name__ == "__main__":
    best_model, best_algo, metadata = train_all_models()
