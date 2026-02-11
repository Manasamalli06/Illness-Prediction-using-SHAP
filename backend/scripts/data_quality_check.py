import pandas as pd
import numpy as np
import os
import joblib
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

CSV_PATH = 'augmented_medical_data.csv'
MODEL_DIR = 'models'

def load_data():
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        return df
    raise FileNotFoundError(CSV_PATH)


def preprocess(df, feature_names):
    df2 = df.copy()
    # Ensure Gender mapping
    if 'Gender' in df2.columns and df2['Gender'].dtype == 'object':
        df2['Gender'] = df2['Gender'].map({'Male': 0, 'Female': 1, 'Transgender': 2})
    X = df2[feature_names]
    y = df2['Risk_Label']
    return X, y


def main():
    print('Loading dataset...')
    df = load_data()
    print(f'Records: {len(df):,}')

    print('\nMissing values per column:')
    print(df.isnull().sum())

    if 'Risk_Label' not in df.columns:
        print('\nNo `Risk_Label` column found. Cannot evaluate supervised performance.')
        return

    print('\nClass distribution:')
    print(df['Risk_Label'].value_counts())

    # Load artifacts
    if not os.path.exists(MODEL_DIR):
        print('\nNo models directory found.')
        return

    try:
        scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
        le = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.pkl'))
        feature_names = joblib.load(os.path.join(MODEL_DIR, 'feature_names.pkl'))
        model = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'illness_risk_model.keras'))
    except Exception as e:
        print(f'Error loading artifacts: {e}')
        return

    print('\nFeature names used by model:')
    print(feature_names)

    # Preprocess
    X, y = preprocess(df, feature_names)

    # Handle missing rows
    if X.isnull().any().any():
        print('\nWarning: dataset contains missing values; dropping rows with nulls for evaluation')
        mask = X.notnull().all(axis=1) & df['Risk_Label'].notnull()
        X = X[mask]
        y = df.loc[mask, 'Risk_Label']

    # Scale
    X_scaled = scaler.transform(X)

    # Model predict probabilities
    probs = model.predict(X_scaled).ravel()

    # Determine which encoder class corresponds to 'High Risk'
    classes = list(le.classes_)
    print('\nLabel encoder classes:', classes)
    # model outputs prob for encoded class 1; compute prob_high accordingly
    if len(classes) > 1 and classes[1] == 'High Risk':
        prob_high = probs
    else:
        prob_high = 1.0 - probs

    y_true_encoded = le.transform(y)
    # get predicted encoded class for High Risk
    y_pred_encoded = (prob_high > 0.5).astype(int)

    # Map encoded to labels using encoder
    y_pred_labels = [ 'High Risk' if v==1 else 'Low Risk' for v in y_pred_encoded]

    # Metrics
    try:
        acc = accuracy_score(y_true_encoded, y_pred_encoded)
        prec = precision_score(y_true_encoded, y_pred_encoded)
        rec = recall_score(y_true_encoded, y_pred_encoded)
        f1 = f1_score(y_true_encoded, y_pred_encoded)
        auc = roc_auc_score(y_true_encoded, prob_high)
    except Exception as e:
        print('Error computing metrics:', e)
        return

    print('\nEvaluation on dataset:')
    print(f'Accuracy: {acc:.4f}')
    print(f'Precision: {prec:.4f}')
    print(f'Recall: {rec:.4f}')
    print(f'F1-score: {f1:.4f}')
    print(f'ROC AUC: {auc:.4f}')
    print('\nConfusion Matrix (rows=true, cols=pred):')
    print(confusion_matrix(y_true_encoded, y_pred_encoded))

    # Show some sample predictions where model disagrees with label
    mismatch = (y_true_encoded != y_pred_encoded)
    print(f'\nMismatches: {mismatch.sum()} / {len(y_true_encoded)}')
    if mismatch.sum() > 0:
        print('\nFirst 5 mismatches:')
        for idx in np.where(mismatch)[0][:5]:
            print('Index:', idx, 'True:', y.iloc[idx], 'Pred_prob_high:', prob_high[idx], 'Pred_label:', y_pred_labels[idx])

if __name__ == '__main__':
    main()
