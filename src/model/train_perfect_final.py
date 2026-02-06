import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, accuracy_score, precision_score, 
    recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
)
import joblib
import os
import sys
import json
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Constants
MODEL_DIR = '../../models'
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

def load_and_preprocess_data():
    """Load and preprocess data"""
    print("Loading data...")
    csv_path = '../../augmented_medical_data.csv'
    df = pd.read_csv(csv_path)
    
    print(f"\nDataset shape: {df.shape}")
    print(f"Risk distribution:")
    print(df['Risk_Label'].value_counts())
    
    # Separate features and target
    X = df.drop('Risk_Label', axis=1)
    y = df['Risk_Label']
    
    # Label Encoding
    label_mapping = {'Low Risk': 0, 'High Risk': 1}
    y_encoded = y.map(label_mapping)
    
    # Encode Categorical Features
    X['Gender'] = X['Gender'].map({'Male': 0, 'Female': 1, 'Transgender': 2})
    
    # Split Data (80-20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Scale Numerical Features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save Scaler and Encoding
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))
    
    le = LabelEncoder()
    le.fit(['Low Risk', 'High Risk'])
    joblib.dump(le, os.path.join(MODEL_DIR, 'label_encoder.pkl'))
    
    with open(os.path.join(MODEL_DIR, 'label_mapping.json'), 'w') as f:
        json.dump(label_mapping, f)
    
    print(f"\nTraining set size: {X_train_scaled.shape[0]}")
    print(f"Test set size: {X_test_scaled.shape[0]}\n")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns

def create_model(input_dim):
    """Create optimized DNN model"""
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        
        # Dense block 1
        layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Dense block 2
        layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Dense block 3
        layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        # Dense block 4
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.15),
        
        # Dense block 5
        layers.Dense(16, activation='relu'),
        
        # Output layer
        layers.Dense(1, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(0.001))
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0008)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train():
    print("=" * 80)
    print("TRAINING MODEL WITH PERFECT MEDICAL DATA")
    print("=" * 80 + "\n")
    
    print("Step 1: Loading and preprocessing data...")
    X_train, X_test, y_train, y_test, feature_names = load_and_preprocess_data()
    
    # Compute class weights
    unique, counts = np.unique(y_train, return_counts=True)
    class_weight_dict = {}
    total = len(y_train)
    for u, c in zip(unique, counts):
        class_weight_dict[u] = total / (2.0 * c)
    
    print(f"Class weights: {class_weight_dict}\n")
    
    print("Step 2: Creating model architecture...")
    model = create_model(X_train.shape[1])
    print("Model architecture created.\n")
    
    # Callbacks for optimal training
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=20,
        restore_best_weights=True,
        verbose=1,
        min_delta=0.0005
    )
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.7,
        patience=10,
        min_lr=1e-7,
        verbose=1
    )
    
    print("Step 3: Starting training...\n")
    history = model.fit(
        X_train, y_train,
        epochs=150,
        batch_size=16,
        validation_split=0.15,
        class_weight=class_weight_dict,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    
    # Evaluate on test set
    print("\n" + "=" * 80)
    print("STEP 4: EVALUATING MODEL ON TEST SET")
    print("=" * 80)
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Predictions
    y_pred_prob = model.predict(X_test, verbose=0).ravel()
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    # Detailed metrics
    print("\n" + "=" * 80)
    print("DETAILED CLASSIFICATION REPORT")
    print("=" * 80)
    print(classification_report(y_test, y_pred, target_names=['Low Risk', 'High Risk']))
    
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_prob)
    
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"ROC AUC: {auc:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(cm)
    tn, fp, fn, tp = cm.ravel()
    print(f"True Negatives:  {tn} (Correct Low Risk predictions)")
    print(f"False Positives: {fp} (Incorrectly predicted High Risk)")
    print(f"False Negatives: {fn} (Incorrectly predicted Low Risk)")
    print(f"True Positives:  {tp} (Correct High Risk predictions)")
    
    # Calculate clinical metrics
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    print(f"\nClinical Metrics:")
    print(f"Sensitivity (finds High Risk): {sensitivity:.4f}")
    print(f"Specificity (correctly identifies Low Risk): {specificity:.4f}")
    print(f"PPV (confidence in High Risk prediction): {ppv:.4f}")
    print(f"NPV (confidence in Low Risk prediction): {npv:.4f}")
    
    # Success indicator
    print("\n" + "=" * 80)
    if accuracy >= 0.95 and sensitivity >= 0.90 and specificity >= 0.90:
        print(f"✓✓✓ EXCELLENT: Model is predicting correctly!")
        print(f"    Accuracy: {accuracy:.2%}")
        print(f"    Sensitivity: {sensitivity:.2%}")
        print(f"    Specificity: {specificity:.2%}")
    elif accuracy >= 0.90:
        print(f"✓ GOOD: Model performance is acceptable")
        print(f"    Accuracy: {accuracy:.2%}")
    else:
        print(f"⚠ Model accuracy: {accuracy:.2%}")
    print("=" * 80)
    
    # Save model
    model.save(os.path.join(MODEL_DIR, 'illness_risk_model.keras'))
    print(f"\nModel saved to {MODEL_DIR}/illness_risk_model.keras")
    
    joblib.dump(feature_names, os.path.join(MODEL_DIR, 'feature_names.pkl'))
    
    # Save training history
    history_dict = {
        'loss': [float(x) for x in history.history['loss']],
        'accuracy': [float(x) for x in history.history['accuracy']],
        'val_loss': [float(x) for x in history.history['val_loss']],
        'val_accuracy': [float(x) for x in history.history['val_accuracy']],
        'test_loss': float(loss),
        'test_accuracy': float(accuracy),
        'test_precision': float(precision),
        'test_recall': float(recall),
        'test_f1': float(f1),
        'test_auc': float(auc),
        'test_sensitivity': float(sensitivity),
        'test_specificity': float(specificity),
        'test_ppv': float(ppv),
        'test_npv': float(npv),
    }
    
    with open(os.path.join(MODEL_DIR, 'training_history.json'), 'w') as f:
        json.dump(history_dict, f, indent=2)
    print(f"Training history saved.")
    
    # Generate plots
    print("\nGenerating training plots...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss plot
    axes[0, 0].plot(history.history['loss'], label='Training Loss', linewidth=2)
    axes[0, 0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=10)
    axes[0, 0].set_ylabel('Loss', fontsize=10)
    axes[0, 0].set_title('Model Loss Over Epochs', fontsize=11, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[0, 1].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    axes[0, 1].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[0, 1].axhline(y=accuracy, color='r', linestyle='--', label=f'Test Accuracy: {accuracy:.2%}', linewidth=2)
    axes[0, 1].set_xlabel('Epoch', fontsize=10)
    axes[0, 1].set_ylabel('Accuracy', fontsize=10)
    axes[0, 1].set_title('Model Accuracy Over Epochs', fontsize=11, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0.4, 1.0])
    
    # Confusion Matrix visualization
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    im = axes[1, 0].imshow(cm_normalized, cmap='Blues', aspect='auto')
    axes[1, 0].set_xticks([0, 1])
    axes[1, 0].set_yticks([0, 1])
    axes[1, 0].set_xticklabels(['Low Risk', 'High Risk'])
    axes[1, 0].set_yticklabels(['Low Risk', 'High Risk'])
    axes[1, 0].set_ylabel('True Label', fontsize=10)
    axes[1, 0].set_xlabel('Predicted Label', fontsize=10)
    axes[1, 0].set_title('Normalized Confusion Matrix', fontsize=11, fontweight='bold')
    for i in range(2):
        for j in range(2):
            axes[1, 0].text(j, i, f'{cm[i, j]}\n({cm_normalized[i, j]:.1%})',
                           ha='center', va='center', color='white' if cm_normalized[i, j] > 0.5 else 'black')
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    axes[1, 1].plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC={auc:.4f})')
    axes[1, 1].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    axes[1, 1].set_xlabel('False Positive Rate', fontsize=10)
    axes[1, 1].set_ylabel('True Positive Rate', fontsize=10)
    axes[1, 1].set_title('ROC Curve', fontsize=11, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(MODEL_DIR, 'training_plots.png')
    plt.savefig(plot_path, dpi=100)
    print(f"Plots saved to {plot_path}")
    plt.close()

if __name__ == "__main__":
    train()
