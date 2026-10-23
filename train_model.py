"""
Model Training and Evaluation Pipeline
Project: At-Risk Student Early Warning System Using Deep Neural Networks
Trains the primary Deep Neural Network (DNN) and baseline ML models,
saves evaluation metrics, confusion matrix, and training history plots.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, callbacks

import config

def train_pipeline():
    print("=" * 65)
    print("  DEEP NEURAL NETWORK (DNN) TRAINING PIPELINE")
    print("=" * 65)
    
    # 1. Load Dataset
    if not os.path.exists(config.DATASET_PATH):
        print(f"[!] Dataset not found at {config.DATASET_PATH}. Running prepare_dataset...")
        from prepare_dataset import prepare_dataset
        prepare_dataset()
        
    df = pd.read_csv(config.DATASET_PATH)
    print(f"[+] Loaded dataset: {len(df)} records with {len(config.FEATURE_COLUMNS)} features.")
    
    X = df[config.FEATURE_COLUMNS].values
    y = df[config.TARGET_COLUMN].values
    
    # 2. Stratified Train / Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"[+] Split: {len(X_train)} training samples, {len(X_test)} testing samples.")
    
    # 3. Standard Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    joblib.dump(scaler, config.SCALER_PATH)
    print(f"[OK] Fitted StandardScaler saved to:\n    {config.SCALER_PATH}")
    
    # 4. Build Deep Neural Network Architecture
    print("\n[*] Building Deep Neural Network Architecture:")
    dnn_model = models.Sequential([
        # Input & First Hidden Layer
        layers.Input(shape=(len(config.FEATURE_COLUMNS),)),
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Second Hidden Layer
        layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.2),
        
        # Third Hidden Layer
        layers.Dense(16, activation='relu'),
        
        # Softmax Multi-Class Output Layer (3 Classes: Low, Medium, High)
        layers.Dense(3, activation='softmax')
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    dnn_model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    dnn_model.summary()
    
    # 5. Callbacks: Early Stopping, Checkpointing, LR Reduction
    training_callbacks = [
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=14,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            filepath=config.MODEL_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-5,
            verbose=1
        )
    ]
    
    # 6. Train the DNN
    print("\n[*] Commencing DNN Model Training...")
    history = dnn_model.fit(
        X_train_scaled,
        y_train,
        validation_data=(X_test_scaled, y_test),
        epochs=60,
        batch_size=32,
        callbacks=training_callbacks,
        verbose=1
    )
    
    # Ensure final best model is saved
    dnn_model.save(config.MODEL_PATH)
    print(f"[OK] Trained DNN Model successfully saved to:\n    {config.MODEL_PATH}")
    
    # 7. Evaluate DNN on Test Set
    y_pred_probs = dnn_model.predict(X_test_scaled)
    y_pred_dnn = np.argmax(y_pred_probs, axis=1)
    
    dnn_acc = float(accuracy_score(y_test, y_pred_dnn))
    dnn_prec = float(precision_score(y_test, y_pred_dnn, average='weighted', zero_division=0))
    dnn_rec = float(recall_score(y_test, y_pred_dnn, average='weighted', zero_division=0))
    dnn_f1 = float(f1_score(y_test, y_pred_dnn, average='weighted', zero_division=0))
    
    dnn_cm = confusion_matrix(y_test, y_pred_dnn).tolist()
    dnn_cr = classification_report(
        y_test, y_pred_dnn,
        target_names=['Low Risk', 'Medium Risk', 'High Risk'],
        output_dict=True
    )
    
    print("\n" + "=" * 65)
    print(f"  DNN TEST RESULTS: Accuracy = {dnn_acc * 100:.2f}%, F1 = {dnn_f1 * 100:.2f}%")
    print("=" * 65)
    
    # 8. Train Baseline ML Models for Rigorous Comparison
    print("\n[*] Training Baseline Models for Comparative Evaluation...")
    # Logistic Regression Baseline
    log_reg = LogisticRegression(max_iter=1000, random_state=42)
    log_reg.fit(X_train_scaled, y_train)
    y_pred_lr = log_reg.predict(X_test_scaled)
    lr_acc = float(accuracy_score(y_test, y_pred_lr))
    lr_prec = float(precision_score(y_test, y_pred_lr, average='weighted', zero_division=0))
    lr_rec = float(recall_score(y_test, y_pred_lr, average='weighted', zero_division=0))
    lr_f1 = float(f1_score(y_test, y_pred_lr, average='weighted', zero_division=0))
    
    # Random Forest Baseline
    rf_clf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
    rf_clf.fit(X_train_scaled, y_train)
    y_pred_rf = rf_clf.predict(X_test_scaled)
    rf_acc = float(accuracy_score(y_test, y_pred_rf))
    rf_prec = float(precision_score(y_test, y_pred_rf, average='weighted', zero_division=0))
    rf_rec = float(recall_score(y_test, y_pred_rf, average='weighted', zero_division=0))
    rf_f1 = float(f1_score(y_test, y_pred_rf, average='weighted', zero_division=0))
    
    # Feature Importances from Random Forest
    rf_importances = rf_clf.feature_importances_
    sorted_idx = np.argsort(rf_importances)[::-1]
    feature_ranking = [
        {'feature': config.FEATURE_COLUMNS[i], 'importance': float(rf_importances[i])}
        for i in sorted_idx
    ]
    
    # Save baseline models
    joblib.dump({'logistic_regression': log_reg, 'random_forest': rf_clf}, config.BASELINE_MODELS_PATH)
    print(f"[OK] Baseline models saved to: {config.BASELINE_MODELS_PATH}")
    
    # Summary Comparison Table
    comparison_table = [
        {
            'model': 'Logistic Regression (Baseline)',
            'accuracy': round(lr_acc * 100, 2),
            'precision': round(lr_prec * 100, 2),
            'recall': round(lr_rec * 100, 2),
            'f1_score': round(lr_f1 * 100, 2)
        },
        {
            'model': 'Random Forest Classifier',
            'accuracy': round(rf_acc * 100, 2),
            'precision': round(rf_prec * 100, 2),
            'recall': round(rf_rec * 100, 2),
            'f1_score': round(rf_f1 * 100, 2)
        },
        {
            'model': 'Deep Neural Network (Proposed)',
            'accuracy': round(dnn_acc * 100, 2),
            'precision': round(dnn_prec * 100, 2),
            'recall': round(dnn_rec * 100, 2),
            'f1_score': round(dnn_f1 * 100, 2)
        }
    ]
    
    print("\n--- MODEL PERFORMANCE COMPARISON ---")
    print(f"{'Model':<32} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    print("-" * 72)
    for row in comparison_table:
        print(f"{row['model']:<32} {row['accuracy']:>6.2f}%   {row['precision']:>6.2f}%   {row['recall']:>6.2f}%   {row['f1_score']:>6.2f}%")
        
    # 9. Save Metrics JSON
    metrics_data = {
        'dnn': {
            'accuracy': round(dnn_acc * 100, 2),
            'precision': round(dnn_prec * 100, 2),
            'recall': round(dnn_rec * 100, 2),
            'f1_score': round(dnn_f1 * 100, 2),
            'classification_report': dnn_cr,
            'confusion_matrix': dnn_cm
        },
        'logistic_regression': {
            'accuracy': round(lr_acc * 100, 2),
            'precision': round(lr_prec * 100, 2),
            'recall': round(lr_rec * 100, 2),
            'f1_score': round(lr_f1 * 100, 2)
        },
        'random_forest': {
            'accuracy': round(rf_acc * 100, 2),
            'precision': round(rf_prec * 100, 2),
            'recall': round(rf_rec * 100, 2),
            'f1_score': round(rf_f1 * 100, 2)
        },
        'comparison': comparison_table,
        'feature_importance': feature_ranking,
        'test_samples_count': int(len(y_test)),
        'epochs_trained': int(len(history.history['loss']))
    }
    
    with open(config.METRICS_PATH, 'w') as f:
        json.dump(metrics_data, f, indent=2)
    print(f"[OK] Full evaluation metrics saved to:\n    {config.METRICS_PATH}")
    
    # 10. Generate and Save Visualizations
    generate_plots(history, dnn_cm, comparison_table, feature_ranking)
    print("\n[OK] Training pipeline finished successfully!")
    print("=" * 65)

def generate_plots(history, cm, comparison, feature_ranking):
    """Generates all publication/presentation quality evaluation graphs."""
    epochs = range(1, len(history.history['accuracy']) + 1)
    
    # 1. Accuracy Plot
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history.history['accuracy'], 'b-o', markersize=4, label='Training Accuracy', linewidth=2)
    plt.plot(epochs, history.history['val_accuracy'], 'g-s', markersize=4, label='Validation Accuracy', linewidth=2)
    plt.title('DNN Model Accuracy vs. Epochs', fontsize=14, fontweight='bold', pad=12)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right', fontsize=11)
    plt.tight_layout()
    acc_path = os.path.join(config.PLOTS_DIR, 'training_validation_accuracy.png')
    plt.savefig(acc_path, dpi=160)
    plt.close()
    print(f"[+] Saved plot: {acc_path}")
    
    # 2. Loss Plot
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history.history['loss'], 'r-o', markersize=4, label='Training Loss', linewidth=2)
    plt.plot(epochs, history.history['val_loss'], 'm-s', markersize=4, label='Validation Loss', linewidth=2)
    plt.title('DNN Model Loss vs. Epochs', fontsize=14, fontweight='bold', pad=12)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Cross-Entropy Loss', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper right', fontsize=11)
    plt.tight_layout()
    loss_path = os.path.join(config.PLOTS_DIR, 'training_validation_loss.png')
    plt.savefig(loss_path, dpi=160)
    plt.close()
    print(f"[+] Saved plot: {loss_path}")
    
    # 3. Confusion Matrix Heatmap
    plt.figure(figsize=(7, 6))
    class_labels = ['Low Risk', 'Medium Risk', 'High Risk']
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_labels,
        yticklabels=class_labels,
        cbar=True,
        annot_kws={"size": 14, "weight": "bold"}
    )
    plt.title('DNN Test Set Confusion Matrix', fontsize=14, fontweight='bold', pad=12)
    plt.xlabel('Predicted Academic Risk', fontsize=12, fontweight='semibold')
    plt.ylabel('Actual Academic Risk', fontsize=12, fontweight='semibold')
    plt.tight_layout()
    cm_path = os.path.join(config.PLOTS_DIR, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=160)
    plt.close()
    print(f"[+] Saved plot: {cm_path}")
    
    # 4. Model Comparison Bar Chart
    plt.figure(figsize=(9, 5))
    model_names = [r['model'].replace(' (Baseline)', '').replace(' (Proposed)', '') for r in comparison]
    accuracies = [r['accuracy'] for r in comparison]
    f1_scores = [r['f1_score'] for r in comparison]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    plt.bar(x - width/2, accuracies, width, label='Accuracy (%)', color='#3b82f6', alpha=0.9)
    plt.bar(x + width/2, f1_scores, width, label='F1-Score (%)', color='#10b981', alpha=0.9)
    
    plt.title('Comparative Model Performance (ML vs DNN)', fontsize=14, fontweight='bold', pad=12)
    plt.ylabel('Percentage (%)', fontsize=12)
    plt.xticks(x, model_names, fontsize=11, fontweight='medium')
    plt.ylim(0, 105)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(loc='lower right', fontsize=11)
    
    for i in range(len(model_names)):
        plt.text(x[i] - width/2, accuracies[i] + 1.5, f"{accuracies[i]:.1f}%", ha='center', fontsize=10, fontweight='bold')
        plt.text(x[i] + width/2, f1_scores[i] + 1.5, f"{f1_scores[i]:.1f}%", ha='center', fontsize=10, fontweight='bold')
        
    plt.tight_layout()
    comp_path = os.path.join(config.PLOTS_DIR, 'model_comparison.png')
    plt.savefig(comp_path, dpi=160)
    plt.close()
    print(f"[+] Saved plot: {comp_path}")
    
    # 5. Feature Importance Horizontal Bar Chart
    plt.figure(figsize=(9, 5.5))
    feats = [item['feature'].replace('_', ' ').title() for item in feature_ranking][::-1]
    imps = [item['importance'] * 100 for item in feature_ranking][::-1]
    
    plt.barh(feats, imps, color='#6366f1', alpha=0.85, edgecolor='#4338ca')
    plt.title('Relative Academic Factor Importance', fontsize=14, fontweight='bold', pad=12)
    plt.xlabel('Importance Contribution (%)', fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    
    for idx, v in enumerate(imps):
        plt.text(v + 0.3, idx, f"{v:.1f}%", va='center', fontsize=9, fontweight='bold')
        
    plt.tight_layout()
    feat_path = os.path.join(config.PLOTS_DIR, 'feature_importance.png')
    plt.savefig(feat_path, dpi=160)
    plt.close()
    print(f"[+] Saved plot: {feat_path}")

if __name__ == '__main__':
    train_pipeline()
