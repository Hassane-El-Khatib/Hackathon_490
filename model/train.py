#!/usr/bin/env python3
"""
Fraud Detection Model Training Script
Usage: python train.py --data bank_transactions_raw_with_fraud.csv --output outputs/
"""

import argparse
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, classification_report
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')


def encode(data, mode=1):
    """
    Encoding function provided by user
    """
    # === CONFIG ===
    ANOMALIES_OUT = "anomalies_isolation_forest.csv"
    CLUSTERED_OUT = "transactions_with_clusters.csv"
    ISO_CONTAMINATION = 0.02    # adjust if you want different anomaly rate
    # ==============

    # 1) Keep relevant digits of IP -> keep first two octets as ip_prefix
    if 'IP Address' in data.columns:
        data['IP Address'] = data['IP Address'].astype(str)
        def ip_prefix(ip):
            parts = ip.split('.')
            if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                return parts[0] + "." + parts[1]
            return "unknown"
        data['ip_prefix'] = data['IP Address'].apply(ip_prefix)
    else:
        data['ip_prefix'] = "no_ip"

    # 2) Drop these columns (user request): ID, Device ID, IP, Occupation
    cols_to_drop = []
    for c in ['TransactionID','ID','DeviceID','Device ID','IP Address','IP','CustomerOccupation','Occupation']:
        if c in data.columns:
            cols_to_drop.append(c)
    data = data.drop(columns=list(set(cols_to_drop)), errors='ignore')
    print("Dropped:", cols_to_drop)

    # 3) Parse dates and compute absolute difference between previous and current
    data['TransactionDate_parsed'] = pd.to_datetime(data['TransactionDate'], errors='coerce')
    if 'PreviousTransactionDate' in data.columns:
        data['PreviousTransactionDate_parsed'] = pd.to_datetime(data['PreviousTransactionDate'], errors='coerce')
        data['time_diff_days'] = (data['TransactionDate_parsed'] - data['PreviousTransactionDate_parsed']).abs().dt.total_seconds() / (3600*24)
    else:
        data['time_diff_days'] = np.nan

    # 4) Categorize Age (CustomerAge) into bins
    if 'CustomerAge' in data.columns:
        data['CustomerAge'] = pd.to_numeric(data['CustomerAge'], errors='coerce')
        bins = [0,17,25,35,50,65,120]
        labels = ['<18','18-25','26-35','36-50','51-65','65+']
        data['age_group'] = pd.cut(data['CustomerAge'], bins=bins, labels=labels, right=True)
        # convert to string to avoid categorical issues later
        data['age_group'] = data['age_group'].astype(str)
    else:
        data['age_group'] = "unknown"

    # 5) Basic numeric imputation (median)
    num_cols = ['TransactionAmount','TransactionDuration','LoginAttempts','AccountBalance','time_diff_days','CustomerAge']
    num_cols = [c for c in num_cols if c in data.columns]
    if num_cols:
        imputer = SimpleImputer(strategy='median')
        data[num_cols] = imputer.fit_transform(data[num_cols])

    # 6) Encode categorical columns needed for modeling
    cat_cols = ['TransactionType','Location','MerchantID','Channel','ip_prefix','age_group']
    cat_cols = [c for c in cat_cols if c in data.columns]
    encoder_map = {}
    for c in cat_cols:
        data[c] = data[c].fillna("<<MISSING>>").astype(str)
        le = LabelEncoder()
        data[c + "_enc"] = le.fit_transform(data[c])
        encoder_map[c] = le

    # 7) Prepare feature matrix
    feature_cols = num_cols + [c + "_enc" for c in cat_cols]
    X = data[feature_cols].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 8) Isolation Forest for anomaly detection (not used in this context since we already have labels)
    iso = IsolationForest(n_estimators=100, contamination=ISO_CONTAMINATION, random_state=42)
    iso_preds = iso.fit_predict(X_scaled)
    # -1 => anomaly, 1 => normal
    data['isolation_anomaly'] = (iso_preds == -1).astype(int)
    print("Anomalies found:", data['isolation_anomaly'].sum())
    
    # return data, feature info, encoders, and scaler for inference
    return data, num_cols, cat_cols, X_scaled, encoder_map, scaler, feature_cols


def load_and_preprocess_data(data_path):
    """Load and preprocess the data using the provided encoding function"""
    print("Loading data...")
    df = pd.read_csv(data_path)
    print(f"Data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Check if fraud_label exists
    if 'fraud_label' not in df.columns:
        raise ValueError("fraud_label column not found in the dataset")
    
    print(f"Fraud distribution: \n{df['fraud_label'].value_counts()}")
    
    # Apply encoding function
    print("Applying encoding function...")
    encoded_data, num_cols, cat_cols, X_scaled, encoder_map, scaler, feature_cols = encode(df.copy(), mode=0)
    
    # Prepare features and target
    y = df['fraud_label'].values
    
    return X_scaled, y, encoded_data, num_cols, cat_cols, encoder_map, scaler, feature_cols


def train_model(X, y):
    """Train the fraud detection model using XGBoost with GPU acceleration"""
    print("Training model with GPU acceleration...")
    
    # Check if CUDA is available for XGBoost
    try:
        # Use XGBoost with GPU acceleration (modern syntax)
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=len(y[y==0]) / len(y[y==1]),  # Handle class imbalance
            tree_method='hist',  # Use histogram method
            device='cuda',  # Use GPU acceleration (modern syntax)
            random_state=42,
            eval_metric='logloss',
            early_stopping_rounds=50,
            verbosity=1
        )
        
        # Fit with early stopping using a validation set
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        model.fit(
            X_train_split, y_train_split,
            eval_set=[(X_val_split, y_val_split)],
            verbose=False
        )
        
        print("GPU training completed successfully!")
        
    except Exception as e:
        print(f"GPU training failed: {e}")
        print("Falling back to CPU training...")
        
        # Fallback to CPU if GPU fails
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=len(y[y==0]) / len(y[y==1]),
            tree_method='hist',  # Use CPU
            random_state=42,
            eval_metric='logloss'
        )
        
        model.fit(X, y)
    
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate the model and return metrics"""
    print("Evaluating model...")
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc
    }
    
    return metrics, y_pred, y_pred_proba


def save_metrics(metrics, output_path):
    """Save metrics to file"""
    metrics_file = os.path.join(output_path, 'metrics.txt')
    
    with open(metrics_file, 'w') as f:
        f.write("Fraud Detection Model Evaluation Metrics\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Accuracy:  {metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall:    {metrics['recall']:.4f}\n")
        f.write(f"F1-Score:  {metrics['f1_score']:.4f}\n")
        f.write(f"ROC-AUC:   {metrics['roc_auc']:.4f}\n")
        f.write(f"PR-AUC:    {metrics['pr_auc']:.4f}\n")
    
    print(f"Metrics saved to {metrics_file}")


def main():
    parser = argparse.ArgumentParser(description='Train fraud detection model')
    parser.add_argument('--data', required=True, help='Path to CSV data file')
    parser.add_argument('--output', required=True, help='Output directory for model and metrics')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load and preprocess data
    X, y, encoded_data, num_cols, cat_cols, encoder_map, scaler, feature_cols = load_and_preprocess_data(args.data)
    
    # Split data
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"Train set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    print(f"Train fraud ratio: {y_train.mean():.4f}")
    print(f"Test fraud ratio: {y_test.mean():.4f}")
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate model
    metrics, y_pred, y_pred_proba = evaluate_model(model, X_test, y_test)
    
    # Print metrics
    print("\nModel Performance:")
    print("=" * 30)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"PR-AUC:    {metrics['pr_auc']:.4f}")
    
    # Save metrics
    save_metrics(metrics, args.output)
    
    # Save model and preprocessing artifacts
    model_path = os.path.join(args.output, 'model.pkl')
    artifacts = {
        'model': model,
        'encoder_map': encoder_map,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'num_cols': num_cols,
        'cat_cols': cat_cols
    }
    
    joblib.dump(artifacts, model_path)
    print(f"Model and artifacts saved to {model_path}")
    
    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()