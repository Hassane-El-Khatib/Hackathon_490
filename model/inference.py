#!/usr/bin/env python3
"""
Fraud Detection Model Inference Script - API Ready
Usage: 
  CLI: python inference.py --data new_data.csv --model outputs/model.pkl --output outputs/predictions.csv
  API: from inference import FraudDetector; detector = FraudDetector('outputs/model.pkl')
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
import xgboost as xgb
import joblib
import json
from typing import Dict, List, Tuple, Union
import warnings
warnings.filterwarnings('ignore')


class FraudDetector:
    """
    API-ready Fraud Detection Class for easy integration with UI/web services
    """
    
    def __init__(self, model_path: str):
        """
        Initialize the fraud detector with a trained model
        
        Args:
            model_path (str): Path to the trained model pickle file
        """
        self.model_path = model_path
        self.model = None
        self.encoder_map = None
        self.scaler = None
        self.feature_cols = None
        self.num_cols = None
        self.cat_cols = None
        self._load_model()
    
    def _load_model(self):
        """Load the trained model and preprocessing artifacts"""
        try:
            artifacts = joblib.load(self.model_path)
            self.model = artifacts['model']
            self.encoder_map = artifacts['encoder_map']
            self.scaler = artifacts['scaler']
            self.feature_cols = artifacts['feature_cols']
            self.num_cols = artifacts['num_cols']
            self.cat_cols = artifacts['cat_cols']
            print(f"✅ Model loaded successfully from {self.model_path}")
        except Exception as e:
            raise ValueError(f"❌ Failed to load model from {self.model_path}: {str(e)}")
    
    def _encode_data(self, data: pd.DataFrame) -> np.ndarray:
        """
        Apply the same encoding transformations used during training
        
        Args:
            data (pd.DataFrame): Raw transaction data
            
        Returns:
            np.ndarray: Encoded and scaled feature matrix
        """
        data_copy = data.copy()
        
        # 1) Keep relevant digits of IP -> keep first two octets as ip_prefix
        if 'IP Address' in data_copy.columns:
            data_copy['IP Address'] = data_copy['IP Address'].astype(str)
            def ip_prefix(ip):
                parts = ip.split('.')
                if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                    return parts[0] + "." + parts[1]
                return "unknown"
            data_copy['ip_prefix'] = data_copy['IP Address'].apply(ip_prefix)
        else:
            data_copy['ip_prefix'] = "no_ip"

        # 2) Drop these columns (user request): ID, Device ID, IP, Occupation
        cols_to_drop = []
        for c in ['TransactionID','ID','DeviceID','Device ID','IP Address','IP','CustomerOccupation','Occupation']:
            if c in data_copy.columns:
                cols_to_drop.append(c)
        data_copy = data_copy.drop(columns=list(set(cols_to_drop)), errors='ignore')

        # 3) Parse dates and compute absolute difference between previous and current
        data_copy['TransactionDate_parsed'] = pd.to_datetime(data_copy['TransactionDate'], errors='coerce')
        if 'PreviousTransactionDate' in data_copy.columns:
            data_copy['PreviousTransactionDate_parsed'] = pd.to_datetime(data_copy['PreviousTransactionDate'], errors='coerce')
            data_copy['time_diff_days'] = (data_copy['TransactionDate_parsed'] - data_copy['PreviousTransactionDate_parsed']).abs().dt.total_seconds() / (3600*24)
        else:
            data_copy['time_diff_days'] = np.nan

        # 4) Categorize Age (CustomerAge) into bins
        if 'CustomerAge' in data_copy.columns:
            data_copy['CustomerAge'] = pd.to_numeric(data_copy['CustomerAge'], errors='coerce')
            bins = [0,17,25,35,50,65,120]
            labels = ['<18','18-25','26-35','36-50','51-65','65+']
            data_copy['age_group'] = pd.cut(data_copy['CustomerAge'], bins=bins, labels=labels, right=True)
            # convert to string to avoid categorical issues later
            data_copy['age_group'] = data_copy['age_group'].astype(str)
        else:
            data_copy['age_group'] = "unknown"

        # 5) Basic numeric imputation (median) - using same strategy
        if self.num_cols:
            available_num_cols = [c for c in self.num_cols if c in data_copy.columns]
            if available_num_cols:
                imputer = SimpleImputer(strategy='median')
                data_copy[available_num_cols] = imputer.fit_transform(data_copy[available_num_cols])

        # 6) Encode categorical columns using saved encoders
        for c in self.cat_cols:
            if c in data_copy.columns and c in self.encoder_map:
                data_copy[c] = data_copy[c].fillna("<<MISSING>>").astype(str)
                
                # Handle unseen categories gracefully
                le = self.encoder_map[c]
                known_classes = set(le.classes_)
                
                def safe_transform(value):
                    if value in known_classes:
                        return le.transform([value])[0]
                    else:
                        # Assign to most frequent class or 0
                        return 0
                
                data_copy[c + "_enc"] = data_copy[c].apply(safe_transform)

        # 7) Prepare feature matrix using the same feature columns
        X_features = np.zeros((len(data_copy), len(self.feature_cols)))
        
        for i, col in enumerate(self.feature_cols):
            if col in data_copy.columns:
                X_features[:, i] = data_copy[col].fillna(0).values
        
        # 8) Apply the same scaling
        X_scaled = self.scaler.transform(X_features)
        
        return X_scaled
    
    def predict_single(self, transaction_data: Dict) -> Dict:
        """
        Predict fraud for a single transaction (API-friendly)
        
        Args:
            transaction_data (Dict): Single transaction as dictionary
            
        Returns:
            Dict: Prediction result with fraud probability and class
        """
        # Convert dict to DataFrame
        df = pd.DataFrame([transaction_data])
        return self.predict_batch(df)[0]
    
    def predict_batch(self, data: pd.DataFrame) -> List[Dict]:
        """
        Predict fraud for multiple transactions
        
        Args:
            data (pd.DataFrame): Batch of transactions
            
        Returns:
            List[Dict]: List of prediction results
        """
        try:
            # Encode the data
            X_scaled = self._encode_data(data)
            
            # Make predictions
            fraud_probabilities = self.model.predict_proba(X_scaled)[:, 1]
            fraud_predictions = self.model.predict(X_scaled)
            
            # Format results
            results = []
            for i in range(len(data)):
                result = {
                    'fraud_prediction': int(fraud_predictions[i]),
                    'fraud_probability': float(fraud_probabilities[i]),
                    'risk_level': self._get_risk_level(fraud_probabilities[i]),
                    'confidence': self._get_confidence(fraud_probabilities[i])
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            raise ValueError(f"❌ Prediction failed: {str(e)}")
    
    def _get_risk_level(self, probability: float) -> str:
        """Convert probability to risk level"""
        if probability >= 0.8:
            return "HIGH"
        elif probability >= 0.5:
            return "MEDIUM"
        elif probability >= 0.3:
            return "LOW"
        else:
            return "VERY_LOW"
    
    def _get_confidence(self, probability: float) -> str:
        """Get confidence level based on probability"""
        if probability >= 0.9 or probability <= 0.1:
            return "HIGH"
        elif probability >= 0.7 or probability <= 0.3:
            return "MEDIUM"
        else:
            return "LOW"
    
    def predict_csv(self, input_path: str, output_path: str) -> pd.DataFrame:
        """
        Predict fraud for CSV file and save results
        
        Args:
            input_path (str): Path to input CSV file
            output_path (str): Path to save predictions CSV
            
        Returns:
            pd.DataFrame: DataFrame with original data and predictions
        """
        # Load data
        print(f"📂 Loading data from {input_path}...")
        data = pd.read_csv(input_path)
        print(f"📊 Data shape: {data.shape}")
        
        # Get predictions
        print("🔮 Making predictions...")
        predictions = self.predict_batch(data)
        
        # Add predictions to original data
        results_df = data.copy()
        for i, pred in enumerate(predictions):
            for key, value in pred.items():
                results_df.loc[i, key] = value
        
        # Save results
        results_df.to_csv(output_path, index=False)
        print(f"💾 Results saved to {output_path}")
        
        # Print summary
        fraud_count = sum(pred['fraud_prediction'] for pred in predictions)
        avg_probability = np.mean([pred['fraud_probability'] for pred in predictions])
        
        print(f"\n📈 Prediction Summary:")
        print(f"   Total transactions: {len(predictions)}")
        print(f"   Predicted frauds: {fraud_count}")
        print(f"   Fraud rate: {fraud_count/len(predictions)*100:.2f}%")
        print(f"   Average fraud probability: {avg_probability:.4f}")
        
        return results_df


# Legacy function for backward compatibility
def predict_fraud(model_path, data_path, output_path):
    """Make fraud predictions on new data (legacy function)"""
    detector = FraudDetector(model_path)
    return detector.predict_csv(data_path, output_path)


def main():
    parser = argparse.ArgumentParser(description='Make fraud predictions on new data')
    parser.add_argument('--data', required=True, help='Path to CSV data file for inference')
    parser.add_argument('--model', required=True, help='Path to trained model file')
    parser.add_argument('--output', required=True, help='Path to output CSV file for predictions')
    
    args = parser.parse_args()
    
    # Make predictions
    results = predict_fraud(args.model, args.data, args.output)
    
    print("\nInference completed successfully!")


if __name__ == "__main__":
    main()