# Full EDA + preprocessing + IsolationForest
# Save as e.g. eda_bank_transactions.py and run with python3
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA


# === CONFIG ===
INPUT_CSV = "bank_transactions_data_2.csv"
# ==============

# Load
df = pd.read_csv(INPUT_CSV)
data = df.copy()
print("Loaded rows,cols:", data.shape)

def encode(data, mode = 1):

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

    # 8) Isolation Forest for anomaly detection
    iso = IsolationForest(n_estimators=100, contamination=ISO_CONTAMINATION, random_state=42)
    iso_preds = iso.fit_predict(X_scaled)
    # -1 => anomaly, 1 => normal
    data['isolation_anomaly'] = (iso_preds == -1).astype(int)
    print("Anomalies found:", data['isolation_anomaly'].sum())

    if mode == 1:
        # -------------------------
        # Add fraud_label to RAW
        # -------------------------

        # Keep only the original columns from df (the raw data you loaded first)
        raw_with_fraud = df.copy()

        # Add fraud_label from processed data
        raw_with_fraud['fraud_label'] = data['isolation_anomaly'].values

        # Save to new CSV
        RAW_WITH_FRAUD_OUT = "bank_transactions_raw_with_fraud.csv"
        raw_with_fraud.to_csv(RAW_WITH_FRAUD_OUT, index=False)

        print(f"Raw data + fraud_label saved to {RAW_WITH_FRAUD_OUT}")
    
    # return data only
    return data, num_cols, cat_cols, X_scaled

def visualize(data):
    # Some visulizations
    data, num_cols, cat_cols, X_scaled = encode(data)
    # Basic Information Insight
    print("Basic info:")
    print(data.info())
    print(data.describe())

    # Displaying some graphs

    # Numeric distributions
    if num_cols:
        data[num_cols].hist(bins=30, figsize=(12,8))
        plt.suptitle("Numeric Features Distribution")
        plt.show()

    # Categorical counts
    for c in cat_cols:
        if c != "MerchantID" and c != "ip_prefix":
            plt.figure(figsize=(8,4))
            sns.countplot(y=data[c])
            plt.title(f"Distribution of {c}")
            plt.show()

    if 'time_diff_days' in data.columns:
        plt.figure(figsize=(8,4))
        sns.histplot(data['time_diff_days'].dropna(), bins=50, kde=True)
        plt.title("Distribution of Time Differences Between Transactions (days)")
        plt.xlabel("Time difference (days)")
        plt.show()

    if 'age_group' in data.columns:
        plt.figure(figsize=(8,4))
        sns.countplot(x='age_group', data=data, order=['<18','18-25','26-35','36-50','51-65','65+'])
        plt.title("Customer Age Groups")
        plt.show()

    if 'isolation_anomaly' in data.columns:
        plt.figure(figsize=(6,3))
        sns.countplot(x='isolation_anomaly', data=data)
        plt.title("Number of anomalies detected by Isolation Forest")
        plt.show()

        # Example scatter of two key numeric features
        key_features = [f for f in ['TransactionAmount','time_diff_days'] if f in data.columns]
        if len(key_features) == 2:
            plt.figure(figsize=(8,5))
            sns.scatterplot(
                x=key_features[0], y=key_features[1],
                hue='isolation_anomaly', data=data, palette={0:'blue',1:'red'}
            )
            plt.title(f"{key_features[0]} vs {key_features[1]} (Anomalies in red)")
            plt.show()

    # PCA projection for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    data['pca1'] = X_pca[:,0]
    data['pca2'] = X_pca[:,1]

    # Overlay anomalies
    if 'isolation_anomaly' in data.columns:
        plt.figure(figsize=(8,5))
        sns.scatterplot(x='pca1', y='pca2', hue='isolation_anomaly', data=data, palette={0:'blue',1:'red'})
        plt.title("Anomalies in PCA space")
        plt.show()

    # === Simple diagnostics to print ===
    print("\nTop correlations with isolation_anomaly (numeric):")
    num_for_corr = [c for c in num_cols if c in data.columns] + ['isolation_anomaly']
    print(data[num_for_corr].corr().abs()['isolation_anomaly'].sort_values(ascending=False).head(10))

visualize(data)