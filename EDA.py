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
ANOMALIES_OUT = "anomalies_isolation_forest.csv"
CLUSTERED_OUT = "transactions_with_clusters.csv"
ISO_CONTAMINATION = 0.02    # adjust if you want different anomaly rate
KM_CLUSTERS = 4             # chosen default (elbow check recommended)
# ==============

# Load
df = pd.read_csv(INPUT_CSV)
data = df.copy()
print("Loaded rows,cols:", data.shape)

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

# 9) K-Means clustering (k = KM_CLUSTERS)
#km = KMeans(n_clusters=KM_CLUSTERS, n_init=10, random_state=42)
#data['kmeans_cluster'] = km.fit_predict(X_scaled)
#print("Cluster counts:\n", data['kmeans_cluster'].value_counts())

# 10) Save outputs
#data.to_csv(CLUSTERED_OUT, index=False)
#data[data['isolation_anomaly'] == 1].to_csv(ANOMALIES_OUT, index=False)
#print("Wrote:", CLUSTERED_OUT)
#print("Wrote:", ANOMALIES_OUT)

print("Basic info:")
print(data.info())
print(data.describe())

# Numeric distributions
if num_cols:
    data[num_cols].hist(bins=30, figsize=(12,8))
    plt.suptitle("Numeric Features Distribution")
    plt.show()

# Categorical counts
for c in cat_cols:
    if c != "'MerchantID" and c != "ip_prefix":
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
'''
if num_cols:
    plt.figure(figsize=(12,4))
    sns.heatmap(data[num_cols].isna(), cbar=False)
    plt.title("Missing Values Heatmap (after imputation)")
    plt.show()

for c in cat_cols:
    enc_col = c + "_enc"
    if enc_col in data.columns:
        plt.figure(figsize=(6,3))
        sns.histplot(data[enc_col], bins=len(data[c].unique()))
        plt.title(f"Encoded values for {c}")
        plt.show()
'''
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

#plt.figure(figsize=(8,5))
#sns.scatterplot(x='pca1', y='pca2', hue='kmeans_cluster', data=data, palette='tab10')
#plt.title("K-Means Clusters (PCA projection)")
#plt.show()

# Overlay anomalies
if 'isolation_anomaly' in data.columns:
    plt.figure(figsize=(8,5))
    sns.scatterplot(x='pca1', y='pca2', hue='isolation_anomaly', data=data, palette={0:'blue',1:'red'})
    plt.title("Anomalies in PCA space")
    plt.show()


# -------------------------
# 1️⃣ Add fraud_label column
# -------------------------
data['fraud_label'] = data['isolation_anomaly']  # 1 = fraud/anomaly, 0 = legitimate

# -------------------------
# 2️⃣ Drop original/unnecessary columns
# Keep only processed columns and labels
# -------------------------
# Columns to remove (original/raw ones)
raw_cols_to_drop = [
    'TransactionDate', 
    'PreviousTransactionDate',
    'TransactionID', 'ID', 'DeviceID', 'Device ID', 'IP Address', 'IP',
    'CustomerOccupation', 'Occupation',
    'CustomerAge',  # original numeric age, we have age_group
    'isolation_anomaly',  # replaced by fraud_label
    'TransactionDate_parsed', 'PreviousTransactionDate_parsed'
]

# Drop only existing columns
data_cleaned = data.drop(columns=[c for c in raw_cols_to_drop if c in data.columns])

# -------------------------
# 3️⃣ Save cleaned and labeled dataset
# -------------------------
FINAL_CSV = "bank_transactions_processed_final.csv"
data_cleaned.to_csv(FINAL_CSV, index=False)
print(f"Processed, labeled, and cleaned dataset saved to {FINAL_CSV}")



# === Simple diagnostics to print ===
print("\nTop correlations with isolation_anomaly (numeric):")
num_for_corr = [c for c in num_cols if c in data.columns] + ['isolation_anomaly']
print(data[num_for_corr].corr().abs()['isolation_anomaly'].sort_values(ascending=False).head(10))
