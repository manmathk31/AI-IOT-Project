# ==============================================================================
# Wind Turbine One-Class SVM Training Notebook — Google Colab
# Team Tech Titans — Manmath & Vishal
# ==============================================================================
#
# INSTRUCTIONS:
# 1. Open Google Colab (colab.research.google.com)
# 2. Create a new notebook
# 3. Copy each section below into separate Colab cells
# 4. Run them in order
# 5. Upload your CSV of NORMAL operating data when prompted in Section 2
#
# After training, download:
#   - model_v1.pkl     → place in backend/ml/
#   - scaler.pkl       → place in backend/ml/
#   - metrics.json     → place in backend/ml/ (replaces placeholder)
#   - score_distribution.png → for documentation
#
# Then restart the backend server.
# ==============================================================================


# ──────────────────────────────────────────────
# SECTION 1 — Install dependencies
# ──────────────────────────────────────────────

# !pip install scikit-learn numpy pandas matplotlib seaborn joblib imbalanced-learn


# ──────────────────────────────────────────────
# SECTION 2 — Upload CSV and load normal data
# ──────────────────────────────────────────────

"""
from google.colab import files
import pandas as pd
import numpy as np
import io

# Upload ONE CSV file of NORMAL operating data only.
# The CSV should have columns: timestamp, temp, humidity, vibration, current, flame
# No label column needed — all data is assumed normal.
print("Upload your CSV file of NORMAL operating data:")
uploaded = files.upload()

# Load the first uploaded file
filename = list(uploaded.keys())[0]
df = pd.read_csv(io.BytesIO(uploaded[filename]))

print(f"\\nLoaded file: {filename}")
print(f"Shape: {df.shape}")
print(f"\\nColumns: {list(df.columns)}")
print(f"\\nBasic statistics:")
print(df.describe())

# Drop rows where critical columns are NaN
before_count = len(df)
df = df.dropna(subset=['temp', 'humidity', 'current'])
after_count = len(df)
if before_count != after_count:
    print(f"\\nDropped {before_count - after_count} rows with NaN values in temp/humidity/current.")

# Sort by timestamp ascending
if 'timestamp' in df.columns:
    df = df.sort_values('timestamp').reset_index(drop=True)

print(f"\\nLoaded {len(df)} rows of normal data. Ready for training.")
"""


# ──────────────────────────────────────────────
# SECTION 3 — Sliding window feature extraction
# ──────────────────────────────────────────────

"""
import numpy as np

WINDOW_SIZE = 20
STEP = 5  # Step of 5 to reduce overlap and speed up training

def extract_features_colab(window_df):
    '''
    Extract 12 model features from a window of sensor readings.
    Same order as backend feature_engineering.py (excluding vibration_count at index 8).
    
    Features (in order):
      0: temp_mean
      1: temp_std
      2: temp_rate_of_change
      3: temp_max
      4: current_mean
      5: current_std
      6: current_spike
      7: current_rate_of_change
      8: humidity_mean
      9: humidity_std
     10: temp_rms
     11: current_rms
    '''
    temps = window_df['temp'].values
    currents = window_df['current'].values
    humidities = window_df['humidity'].values if 'humidity' in window_df.columns else np.zeros(len(temps))
    
    temp_mean = np.mean(temps)
    temp_std = np.std(temps)
    temp_rate_of_change = temps[-1] - temps[0]
    temp_max = np.max(temps)
    
    current_mean = np.mean(currents)
    current_std = np.std(currents)
    current_spike = np.max(currents) - current_mean
    current_rate_of_change = currents[-1] - currents[0]
    
    humidity_mean = np.mean(humidities)
    humidity_std = np.std(humidities)
    
    # RMS calculations
    temp_rms = np.sqrt(np.mean(temps ** 2))
    current_rms = np.sqrt(np.mean(currents ** 2))
    
    return [
        temp_mean, temp_std, temp_rate_of_change, temp_max,
        current_mean, current_std, current_spike, current_rate_of_change,
        humidity_mean, humidity_std,
        temp_rms, current_rms,
    ]


# Build feature matrix from sliding windows
feature_rows = []

for i in range(0, len(df) - WINDOW_SIZE + 1, STEP):
    window = df.iloc[i:i + WINDOW_SIZE]
    features = extract_features_colab(window)
    feature_rows.append(features)

FEATURE_NAMES = [
    'temp_mean', 'temp_std', 'temp_rate_of_change', 'temp_max',
    'current_mean', 'current_std', 'current_spike', 'current_rate_of_change',
    'humidity_mean', 'humidity_std',
    'temp_rms', 'current_rms',
]

X = np.array(feature_rows)
print(f"Extracted {len(X)} feature windows from normal data.")
print(f"Feature matrix shape: {X.shape}")
print(f"Feature names ({len(FEATURE_NAMES)}): {FEATURE_NAMES}")
"""


# ──────────────────────────────────────────────
# SECTION 4 — Scale features
# ──────────────────────────────────────────────

"""
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Feature scaling complete.")
print(f"\\nScaled feature means (should be ~0):")
for name, mean in zip(FEATURE_NAMES, X_scaled.mean(axis=0)):
    print(f"  {name:30s}: {mean:.6f}")

print(f"\\nScaled feature stds (should be ~1):")
for name, std in zip(FEATURE_NAMES, X_scaled.std(axis=0)):
    print(f"  {name:30s}: {std:.6f}")
"""


# ──────────────────────────────────────────────
# SECTION 5 — Train One-Class SVM
# ──────────────────────────────────────────────

"""
from sklearn.svm import OneClassSVM

# nu=0.05 means the model expects at most 5% of training data to be outliers.
# This is appropriate for real sensor data which always has some noise.
model = OneClassSVM(kernel='rbf', nu=0.05, gamma='scale')
model.fit(X_scaled)

print(f"Model trained on {len(X_scaled)} normal samples.")
print(f"Kernel: rbf, nu: 0.05, gamma: scale")
print(f"Support vectors: {model.n_support_}")
"""


# ──────────────────────────────────────────────
# SECTION 6 — Evaluate on training data
# ──────────────────────────────────────────────

"""
import matplotlib.pyplot as plt
import seaborn as sns

pred_train = model.predict(X_scaled)
scores_train = model.decision_function(X_scaled)

normal_rate = (pred_train == 1).mean()
anomaly_rate = (pred_train == -1).mean()

print(f"Normal detection rate on training data: {normal_rate:.2%}")
print(f"Anomaly rate on training data: {anomaly_rate:.2%}")
print(f"\\nScore range: min={scores_train.min():.3f}  max={scores_train.max():.3f}")
print(f"Score mean: {scores_train.mean():.3f}  std: {scores_train.std():.3f}")

# Plot histogram of anomaly scores
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(scores_train, bins=50, color='#2563EB', alpha=0.7, edgecolor='#1e40af')

# Mark the two thresholds
ax.axvline(x=-0.2, color='#D97706', linestyle='--', linewidth=2, label='Warning threshold (-0.2)')
ax.axvline(x=-0.5, color='#DC2626', linestyle='--', linewidth=2, label='Fault threshold (-0.5)')

ax.set_xlabel('Anomaly Score (decision_function)', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('One-Class SVM Anomaly Score Distribution on Training Data', fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('score_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
print("\\nSaved score_distribution.png")
print("\\nVerify that most scores are > -0.2 (normal zone).")
print("The -0.2 and -0.5 thresholds should separate the distribution tail.")
"""


# ──────────────────────────────────────────────
# SECTION 7 — Compute approximate feature importances via permutation
# ──────────────────────────────────────────────

"""
import numpy as np

# One-Class SVM does not have native feature importances.
# We approximate them using permutation importance:
# For each feature, shuffle it and measure how much the mean anomaly score changes.

original_mean_score = model.decision_function(X_scaled).mean()
importances = []

for i in range(X_scaled.shape[1]):
    X_permuted = X_scaled.copy()
    np.random.shuffle(X_permuted[:, i])  # Shuffle column i
    score_permuted = model.decision_function(X_permuted).mean()
    importance = original_mean_score - score_permuted
    importances.append(abs(importance))

# Normalize importances to sum to 1
total = sum(importances)
if total > 0:
    importances = [imp / total for imp in importances]

# Print sorted
print("Feature importances (permutation-based, normalized to sum=1):")
print("=" * 50)
sorted_indices = np.argsort(importances)[::-1]
for idx in sorted_indices:
    print(f"  {FEATURE_NAMES[idx]:30s}: {importances[idx]:.4f}")
"""


# ──────────────────────────────────────────────
# SECTION 8 — Save model and metrics
# ──────────────────────────────────────────────

"""
import joblib
import json

# Save model and scaler
joblib.dump(model, 'model_v1.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Saved model_v1.pkl and scaler.pkl")

# Build metrics dict
metrics = {
    "model_type": "One-Class SVM",
    "kernel": "rbf",
    "nu": 0.05,
    "training_samples": int(len(X)),
    "normal_detection_rate": round(float(normal_rate), 4),
    "anomaly_detection_rate": round(float(anomaly_rate), 4),
    "false_positive_rate": round(float(anomaly_rate), 4),
    "score_mean": round(float(scores_train.mean()), 4),
    "score_std": round(float(scores_train.std()), 4),
    "threshold_warning": -0.2,
    "threshold_fault": -0.5,
    "feature_importances": {
        name: round(float(imp), 4)
        for name, imp in zip(FEATURE_NAMES, importances)
    }
}

with open('metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print("\\nSaved metrics.json")
print(json.dumps(metrics, indent=2))
"""


# ──────────────────────────────────────────────
# SECTION 9 — Download files
# ──────────────────────────────────────────────

"""
from google.colab import files

# Download all output files
files.download('model_v1.pkl')
files.download('scaler.pkl')
files.download('metrics.json')
files.download('score_distribution.png')

# ═══════════════════════════════════════════════════════════════
# AFTER DOWNLOADING:
# ═══════════════════════════════════════════════════════════════
# 1. Place model_v1.pkl  in  backend/ml/model_v1.pkl
# 2. Place scaler.pkl    in  backend/ml/scaler.pkl
# 3. Place metrics.json  in  backend/ml/metrics.json  (replaces the placeholder)
# 4. Restart the backend server:
#       cd backend && uvicorn app:app --reload --port 8000
# 5. The system will now use the trained One-Class SVM for inference.
# ═══════════════════════════════════════════════════════════════
"""
