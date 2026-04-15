# ==============================================================================
# Wind Turbine ML Training Notebook — Google Colab
# Team Tech Titans — Manmath & Vishal
# ==============================================================================
#
# INSTRUCTIONS:
# 1. Open Google Colab (colab.research.google.com)
# 2. Create a new notebook
# 3. Copy each cell below into separate Colab cells
# 4. Run them in order
# 5. Upload your CSV files when prompted in Cell 2
#
# After training, download:
#   - model_v1.pkl  → place in backend/ml/
#   - scaler.pkl    → place in backend/ml/
#   - metrics.json  → place in backend/ml/
# ==============================================================================

# ──────────────────────────────────────────────
# CELL 1 — Install dependencies
# ──────────────────────────────────────────────
# !pip install imbalanced-learn scikit-learn numpy pandas matplotlib seaborn joblib


# ──────────────────────────────────────────────
# CELL 2 — Upload and merge CSVs
# ──────────────────────────────────────────────
"""
from google.colab import files
import pandas as pd
import io

# Upload all 4 CSV files (normal.csv, warning.csv, fault.csv, flame.csv)
uploaded = files.upload()

dfs = []
for name, content in uploaded.items():
    df = pd.read_csv(io.BytesIO(content))
    print(f"Loaded {name}: {len(df)} rows, labels: {df['label'].unique()}")
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)
print(f"\\nTotal dataset: {len(df)} rows")
print(df['label'].value_counts())
"""


# ──────────────────────────────────────────────
# CELL 3 — Feature engineering (sliding window)
# ──────────────────────────────────────────────
"""
import numpy as np

def extract_features_from_window(window_df):
    temps = window_df['temp'].values
    currents = window_df['current'].values
    vibrations = window_df['vibration'].values
    flames = window_df['flame'].values
    humidities = window_df['humidity'].values if 'humidity' in window_df.columns else np.zeros(len(temps))

    return {
        'temp_mean': np.mean(temps),
        'temp_std': np.std(temps),
        'temp_rate_of_change': temps[-1] - temps[0],
        'temp_max': np.max(temps),
        'current_mean': np.mean(currents),
        'current_std': np.std(currents),
        'current_spike': np.max(currents) - np.mean(currents),
        'current_rate_of_change': currents[-1] - currents[0],
        'vibration_count': int(np.sum(vibrations)),
        'flame_count': int(np.sum(flames)),
        'humidity_mean': np.mean(humidities),
        'humidity_std': np.std(humidities),
    }

WINDOW_SIZE = 20
STEP = 1
feature_rows = []

for i in range(0, len(df) - WINDOW_SIZE, STEP):
    window = df.iloc[i:i + WINDOW_SIZE]
    label = window['label'].mode()[0]
    features = extract_features_from_window(window)
    features['label'] = label
    feature_rows.append(features)

features_df = pd.DataFrame(feature_rows)
print(f"Feature dataset: {len(features_df)} windows")
print(features_df['label'].value_counts())
"""


# ──────────────────────────────────────────────
# CELL 4 — Encode labels and handle class imbalance
# ──────────────────────────────────────────────
"""
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

label_encoder = LabelEncoder()
LABEL_ORDER = ['Normal', 'Warning', 'Fault', 'CRITICAL_FLAME']
label_encoder.fit(LABEL_ORDER)

feature_cols = ['temp_mean', 'temp_std', 'temp_rate_of_change', 'temp_max',
                'current_mean', 'current_std', 'current_spike', 'current_rate_of_change',
                'vibration_count', 'flame_count', 'humidity_mean', 'humidity_std']

X = features_df[feature_cols].values
y = label_encoder.transform(features_df['label'].values)

print("Class distribution before SMOTE:", dict(zip(label_encoder.classes_, np.bincount(y))))

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
print("Class distribution after SMOTE:", dict(zip(label_encoder.classes_, np.bincount(y_resampled))))
"""


# ──────────────────────────────────────────────
# CELL 5 — Scale features
# ──────────────────────────────────────────────
"""
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_resampled)
"""


# ──────────────────────────────────────────────
# CELL 6 — Train/test split and model training
# ──────────────────────────────────────────────
"""
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)
print("Model trained.")
"""


# ──────────────────────────────────────────────
# CELL 7 — Evaluate
# ──────────────────────────────────────────────
"""
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score (weighted): {f1:.4f}")
print("\\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

cm = confusion_matrix(y_test, y_pred)
print("\\nConfusion Matrix:")
print(cm)
"""


# ──────────────────────────────────────────────
# CELL 8 — Plot confusion matrix and feature importance
# ──────────────────────────────────────────────
"""
import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_, ax=axes[0])
axes[0].set_title('Confusion Matrix')
axes[0].set_ylabel('Actual')
axes[0].set_xlabel('Predicted')

# Feature importances
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
axes[1].barh(range(len(feature_cols)), importances[indices], color='#2563EB')
axes[1].set_yticks(range(len(feature_cols)))
axes[1].set_yticklabels([feature_cols[i] for i in indices])
axes[1].set_title('Feature Importances')
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig('model_evaluation.png', dpi=150, bbox_inches='tight')
plt.show()
"""


# ──────────────────────────────────────────────
# CELL 9 — Save model files
# ──────────────────────────────────────────────
"""
import joblib
import json
from sklearn.metrics import precision_score, recall_score

joblib.dump(model, 'model_v1.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Saved model_v1.pkl and scaler.pkl")

metrics = {
    "accuracy": round(float(accuracy), 4),
    "f1": round(float(f1), 4),
    "precision": round(float(precision_score(y_test, y_pred, average='weighted')), 4),
    "recall": round(float(recall_score(y_test, y_pred, average='weighted')), 4),
    "class_labels": list(label_encoder.classes_),
    "confusion_matrix": cm.tolist(),
    "feature_importances": {
        col: round(float(imp), 4)
        for col, imp in zip(feature_cols, model.feature_importances_)
    }
}

with open('metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print("Saved metrics.json")
print(json.dumps(metrics, indent=2))
"""


# ──────────────────────────────────────────────
# CELL 10 — Download files
# ──────────────────────────────────────────────
"""
from google.colab import files

files.download('model_v1.pkl')
files.download('scaler.pkl')
files.download('metrics.json')
files.download('model_evaluation.png')
"""
