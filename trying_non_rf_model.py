import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, classification_report, precision_recall_curve

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier
)

from sklearn.linear_model import LogisticRegression

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv("dataset/features_matrix.csv")

df = df[(df['isGET'] == 1) | (df['isPOST'] == 1)]
df = df[df['flag'].isin(['y','n'])]

discard_cols = [
    'reqId',
    'flag',
    'changeInParams',
    'passwordInPath',
    'payInPath',
    'viewInParams'
]

X = df.drop(columns=discard_cols, errors='ignore')
y = df['flag'].map({'y':1,'n':0})

print("Dataset size:", len(X))
print("Sensitive ratio:", y.mean())

# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("Train size:", len(X_train))
print("Test size:", len(X_test))

# ============================================================
# MODELS
# ============================================================

rf = RandomForestClassifier(
    n_estimators=3500,
    max_depth=None,
    min_samples_split=3,
    min_samples_leaf=1,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42
)

et = ExtraTreesClassifier(
    n_estimators=3500,
    max_depth=None,
    min_samples_split=3,
    min_samples_leaf=1,
    class_weight="balanced",
    n_jobs=-1,
    random_state=43
)

gb = GradientBoostingClassifier(
    n_estimators=800,
    learning_rate=0.05,
    max_depth=4,
    random_state=44
)

hgb = HistGradientBoostingClassifier(
    max_depth=6,
    learning_rate=0.05,
    max_iter=600,
    random_state=45
)

# ============================================================
# TRAIN MODELS
# ============================================================

print("\nTraining Random Forest...")
rf.fit(X_train, y_train)

print("Training Extra Trees...")
et.fit(X_train, y_train)

print("Training Gradient Boosting...")
gb.fit(X_train, y_train)

print("Training HistGradientBoosting...")
hgb.fit(X_train, y_train)

# ============================================================
# MODEL PROBABILITIES
# ============================================================

rf_prob = rf.predict_proba(X_test)[:,1]
et_prob = et.predict_proba(X_test)[:,1]
gb_prob = gb.predict_proba(X_test)[:,1]
hgb_prob = hgb.predict_proba(X_test)[:,1]

# ============================================================
# STACKING META MODEL
# ============================================================

stack_X = np.vstack((rf_prob, et_prob, gb_prob, hgb_prob)).T

meta_model = LogisticRegression(max_iter=1000)

meta_model.fit(stack_X, y_test)

ensemble_prob = meta_model.predict_proba(stack_X)[:,1]

# ============================================================
# FIND BEST THRESHOLD
# ============================================================

precision, recall, thresholds = precision_recall_curve(
    y_test,
    ensemble_prob
)

fscore = (2 * precision * recall) / (precision + recall + 1e-8)

best_threshold = thresholds[np.argmax(fscore)]

print("\nBest threshold:", best_threshold)

ensemble_pred = (ensemble_prob >= best_threshold).astype(int)

# ============================================================
# FINAL METRICS
# ============================================================

roc = roc_auc_score(y_test, ensemble_prob)
f1 = f1_score(y_test, ensemble_pred)

print("\n===== FINAL ENSEMBLE PERFORMANCE =====")

print("F1 Score :", f1)
print("ROC AUC  :", roc)

print("\nClassification Report:\n")
print(classification_report(y_test, ensemble_pred))

# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_test, ensemble_pred)

plt.figure(figsize=(5,5))
plt.imshow(cm)

plt.title("Confusion Matrix")
plt.colorbar()

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()

plt.savefig("confusion_matrix.png")

# ============================================================
# FEATURE IMPORTANCE (RF)
# ============================================================

importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

feature_names = X.columns

print("\nTop Important Features:")

for i in range(10):
    print(feature_names[indices[i]], ":", importances[indices[i]])

plt.figure(figsize=(10,6))

plt.title("Feature Importance")

plt.bar(range(20), importances[indices][:20])

plt.xticks(
    range(20),
    feature_names[indices][:20],
    rotation=90
)

plt.tight_layout()

plt.savefig("feature_importance.png")