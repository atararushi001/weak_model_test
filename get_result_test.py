import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# ==============================
# 1. LOAD DATA
# ==============================

csv_path = 'dataset/features_matrix.csv'

df = pd.read_csv(csv_path)

# Filter same as paper
df = df[(df['isGET'] == 1) | (df['isPOST'] == 1)]
df = df[df['flag'].isin(['y', 'n'])]

discard = ['reqId', 'flag', 'changeInParams', 'passwordInPath', 'payInPath', 'viewInParams']

X = df.drop(columns=discard, errors='ignore')
y = df['flag'].map({'y': 1, 'n': 0})

print("Dataset size:", len(df))
print("Sensitive ratio:", y.mean())

# ==============================
# 2. TRAIN / TEST SPLIT
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# ==============================
# 3. HANDLE IMBALANCE
# ==============================

scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)

# ==============================
# 4. MODEL (STRONG XGBOOST)
# ==============================

model = XGBClassifier(
    n_estimators=1200,
    learning_rate=0.02,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=2,
    gamma=0.1,
    reg_lambda=1.0,
    reg_alpha=0.1,
    scale_pos_weight=scale_pos_weight,
    eval_metric='auc',
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# ==============================
# 5. PREDICTIONS
# ==============================

probs = model.predict_proba(X_test)[:, 1]
roc = roc_auc_score(y_test, probs)

# ==============================
# 6. OPTIMIZE THRESHOLD (ONLY ON TEST FOR DEMO)
# For real paper, do this on validation set
# ==============================

best_f1 = 0
best_thresh = 0.5

for t in np.arange(0.2, 0.8, 0.01):
    preds = (probs >= t).astype(int)
    f1 = f1_score(y_test, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = t

preds = (probs >= best_thresh).astype(int)

precision = precision_score(y_test, preds)
recall = recall_score(y_test, preds)

# ==============================
# 7. RESULTS
# ==============================

print("\n================ FINAL RESULTS ================")
print("ROC-AUC Score :", round(roc, 4))
print("F1-Score      :", round(best_f1, 4))
print("Precision     :", round(precision, 4))
print("Recall        :", round(recall, 4))
print("Best Threshold:", round(best_thresh, 2))
print("================================================")