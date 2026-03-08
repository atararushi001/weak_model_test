import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, classification_report
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

np.random.seed(42)

# ============================================================
# STEP 1: LOAD DATA
# ============================================================

df = pd.read_csv("dataset/features_matrix.csv")
df = df[(df['isGET'] == 1) | (df['isPOST'] == 1)]
df = df[df['flag'].isin(['y', 'n'])]

discard_cols = [
    'reqId', 'flag',
    'changeInParams',
    'passwordInPath',
    'payInPath',
    'viewInParams'
]

X = df.drop(columns=discard_cols, errors='ignore').fillna(0)
y = df['flag'].map({'y': 1, 'n': 0})

print("Dataset size  :", len(X))
print("Sensitive ratio:", round(y.mean(), 4))

# ============================================================
# STEP 2: TRAIN/TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.1,
    stratify=y,
    random_state=42
)

X_train = X_train.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
X_test  = X_test.reset_index(drop=True)
y_test  = y_test.reset_index(drop=True)

N_train = len(X_train)

# Compute scale_pos_weight for XGBoost (paper-recommended)
neg_count  = (y_train == 0).sum()
pos_count  = (y_train == 1).sum()
spw        = neg_count / pos_count

print(f"\nTrain size     : {N_train}")
print(f"Test size      : {len(X_test)}")
print(f"scale_pos_weight: {spw:.2f}")

# ============================================================
# STEP 3: OOF STACKING
# SMOTE-Tomek inside each fold (paper-backed best practice)
# ============================================================

SKF = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Define base models - paper confirmed: XGBoost best, ET second
base_models = [

    ("xgb_tuned_1", XGBClassifier(
        n_estimators=800, max_depth=4, learning_rate=0.05,
        subsample=0.8,    colsample_bytree=0.6,
        min_child_weight=5, gamma=1,
        scale_pos_weight=spw,          # paper: use with imbalance
        eval_metric='auc',
        random_state=1, n_jobs=-1
    )),

    ("xgb_tuned_2", XGBClassifier(
        n_estimators=800, max_depth=5, learning_rate=0.03,
        subsample=0.7,    colsample_bytree=0.8,
        min_child_weight=3, gamma=0.5,
        scale_pos_weight=spw,
        eval_metric='auc',
        random_state=2, n_jobs=-1
    )),

    ("xgb_tuned_3", XGBClassifier(
        n_estimators=600, max_depth=6, learning_rate=0.02,
        subsample=0.9,    colsample_bytree=0.7,
        min_child_weight=1, gamma=0.1,
        scale_pos_weight=spw,
        eval_metric='auc',
        random_state=3, n_jobs=-1
    )),

    ("xgb_smote_1", XGBClassifier(   # XGBoost on SMOTE-Tomek balanced (no spw needed)
        n_estimators=800, max_depth=5, learning_rate=0.05,
        subsample=0.8,    colsample_bytree=0.8,
        min_child_weight=3, gamma=0.5,
        eval_metric='auc',
        random_state=4, n_jobs=-1
    )),

    ("xgb_smote_2", XGBClassifier(
        n_estimators=600, max_depth=4, learning_rate=0.1,
        subsample=0.8,    colsample_bytree=0.6,
        min_child_weight=5, gamma=1,
        eval_metric='auc',
        random_state=5, n_jobs=-1
    )),

    ("et_balanced", ExtraTreesClassifier(
        n_estimators=1000, max_depth=None,
        min_samples_leaf=1, max_features='sqrt',
        class_weight='balanced',       # paper: ET + balanced
        random_state=6, n_jobs=-1
    )),

    ("et_smote", ExtraTreesClassifier(
        n_estimators=1000, max_depth=None,
        min_samples_leaf=1, max_features=0.5,
        class_weight='balanced',
        random_state=7, n_jobs=-1
    )),

    ("rf_balanced", RandomForestClassifier(
        n_estimators=1000, max_depth=None,
        min_samples_leaf=1, max_features='sqrt',
        class_weight='balanced_subsample',
        random_state=8, n_jobs=-1
    )),
]

oof_matrix  = np.zeros((N_train, len(base_models)))
test_matrix = np.zeros((len(X_test), len(base_models)))

print("\nTraining base models (OOF + SMOTE-Tomek inside folds)...")
print("="*55)

for m_idx, (name, model) in enumerate(base_models):

    oof_probs  = np.zeros(N_train)
    test_folds = np.zeros(len(X_test))

    use_smote = "smote" in name   # SMOTE-Tomek only for smote-tagged models

    for fold_idx, (tr_idx, val_idx) in enumerate(SKF.split(X_train, y_train)):

        X_f = X_train.iloc[tr_idx]
        y_f = y_train.iloc[tr_idx]

        if use_smote:
            # Paper-backed: SMOTE-Tomek inside fold only
            smt = SMOTETomek(random_state=42 + fold_idx)
            X_f, y_f = smt.fit_resample(X_f, y_f)

        model.fit(X_f, y_f)

        oof_probs[val_idx] = model.predict_proba(
            X_train.iloc[val_idx]
        )[:, 1]

        test_folds += model.predict_proba(X_test)[:, 1] / SKF.n_splits

    oof_matrix[:, m_idx]  = oof_probs
    test_matrix[:, m_idx] = test_folds

    fold_auc = roc_auc_score(y_train, oof_probs)
    print(f"  [{name:20s}] OOF AUC = {fold_auc:.4f}")

# ============================================================
# STEP 4: AUC-WEIGHTED BLEND (paper: weight by model quality)
# ============================================================

oof_aucs = np.array([
    roc_auc_score(y_train, oof_matrix[:, i])
    for i in range(len(base_models))
])

weights        = oof_aucs / oof_aucs.sum()
weighted_probs = test_matrix @ weights
avg_probs      = test_matrix.mean(axis=1)

print("\n--- Model weights (AUC-based) ---")
for (name, _), auc, w in zip(base_models, oof_aucs, weights):
    print(f"  {name:20s}  OOF AUC={auc:.4f}  weight={w:.4f}")

# ============================================================
# STEP 5: THRESHOLD TUNING ON OOF PROBS (no leakage)
# ============================================================

oof_weighted = oof_matrix @ weights

best_thresh = 0.5
best_f1     = 0.0

for thresh in np.arange(0.05, 0.95, 0.01):
    preds = (oof_weighted >= thresh).astype(int)
    f1_t  = f1_score(y_train, preds)
    if f1_t > best_f1:
        best_f1     = f1_t
        best_thresh = thresh

print(f"\nBest threshold (OOF): {best_thresh:.2f}  OOF F1: {best_f1:.4f}")

# ============================================================
# STEP 6: FINAL EVALUATION
# ============================================================

auc_weighted = roc_auc_score(y_test, weighted_probs)
auc_avg      = roc_auc_score(y_test, avg_probs)

f1_default   = f1_score(y_test, (weighted_probs >= 0.50).astype(int))
f1_best      = f1_score(y_test, (weighted_probs >= best_thresh).astype(int))

print("\n================ FINAL RESULTS ================")
print(f"Simple Avg AUC         : {auc_avg:.4f}")
print(f"Weighted Blend AUC     : {auc_weighted:.4f}")
print(f"F1 (threshold=0.50)    : {f1_default:.4f}")
print(f"F1 (best thresh={best_thresh:.2f})  : {f1_best:.4f}")
print("================================================")
print("\nDetailed Report:")
print(classification_report(
    y_test,
    (weighted_probs >= best_thresh).astype(int),
    target_names=['non-sensitive', 'sensitive']
))