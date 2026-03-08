import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score

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

X = df.drop(columns=discard_cols, errors='ignore')
y = df['flag'].map({'y': 1, 'n': 0})

print("Dataset size:", len(X))
print("Sensitive ratio:", y.mean())

# ============================================================
# STEP 2: SPLIT FIRST (80/20)
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

X_train = X_train.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)

X_test = X_test.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

print("\nTrain size (before balancing):", len(X_train))
print("Test size :", len(X_test))

# ============================================================
# STEP 3: BALANCE TRAINING DATA
# Keep ALL sensitive (y=1), random sample non-sensitive (y=0)
# to match the count of sensitive samples
# ============================================================

sensitive_idx = y_train[y_train == 1].index.tolist()
non_sensitive_idx = y_train[y_train == 0].index.tolist()

n_sensitive = len(sensitive_idx)

# Randomly sample non-sensitive equal to sensitive count
np.random.seed(42)
sampled_non_sensitive_idx = np.random.choice(
    non_sensitive_idx,
    size=n_sensitive,
    replace=False
).tolist()

# Combine and shuffle
balanced_idx = sensitive_idx + sampled_non_sensitive_idx
np.random.shuffle(balanced_idx)

X_train_bal = X_train.iloc[balanced_idx].reset_index(drop=True)
y_train_bal = y_train.iloc[balanced_idx].reset_index(drop=True)

N_train_bal = len(X_train_bal)

print(f"\nSensitive samples   : {n_sensitive}")
print(f"Non-sensitive sampled: {len(sampled_non_sensitive_idx)}")
print(f"Balanced train size  : {N_train_bal}")
print(f"Balanced ratio       : {y_train_bal.mean():.4f}  (should be ~0.50)")

# ============================================================
# STEP 4: TRAIN WEAK MODELS ON BALANCED DATA
# ============================================================

NUM_WEAK_MODELS = 10

correctness_matrix = np.zeros((N_train_bal, NUM_WEAK_MODELS))
weak_auc_scores = []

print("\nTraining weak models on balanced data...")

for i in range(NUM_WEAK_MODELS):

    clf = RandomForestClassifier(
        n_estimators=10,
        max_depth=3,
        min_samples_leaf=10,
        max_features='sqrt',
        random_state=10 + i,
        n_jobs=-1
    )

    clf.fit(X_train_bal, y_train_bal)

    probs = clf.predict_proba(X_train_bal)[:, 1]
    preds = (probs >= 0.5).astype(int)

    correctness_matrix[:, i] = (preds == y_train_bal).astype(int)

    auc_i = roc_auc_score(y_train_bal, probs)
    weak_auc_scores.append(auc_i)

    print(f"Weak Model {i+1} Train AUC = {auc_i:.4f}")

print("\nMean Weak AUC (Train) = {:.4f}".format(np.mean(weak_auc_scores)))

# ============================================================
# STEP 5: HARDNESS BASED ON CORRECTNESS
# 0 = HARDEST, 10 = EASIEST
# ============================================================

correct_count = correctness_matrix.sum(axis=1)

ranking_df = pd.DataFrame({
    "row_id": np.arange(N_train_bal),
    "true_label": y_train_bal,
    "rank_0_10": correct_count
})

print("\nRank distribution (0 = hard, 10 = easy):")
print(ranking_df["rank_0_10"].value_counts().sort_index())