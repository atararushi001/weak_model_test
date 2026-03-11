import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

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
# STEP 2: TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.1,
    stratify=y,
    random_state=None
)

X_train = X_train.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)

X_test = X_test.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

N_train = len(X_train)
N_test = len(X_test)

# ============================================================
# PRINT DATASET STATISTICS
# ============================================================

print("\n================ DATASET SPLIT =================")

print("Train size:", N_train)
print("Test size :", N_test)

print("Train sensitive:", y_train.sum())
print("Test sensitive :", y_test.sum())

print("===============================================")

# ============================================================
# STEP 3: TRAIN WEAK MODELS
# ============================================================

NUM_WEAK_MODELS = 10

correctness_matrix = np.zeros((N_train, NUM_WEAK_MODELS))
prob_matrix = np.zeros((N_train, NUM_WEAK_MODELS))

weak_auc_scores = []

print("\nTraining weak models...")

for i in range(NUM_WEAK_MODELS):

    clf = RandomForestClassifier(
        n_estimators=3,
        max_depth=3,
        min_samples_leaf=10,
        max_features='sqrt',
        random_state=10 + i,
        n_jobs=-1
    )

    clf.fit(X_train, y_train)

    probs = clf.predict_proba(X_train)[:,1]
    preds = (probs >= 0.5).astype(int)

    prob_matrix[:, i] = probs
    correctness_matrix[:, i] = (preds == y_train).astype(int)

    auc_i = roc_auc_score(y_train, probs)
    weak_auc_scores.append(auc_i)

    print(f"Weak Model {i+1} Train AUC = {auc_i:.4f}")

print("\nMean Weak AUC =", np.mean(weak_auc_scores))
print("Std Weak AUC =", np.std(weak_auc_scores))

# ============================================================
# STEP 4: HARDNESS (STD DISAGREEMENT)
# ============================================================

std_prob = prob_matrix.std(axis=1)

rank_0_10 = np.clip(
    np.round(
        10 - 10 * (std_prob - std_prob.min()) /
        (std_prob.max() - std_prob.min())
    ),
    0, 10
).astype(int)

ranking_df = pd.DataFrame({
    "row_id": np.arange(N_train),
    "true_label": y_train,
    "std_prob": std_prob,
    "rank_0_10": rank_0_10
})

print("\nRank distribution")
print(ranking_df["rank_0_10"].value_counts().sort_index())

# ============================================================
# SAVE CSV WITH PROBABILITIES + STD + RANK
# ============================================================

details_df = pd.DataFrame()

details_df["row_id"] = np.arange(N_train)
details_df["true_label"] = y_train
details_df["std_prob"] = std_prob
details_df["rank_0_10"] = rank_0_10

for i in range(NUM_WEAK_MODELS):
    details_df[f"prob_model_{i+1}"] = prob_matrix[:,i]

details_df.to_csv("request_hardness_details.csv", index=False)

print("\nSaved request_hardness_details.csv")

# ============================================================
# STD HISTOGRAM (ALL REQUESTS)
# ============================================================

plt.figure(figsize=(8,5))

plt.hist(std_prob, bins=40)

plt.title("STD Distribution of Weak Model Probabilities")
plt.xlabel("STD")
plt.ylabel("Number of Requests")

plt.savefig("std_histogram_all_requests.png")
plt.close()

print("Saved std_histogram_all_requests.png")

# ============================================================
# STD HISTOGRAM (Sensitive vs Normal)
# ============================================================

plt.figure(figsize=(8,5))

plt.hist(std_prob[y_train==0], bins=40, alpha=0.5, label="Normal")
plt.hist(std_prob[y_train==1], bins=40, alpha=0.5, label="Sensitive")

plt.legend()
plt.title("STD Distribution (Sensitive vs Normal)")
plt.xlabel("STD")
plt.ylabel("Number of Requests")

plt.savefig("std_histogram_sensitive_vs_normal.png")
plt.close()

print("Saved std_histogram_sensitive_vs_normal.png")

# ============================================================
# STEP 5: HARDNESS SAMPLING
# ============================================================

D = 80
x = 0.9

ranking_df = ranking_df.sort_values("rank_0_10").reset_index(drop=True)
ranking_df["r"] = np.arange(1, N_train + 1)

ranking_df["P_r"] = D * np.exp(-(ranking_df["r"] ** x) / N_train)
ranking_df["P_r"] /= ranking_df["P_r"].sum()
# ============================================================
# STEP 6: TRAIN HARDNESS MODELS
# ============================================================
# ============================================================
# STEP 6: TRAIN HARDNESS MODELS
# ============================================================

NUM_MODELS = 30
TRAIN_SIZE = int(0.9 * N_train)

print("\nTraining hardness-based models...")

for m in range(NUM_MODELS):

    print(f"\n================ MODEL {m+1} ================")

    np.random.seed(42 + m)

    sampled_idx = np.random.choice(
        ranking_df["row_id"],
        size=TRAIN_SIZE,
        replace=True,
        p=ranking_df["P_r"]
    )

    clf = RandomForestClassifier(
        n_estimators=2500,
        max_depth=None,
        min_samples_leaf=2,
        max_features=None,
        class_weight="balanced",
        random_state=100 + m,
        n_jobs=-1
    )

    clf.fit(
        X_train.iloc[sampled_idx],
        y_train.iloc[sampled_idx]
    )

    # ========================================================
    # TRAINING PERFORMANCE (SAMPLED DATA)
    # ========================================================

    train_sample_probs = clf.predict_proba(
        X_train.iloc[sampled_idx]
    )[:,1]

    train_sample_preds = (train_sample_probs >= 0.5).astype(int)

    train_sample_auc = roc_auc_score(
        y_train.iloc[sampled_idx],
        train_sample_probs
    )

    print("\n--- Training (Sampled Data) ---")
    print("AUC:", round(train_sample_auc,4))

    cm = confusion_matrix(
        y_train.iloc[sampled_idx],
        train_sample_preds
    )

    TN, FP, FN, TP = cm.ravel()

    print("Confusion Matrix")
    print(cm)

    print("True Positive:", TP)
    print("True Negative:", TN)
    print("False Positive:", FP)
    print("False Negative:", FN)

    # ========================================================
    # TRAINING PERFORMANCE (FULL TRAIN DATA)
    # ========================================================

    train_full_probs = clf.predict_proba(X_train)[:,1]
    train_full_preds = (train_full_probs >= 0.5).astype(int)

    train_full_auc = roc_auc_score(y_train, train_full_probs)

    print("\n--- Training (Full Dataset) ---")
    print("AUC:", round(train_full_auc,4))

    cm = confusion_matrix(y_train, train_full_preds)

    TN, FP, FN, TP = cm.ravel()

    print("Confusion Matrix")
    print(cm)

    print("True Positive:", TP)
    print("True Negative:", TN)
    print("False Positive:", FP)
    print("False Negative:", FN)

    # ========================================================
    # TEST PERFORMANCE
    # ========================================================

    test_probs = clf.predict_proba(X_test)[:,1]
    test_preds = (test_probs >= 0.5).astype(int)

    test_auc = roc_auc_score(y_test, test_probs)
    f1 = f1_score(y_test, test_preds)

    print("\n--- Testing ---")
    print("AUC:", round(test_auc,4))
    print("F1 :", round(f1,4))

    cm = confusion_matrix(y_test, test_preds)

    TN, FP, FN, TP = cm.ravel()

    print("Confusion Matrix")
    print(cm)

    print("True Positive:", TP)
    print("True Negative:", TN)
    print("False Positive:", FP)
    print("False Negative:", FN)

    print("\nClassification Report")
    print(classification_report(y_test, test_preds))