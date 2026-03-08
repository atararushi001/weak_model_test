import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
# STEP 2: SPLIT FIRST (NO LEAKAGE)
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,        # 80/20 split
    stratify=y,
    random_state=42
)

X_train = X_train.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

N_train = len(X_train)
N_test = len(X_test)

# ============================================================
# PRINT DETAILED SPLIT STATISTICS
# ============================================================

total = N_train + N_test

train_sensitive = y_train.sum()
train_normal    = (y_train == 0).sum()
test_sensitive  = y_test.sum()
test_normal     = (y_test == 0).sum()

print("\n" + "="*55)
print("           DATASET SPLIT STATISTICS")
print("="*55)

print(f"\n{'OVERALL':}")
print(f"  Total requests         : {total}")
print(f"  Total sensitive        : {int(y.sum())} ({y.mean()*100:.2f}%)")
print(f"  Total normal           : {int((y==0).sum())} ({(1-y.mean())*100:.2f}%)")

print(f"\n{'TRAINING SET (80%)':}")
print(f"  Total requests         : {N_train} ({N_train/total*100:.1f}%)")
print(f"  Sensitive requests     : {int(train_sensitive)} ({train_sensitive/N_train*100:.2f}%)")
print(f"  Normal requests        : {int(train_normal)} ({train_normal/N_train*100:.2f}%)")

print(f"\n{'TESTING SET (20%)':}")
print(f"  Total requests         : {N_test} ({N_test/total*100:.1f}%)")
print(f"  Sensitive requests     : {int(test_sensitive)} ({test_sensitive/N_test*100:.2f}%)")
print(f"  Normal requests        : {int(test_normal)} ({test_normal/N_test*100:.2f}%)")

print("="*55)

# ============================================================
# SAVE TRAIN/TEST TO CSV (two sheets in one Excel file)
# ============================================================

train_df = X_train.copy()
train_df['label'] = y_train
train_df['split'] = 'train'

test_df = X_test.copy()
test_df['label'] = y_test
test_df['split'] = 'test'

with pd.ExcelWriter("train_test_split.xlsx", engine='openpyxl') as writer:
    train_df.to_excel(writer, sheet_name='Training', index=False)
    test_df.to_excel(writer, sheet_name='Testing', index=False)

print("\nSaved train_test_split.xlsx with Training and Testing sheets")

# ============================================================
# STEP 3: TRAIN WEAK MODELS ON FULL TRAIN DATA
# ============================================================

NUM_WEAK_MODELS = 10

correctness_matrix = np.zeros((N_train, NUM_WEAK_MODELS))
prob_matrix = np.zeros((N_train, NUM_WEAK_MODELS))

weak_auc_scores = []

print("\nTraining weak models on FULL training data...")

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

    probs = clf.predict_proba(X_train)[:, 1]
    preds = (probs >= 0.5).astype(int)

    prob_matrix[:, i] = probs
    correctness_matrix[:, i] = (preds == y_train).astype(int)

    auc_i = roc_auc_score(y_train, probs)
    weak_auc_scores.append(auc_i)

    print(f"Weak Model {i+1} Train AUC = {auc_i:.4f}")

print("\nMean Weak AUC (Train) = {:.4f}".format(np.mean(weak_auc_scores)))
print("Std  Weak AUC (Train) = {:.4f}".format(np.std(weak_auc_scores)))

# ============================================================
# STEP 4: HARDNESS USING DISAGREEMENT
# 0 = Hardest
# 10 = Easiest
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

print("\nRank distribution (0 = hard, 10 = easy):")
print(ranking_df["rank_0_10"].value_counts().sort_index())

# ============================================================
# STEP 5: PDF SAMPLING FUNCTION
# ============================================================

D = 80
x = 0.9

ranking_df = ranking_df.sort_values("rank_0_10").reset_index(drop=True)
ranking_df["r"] = np.arange(1, N_train + 1)

ranking_df["P_r"] = D * np.exp(-(ranking_df["r"] ** x) / N_train)
ranking_df["P_r"] /= ranking_df["P_r"].sum()

# Optional visualization
plt.figure(figsize=(8, 5))
plt.plot(ranking_df["r"], ranking_df["P_r"])
plt.title("Hardness-Based Sampling Probability")
plt.xlabel("Rank (Hard → Easy)")
plt.ylabel("P(r)")
plt.grid(True)
plt.savefig("sampling_curve.png")
plt.close()
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve
)

# ============================================================
# STEP 6: MULTIPLE HARDNESS-AWARE ENSEMBLE RUNS
# ============================================================
# ============================================================
# STEP 6: TRAIN 5 HARDNESS MODELS (NO ENSEMBLE)
# ============================================================

NUM_MODELS = 10
TRAIN_SIZE = int(0.9 * N_train)

print("\nTraining 5 hardness-based models...")

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

    # ======================================================
    # PREDICTIONS
    # ======================================================

    probs = clf.predict_proba(X_test)[:,1]
    preds = (probs >= 0.5).astype(int)

    # ======================================================
    # METRICS
    # ======================================================

    f1 = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)

    print("F1 :", round(f1,4))
    print("AUC:", round(auc,4))

    # ======================================================
    # CONFUSION MATRIX
    # ======================================================

    cm = confusion_matrix(y_test, preds)

    TN, FP, FN, TP = cm.ravel()

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    specificity = TN / (TN + FP)

    fpr = FP / (FP + TN)
    fnr = FN / (FN + TP)

    print("\nConfusion Matrix")
    print(cm)

    print("\nTrue Positive:", TP)
    print("True Negative:", TN)
    print("False Positive:", FP)
    print("False Negative:", FN)

    print("\nPrecision:", precision)
    print("Recall:", recall)
    print("Specificity:", specificity)

    print("False Positive Rate:", fpr)
    print("False Negative Rate:", fnr)

    print("\nClassification Report")
    print(classification_report(y_test, preds))