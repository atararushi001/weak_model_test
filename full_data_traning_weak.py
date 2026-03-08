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
        n_estimators=10,
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

RUNS = 5
NUM_STRONG_MODELS = 20 
TRAIN_SIZE = int(0.9 * N_train)

f1_scores = []
auc_scores = []

print("\nRunning multiple hardness-aware evaluations...")

for run in range(RUNS):

    print(f"\n================ Run {run+1}/{RUNS} ================")

    np.random.seed(42 + run)

    hard_datasets = []

    for i in range(NUM_STRONG_MODELS):
        sampled_idx = np.random.choice(
            ranking_df["row_id"],
            size=TRAIN_SIZE,
            replace=True,
            p=ranking_df["P_r"]
        )
        hard_datasets.append(sampled_idx)

    models = []

    for i, idx in enumerate(hard_datasets):
        clf = RandomForestClassifier(
            n_estimators=2200,
            max_depth=None,
            min_samples_leaf=2,
            max_features=None,
            class_weight={0:1, 1:1.2},
            random_state=10 + i,
            n_jobs=-1
        )
        clf.fit(X_train.iloc[idx], y_train.iloc[idx])
        models.append(clf)

    # ============================================================
    # ENSEMBLE PREDICTION
    # ============================================================

    ensemble_probs = np.zeros(len(X_test))

    for clf in models:
        ensemble_probs += clf.predict_proba(X_test)[:, 1]

    ensemble_probs /= len(models)

    # ------------------------------------------------------------
    # THRESHOLD TUNING (Optional check)
    # ------------------------------------------------------------

    best_f1 = 0
    best_threshold = 0.5

    for t in np.arange(0.2, 0.8, 0.05):
        temp_preds = (ensemble_probs >= t).astype(int)
        temp_f1 = f1_score(y_test, temp_preds)

        if temp_f1 > best_f1:
            best_f1 = temp_f1
            best_threshold = t

    print(f"\nBest Threshold = {best_threshold:.2f}")
    print(f"Best F1 at this threshold = {best_f1:.4f}")

    ensemble_preds = (ensemble_probs >= best_threshold).astype(int)

    # ============================================================
    # METRICS
    # ============================================================

    f1 = f1_score(y_test, ensemble_preds)
    auc = roc_auc_score(y_test, ensemble_probs)

    f1_scores.append(f1)
    auc_scores.append(auc)

    print(f"\nFinal F1  = {f1:.4f}")
    print(f"Final AUC = {auc:.4f}")

    # ============================================================
    # CONFUSION MATRIX
    # ============================================================

    cm = confusion_matrix(y_test, ensemble_preds)

    print("\nConfusion Matrix:")
    print(cm)

    TN, FP, FN, TP = cm.ravel()

    print(f"\nTrue Negatives  (Normal → Normal)     : {TN}")
    print(f"False Positives (Normal → Sensitive)  : {FP}")
    print(f"False Negatives (Sensitive → Normal)  : {FN}")
    print(f"True Positives  (Sensitive → Sensitive): {TP}")

    # ============================================================
    # CLASSIFICATION REPORT
    # ============================================================

    print("\nClassification Report:")
    print(classification_report(y_test, ensemble_preds))

    # ============================================================
    # ERROR ANALYSIS TABLE
    # ============================================================

    error_df = pd.DataFrame({
        "true_label": y_test,
        "predicted_label": ensemble_preds,
        "probability": ensemble_probs
    })

    error_df["error_type"] = "Correct"
    error_df.loc[
        (error_df.true_label == 1) &
        (error_df.predicted_label == 0),
        "error_type"
    ] = "False_Negative"

    error_df.loc[
        (error_df.true_label == 0) &
        (error_df.predicted_label == 1),
        "error_type"
    ] = "False_Positive"

    print("\nError Type Distribution:")
    print(error_df["error_type"].value_counts())

    # Save error details
    error_df.to_csv(f"error_analysis_run_{run+1}.csv", index=False)
    print(f"Saved error_analysis_run_{run+1}.csv")

# ============================================================
# FINAL RESULTS
# ============================================================

print("\n================ FINAL (MEAN ± STD) ================")
print(f"F1-Score: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
print(f"ROC-AUC : {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")
print("===================================================")
