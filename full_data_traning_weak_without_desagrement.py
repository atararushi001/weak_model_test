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
# STEP 2: SPLIT FIRST
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.1,
    stratify=y,
    random_state=42
)

X_train = X_train.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)

X_test = X_test.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

N_train = len(X_train)

print("\nTrain size:", N_train)
print("Test size :", len(X_test))

# ============================================================
# STEP 3: TRAIN WEAK MODELS (FULL TRAIN)
# ============================================================

NUM_WEAK_MODELS = 10

correctness_matrix = np.zeros((N_train, NUM_WEAK_MODELS))
weak_auc_scores = []

print("\nTraining weak models...")

for i in range(NUM_WEAK_MODELS):

    clf = RandomForestClassifier(
        n_estimators=5,
        max_depth=3,
        min_samples_leaf=10,
        max_features='sqrt',
        random_state=10 + i,
        n_jobs=-1
    )

    clf.fit(X_train, y_train)

    probs = clf.predict_proba(X_train)[:, 1]
    preds = (probs >= 0.5).astype(int)

    correctness_matrix[:, i] = (preds == y_train).astype(int)

    auc_i = roc_auc_score(y_train, probs)
    weak_auc_scores.append(auc_i)

    print(f"Weak Model {i+1} Train AUC = {auc_i:.4f}")

print("\nMean Weak AUC (Train) = {:.4f}".format(np.mean(weak_auc_scores)))

# ============================================================
# STEP 4: HARDNESS BASED ON CORRECTNESS
# 0 = HARDEST
# 10 = EASIEST
# ============================================================

correct_count = correctness_matrix.sum(axis=1)

rank_0_10 = correct_count  # directly number of correct models

ranking_df = pd.DataFrame({
    "row_id": np.arange(N_train),
    "true_label": y_train,
    "rank_0_10": rank_0_10
})

print("\nRank distribution (0 = hard, 10 = easy):")
print(ranking_df["rank_0_10"].value_counts().sort_index())

# # ============================================================
# # STEP 5: PDF SAMPLING
# # ============================================================

# D = 100
# x = 1.0

# ranking_df = ranking_df.sort_values("rank_0_10").reset_index(drop=True)
# ranking_df["r"] = np.arange(1, N_train + 1)

# ranking_df["P_r"] = D * np.exp(-(ranking_df["r"] ** x) / N_train)
# ranking_df["P_r"] /= ranking_df["P_r"].sum()

# # ============================================================
# # STEP 6: HARDNESS-AWARE ENSEMBLE
# # ============================================================

# RUNS = 5
# NUM_STRONG_MODELS = 5
# TRAIN_SIZE = int(0.9 * N_train)

# f1_scores = []
# auc_scores = []

# print("\nRunning hardness-aware ensemble...")

# for run in range(RUNS):

#     np.random.seed(42 + run)

#     hard_datasets = []

#     for i in range(NUM_STRONG_MODELS):
#         sampled_idx = np.random.choice(
#             ranking_df["row_id"],
#             size=TRAIN_SIZE,
#             replace=True,
#             p=ranking_df["P_r"]
#         )
#         hard_datasets.append(sampled_idx)

#     models = []

#     for i, idx in enumerate(hard_datasets):
#         clf = RandomForestClassifier(
#         n_estimators=1500,
#         max_depth=None,
#         min_samples_leaf=1,
#         max_features=None,   
#         class_weight="balanced_subsample",
#         random_state=1000 + i,
#         n_jobs=-1
#         )
#         clf.fit(X_train.iloc[idx], y_train.iloc[idx])
#         models.append(clf)

#     ensemble_probs = np.zeros(len(X_test))

#     for clf in models:
#         ensemble_probs += clf.predict_proba(X_test)[:, 1]

#     ensemble_probs /= len(models)
#     ensemble_preds = (ensemble_probs >= 0.5).astype(int)

#     f1_scores.append(f1_score(y_test, ensemble_preds))
#     auc_scores.append(roc_auc_score(y_test, ensemble_probs))

#     print(f"Run {run+1}: F1={f1_scores[-1]:.4f}, AUC={auc_scores[-1]:.4f}")

# # ============================================================
# # FINAL RESULTS
# # ============================================================

# print("\n================ FINAL (MEAN ± STD) ================")
# print(f"F1-Score: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
# print(f"ROC-AUC :  {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")
# print("===================================================")