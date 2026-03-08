import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# ============================================================
# STEP 1: LOAD DATA
# ============================================================

df = pd.read_csv("dataset/features_matrix.csv")

df = df[(df["isGET"] == 1) | (df["isPOST"] == 1)]
df = df[df["flag"].isin(["y", "n"])]

X = df.drop(columns=[
    "reqId", "flag",
    "changeInParams",
    "passwordInPath",
    "payInPath",
    "viewInParams"
], errors="ignore").fillna(0)

y = df["flag"].map({"y": 1, "n": 0})

print("Total samples:", len(X))
print("Sensitive ratio:", y.mean())

# ============================================================
# STEP 2: TRAIN / TEST SPLIT (NO LEAKAGE)
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

N_train = len(X_train)
print("Train size:", N_train)
print("Test size :", len(X_test))

# ============================================================
# STEP 3: BUILD DECISION MATRIX (OOF, PAPER STYLE)
# ============================================================

N0 = 10   # number of weak models
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

decision_matrix = np.zeros((N0, N_train))
weak_auc_scores = []

print("\nTraining weak models (paper-correct OOF)...")

for i in range(N0):

    weak_model = RandomForestClassifier(
        n_estimators=10,
        max_depth=2,
        min_samples_leaf=20,
        max_features=0.4,
        random_state=1000 + i,
        n_jobs=-1
    )

    oof_preds = np.zeros(N_train)
    oof_probs = np.zeros(N_train)

    for train_idx, val_idx in skf.split(X_train, y_train):

        weak_model.fit(
            X_train.iloc[train_idx],
            y_train.iloc[train_idx]
        )

        probs = weak_model.predict_proba(X_train.iloc[val_idx])[:, 1]
        preds = (probs >= 0.5).astype(int)

        oof_preds[val_idx] = preds
        oof_probs[val_idx] = probs

    # Binary correctness (M(p,q))
    decision_matrix[i] = (
        oof_preds == y_train.values
    ).astype(int)

    # Compute AUC using OOF probabilities
    auc = roc_auc_score(y_train, oof_probs)
    weak_auc_scores.append(auc)

    print(f"Weak Model {i+1} AUC: {auc:.4f}")

print("\nMean Weak AUC:", round(np.mean(weak_auc_scores), 4))

print("✅ Weak models completed (no sample trained on itself)")

# ============================================================
# STEP 4: OPINION SCORE (Σ)
# ============================================================

opinion_score = decision_matrix.sum(axis=0)

print("\nOpinion Score stats:")
print("Min correct votes:", opinion_score.min())
print("Max correct votes:", opinion_score.max())

# ============================================================
# STEP 5: HARDNESS
# ============================================================

hardness = N0 - opinion_score
hardness_norm = hardness / N0

# ============================================================
# STEP 6: RANK (0 = HARDEST, 10 = EASIEST)
# ============================================================

easiness_norm = opinion_score / N0

rank_0_10 = np.round(easiness_norm * 10).astype(int)
rank_0_10 = np.clip(rank_0_10, 0, 10)

ranking_df = pd.DataFrame({
    "row_id": X_train.index,
    "true_label": y_train.values,
    "Opinion_Score": opinion_score,
    "Hardness": hardness_norm,
    "rank_0_10": rank_0_10
})

# ============================================================
# STEP 7: PRINT DISTRIBUTION
# ============================================================

print("\nRank Distribution (0 = HARDEST → 10 = EASIEST):")
print(ranking_df["rank_0_10"].value_counts().sort_index())

# ============================================================
# STEP 8: SHOW HARDEST & EASIEST
# ============================================================

print("\nTop 10 HARDEST samples:")
print(ranking_df.sort_values("Hardness", ascending=False).head(10))

print("\nTop 10 EASIEST samples:")
print(ranking_df.sort_values("Hardness", ascending=True).head(10))

# ============================================================
# STEP 9: SAVE RANKING
# ============================================================

ranking_df.to_excel("WeakModel_PaperCorrect_Ranking.xlsx", index=False)

print("\n✅ Ranking saved to WeakModel_PaperCorrect_Ranking.xlsx")
