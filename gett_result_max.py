import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score
np.random.seed(42)
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
# STEP 2: TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)
N_train = len(X_train)
print("\nTrain size:", N_train)
print("Test size :", len(X_test))


# ============================================================
# STEP 3: WEAK MODELS
# ============================================================

N0 = 10

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

decision_matrix = np.zeros((N0, N_train))
weak_auc_scores = []

print("\nTraining weak models...")

for i in range(N0):

    weak_model = RandomForestClassifier(
        n_estimators=10,
        max_depth=2,
        min_samples_leaf=20,
        max_features=0.4,
        bootstrap=True,
        random_state=1000+i,
        n_jobs=-1
    )

    oof_preds = np.zeros(N_train)
    oof_probs = np.zeros(N_train)

    for train_idx, val_idx in skf.split(X_train, y_train):

        weak_model.fit(
            X_train.iloc[train_idx],
            y_train.iloc[train_idx]
        )

        probs = weak_model.predict_proba(
            X_train.iloc[val_idx]
        )[:,1]

        preds = (probs >= 0.5).astype(int)

        oof_preds[val_idx] = preds
        oof_probs[val_idx] = probs


    decision_matrix[i] = (
        oof_preds == y_train.values
    ).astype(int)

    auc = roc_auc_score(y_train, oof_probs)

    weak_auc_scores.append(auc)


print("\nMean Weak AUC:", round(np.mean(weak_auc_scores),4))


# ============================================================
# STEP 4: HARDNESS CALCULATION
# ============================================================

opinion_score = decision_matrix.sum(axis=0)

hardness = N0 - opinion_score


ranking_df = pd.DataFrame({

    "row_id": X_train.index,
    "Hardness": hardness

})


# ============================================================
# STEP 5: CREATE RANK (0 = HARD , 10 = EASY)
# ============================================================
ranking_df["Rank"] = ranking_df["Hardness"]

print("\nRank distribution (0 = hard, 10 = easy):")
print(
    ranking_df["Rank"]
    .value_counts()
    .sort_index()
)

# ============================================================
# STEP 6: SORT FOR SAMPLING
# ============================================================
ranking_df = ranking_df.sort_values(
    "Hardness",
    ascending=False
).reset_index(drop=True)

ranking_df["r"] = np.arange(
    1,
    N_train+1
)

# ============================================================
# STEP 7: EXPONENTIAL PDF
# ============================================================

D = 100
x_param = 0.6

ranking_df["H_r"] = D * np.exp(
    -(ranking_df["r"] ** x_param) / N_train
)

ranking_df["P_r"] = (

    ranking_df["H_r"]
    /
    ranking_df["H_r"].sum()
)
print("\nHardness PDF created.")


# ============================================================
# STEP 8: HARDNESS SAMPLING
# ============================================================

NUM_FINAL_MODELS = 10
TRAIN_SIZE = int(0.9 * N_train)
hard_datasets = []


for i in range(NUM_FINAL_MODELS):
    sampled_positions = np.random.choice(
        ranking_df.index,
        size=TRAIN_SIZE,
        replace=True,
        p=ranking_df["P_r"]
    )

    sampled_ids = ranking_df.iloc[
        sampled_positions
    ]["row_id"].values
    hard_datasets.append(sampled_ids)

print("Hard datasets created.")
# ============================================================
# STEP 9: FINAL STRONG MODELS
# ============================================================
models = []

print("\nTraining final ensemble...")

for i, idx in enumerate(hard_datasets):

    clf = RandomForestClassifier(
        n_estimators=1200,
        max_depth=None,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight="balanced",
        random_state=5000+i,
        n_jobs=-1
    )
    clf.fit(
        X_train.loc[idx],
        y_train.loc[idx]
    )
    models.append(clf)

print("Final models trained.")


# ============================================================
# STEP 10: FINAL TEST EVALUATION
# ============================================================


ensemble_probs = np.zeros(len(X_test))

for clf in models:
    ensemble_probs += clf.predict_proba(
        X_test
    )[:,1]

ensemble_probs /= len(models)

auc = roc_auc_score(
    y_test,
    ensemble_probs
)


ensemble_preds = (
    ensemble_probs >= 0.5
).astype(int)

f1 = f1_score(
    y_test,
    ensemble_preds
)

print("\n================ FINAL RESULTS ================")
print("ROC-AUC Score :", round(auc,4))
print("F1 Score      :", round(f1,4))
print("===============================================")