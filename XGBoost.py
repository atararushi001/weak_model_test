import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, classification_report

from xgboost import XGBClassifier

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv("dataset/features_matrix.csv")

df = df[(df['isGET'] == 1) | (df['isPOST'] == 1)]
df = df[df['flag'].isin(['y','n'])]

discard_cols = [
    'reqId','flag',
    'changeInParams',
    'passwordInPath',
    'payInPath',
    'viewInParams'
]

X = df.drop(columns=discard_cols, errors="ignore")
y = df['flag'].map({'y':1,'n':0})

print("Dataset size:", len(X))
print("Sensitive ratio:", y.mean())

# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

X_train = X_train.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)

X_test = X_test.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

N_train = len(X_train)

# ============================================================
# WEAK MODELS FOR HARDNESS
# ============================================================

NUM_WEAK = 10
prob_matrix = np.zeros((N_train, NUM_WEAK))

print("\nTraining weak models...")

for i in range(NUM_WEAK):

    clf = RandomForestClassifier(
        n_estimators=20,
        max_depth=4,
        random_state=10+i,
        n_jobs=-1
    )

    clf.fit(X_train, y_train)

    prob_matrix[:,i] = clf.predict_proba(X_train)[:,1]

# ============================================================
# HARDNESS RANKING
# ============================================================

std_prob = prob_matrix.std(axis=1)

rank = np.clip(
    np.round(
        10 - 10*(std_prob-std_prob.min())/
        (std_prob.max()-std_prob.min())
    ),
    0,10
).astype(int)

ranking_df = pd.DataFrame({
    "row_id":np.arange(N_train),
    "rank":rank
})

print("\nRank distribution")
print(pd.Series(rank).value_counts().sort_index())

# ============================================================
# HARDNESS SAMPLING
# ============================================================

ranking_df = ranking_df.sort_values("rank").reset_index(drop=True)

ranking_df["r"] = np.arange(1,N_train+1)

D = 100
x = 1.0

ranking_df["P_r"] = D*np.exp(-(ranking_df["r"]**x)/N_train)
ranking_df["P_r"] /= ranking_df["P_r"].sum()

TRAIN_SIZE = int(0.9*N_train)

sample_idx = np.random.choice(
    ranking_df["row_id"],
    size=TRAIN_SIZE,
    replace=True,
    p=ranking_df["P_r"]
)

X_sub = X_train.iloc[sample_idx]
y_sub = y_train.iloc[sample_idx]

# ============================================================
# MODEL 1: RANDOM FOREST
# ============================================================

rf = RandomForestClassifier(
    n_estimators=2000,
    max_depth=None,
    min_samples_leaf=2,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42
)

rf.fit(X_sub, y_sub)

# ============================================================
# MODEL 2: EXTRA TREES
# ============================================================

et = ExtraTreesClassifier(
    n_estimators=2000,
    max_depth=None,
    min_samples_leaf=2,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42
)

et.fit(X_sub, y_sub)

# ============================================================
# MODEL 3: XGBOOST
# ============================================================

xgb = XGBClassifier(
    n_estimators=2000,
    max_depth=9,
    learning_rate=0.03,
    subsample=0.9,
    colsample_bytree=0.9,
    eval_metric="auc",
    tree_method="hist",
    scale_pos_weight=(len(y_sub)-sum(y_sub))/sum(y_sub),
    n_jobs=-1,
    random_state=42
)

xgb.fit(X_sub, y_sub)

# ============================================================
# STACKED PREDICTIONS
# ============================================================

rf_probs = rf.predict_proba(X_test)[:,1]
et_probs = et.predict_proba(X_test)[:,1]
xgb_probs = xgb.predict_proba(X_test)[:,1]

# weighted ensemble
ensemble_probs = (
    0.35*rf_probs +
    0.25*et_probs +
    0.40*xgb_probs
)

# ============================================================
# THRESHOLD OPTIMIZATION
# ============================================================

best_f1 = 0
best_t = 0.5

for t in np.arange(0.2,0.8,0.01):

    preds = (ensemble_probs >= t).astype(int)

    f1 = f1_score(y_test,preds)

    if f1 > best_f1:
        best_f1 = f1
        best_t = t

preds = (ensemble_probs >= best_t).astype(int)

# ============================================================
# RESULTS
# ============================================================

auc = roc_auc_score(y_test, ensemble_probs)

print("\nBest threshold:", best_t)

print("\nF1:", best_f1)
print("AUC:", auc)

print("\nConfusion Matrix")
print(confusion_matrix(y_test,preds))

print("\nClassification Report")
print(classification_report(y_test,preds))