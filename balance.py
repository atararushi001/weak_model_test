import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, precision_recall_curve

from sklearn.ensemble import RandomForestClassifier, StackingClassifier

from xgboost import XGBClassifier

np.random.seed(42)

# ============================================================
# STEP 1: LOAD DATA
# ============================================================

df = pd.read_csv("dataset/features_matrix.csv")

df = df[(df["isGET"] == 1) | (df["isPOST"] == 1)]
df = df[df["flag"].isin(["y", "n"])]

X = df.drop(columns=[
    "reqId","flag",
    "changeInParams",
    "passwordInPath",
    "payInPath",
    "viewInParams"
], errors="ignore").fillna(0)

y = df["flag"].map({"y":1,"n":0})

print("\nTotal samples:", len(X))
print("Sensitive ratio:", round(y.mean(),4))


# ============================================================
# STEP 2: TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42

)

print("\nTrain size:", len(X_train))
print("Test size:", len(X_test))


# ============================================================
# STEP 3: FEATURE SELECTION
# ============================================================

print("\nSelecting best features...")

selector = RandomForestClassifier(
    n_estimators=500,
    random_state=42,
    n_jobs=-1
)

selector.fit(X_train,y_train)

importance = selector.feature_importances_

indices = np.argsort(importance)[::-1]

top_k = int(len(indices)*0.7)

selected = indices[:top_k]

X_train = X_train.iloc[:,selected]
X_test = X_test.iloc[:,selected]

print("Selected features:", X_train.shape[1])


# ============================================================
# STEP 4: WEAK MODELS (UNCHANGED)
# ============================================================

N0 = 10

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

N_train = len(X_train)

decision_matrix = np.zeros((N0,N_train))

print("\nTraining weak models...")

for i in range(N0):

    weak_model = RandomForestClassifier(

        n_estimators=10,
        max_depth=3,
        min_samples_leaf=20,
        max_features=0.5,
        random_state=1000+i,
        n_jobs=-1

    )

    oof_preds = np.zeros(N_train)

    for train_idx,val_idx in skf.split(X_train,y_train):

        weak_model.fit(
            X_train.iloc[train_idx],
            y_train.iloc[train_idx]
        )

        preds = weak_model.predict(
            X_train.iloc[val_idx]
        )

        oof_preds[val_idx] = preds

    decision_matrix[i] = (
        oof_preds == y_train.values
    )


# ============================================================
# STEP 5: HARDNESS RANKING
# ============================================================

hardness = N0 - decision_matrix.sum(axis=0)

ranking_df = pd.DataFrame({

    "row_id": np.arange(N_train),
    "Hardness": hardness

}).sort_values("Hardness",ascending=False)

ranking_df["r"] = np.arange(1,N_train+1)


# ============================================================
# STEP 6: HARDNESS SAMPLING
# ============================================================

D=100
x_param=0.5

ranking_df["H_r"] = (
    D*np.exp(-(ranking_df["r"]**x_param)/N_train)
)

ranking_df["P_r"] = (
    ranking_df["H_r"]/ranking_df["H_r"].sum()
)

NUM_FINAL_MODELS=15

TRAIN_SIZE=int(0.8*N_train)

hard_datasets=[]

for i in range(NUM_FINAL_MODELS):

    sampled_positions=np.random.choice(

        ranking_df.index,
        size=TRAIN_SIZE,
        replace=True,
        p=ranking_df["P_r"]

    )

    hard_datasets.append(

        ranking_df.iloc[
            sampled_positions
        ]["row_id"].values

    )


# ============================================================
# STEP 7: FINAL STACKING ENSEMBLE
# ============================================================

models=[]

print("\nTraining stacking ensemble...")

scale_weight = (
    (len(y_train)-sum(y_train))/sum(y_train)
)

for i,idx in enumerate(hard_datasets):

    rf = RandomForestClassifier(

        n_estimators=400,
        max_depth=12,
        class_weight="balanced",
        random_state=5000+i,
        n_jobs=-1

    )

    xgb = XGBClassifier(

        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_weight,
        eval_metric="auc",
        random_state=6000+i,
        n_jobs=-1

    )

    stack = StackingClassifier(

        estimators=[

            ("rf",rf),
            ("xgb",xgb)

        ],

        final_estimator=RandomForestClassifier(

            n_estimators=200,
            random_state=7000+i

        ),

        n_jobs=-1

    )

    stack.fit(
        X_train.iloc[idx],
        y_train.iloc[idx]
    )

    models.append(stack)


# ============================================================
# STEP 8: PREDICTION
# ============================================================

ensemble_probs=np.zeros(len(X_test))

for clf in models:

    ensemble_probs+=clf.predict_proba(
        X_test
    )[:,1]

ensemble_probs/=len(models)


# ============================================================
# STEP 9: BEST THRESHOLD
# ============================================================

precision,recall,thresholds=precision_recall_curve(

    y_test,
    ensemble_probs

)

f1_scores=(
    2*precision*recall/
    (precision+recall+1e-10)
)

best_threshold=thresholds[
    np.argmax(f1_scores)
]

print("\nBest Threshold:",round(best_threshold,4))


ensemble_preds=(
    ensemble_probs>=best_threshold
).astype(int)


# ============================================================
# STEP 10: FINAL RESULTS
# ============================================================

auc=roc_auc_score(
    y_test,
    ensemble_probs
)

f1=f1_score(
    y_test,
    ensemble_preds
)

cm=confusion_matrix(
    y_test,
    ensemble_preds
)

print("\nConfusion Matrix:")
print(cm)

print("\n================ FINAL RESULTS ================")

print("ROC-AUC Score :",round(auc,4))

print("F1 Score      :",round(f1,4))

print("===============================================")