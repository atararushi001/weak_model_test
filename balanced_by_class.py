import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    classification_report,
    f1_score
)

# ============================================================
# STEP 1: LOAD DATA
# ============================================================

df = pd.read_csv("dataset/features_matrix.csv")

df = df[(df['isGET']==1) | (df['isPOST']==1)]
df = df[df['flag'].isin(['y','n'])]

discard_cols = [
    'reqId','flag',
    'changeInParams',
    'passwordInPath',
    'payInPath',
    'viewInParams'
]

X = df.drop(columns=discard_cols,errors="ignore")
y = df['flag'].map({'y':1,'n':0})

print("Dataset size:",len(X))
print("Sensitive ratio:",y.mean())

# ============================================================
# STEP 2: TRAIN TEST SPLIT
# ============================================================

X_train,X_test,y_train,y_test = train_test_split(
    X,y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

X_train=X_train.reset_index(drop=True)
y_train=y_train.reset_index(drop=True)

X_test=X_test.reset_index(drop=True)
y_test=y_test.reset_index(drop=True)

print("\nTrain size:",len(X_train))
print("Test size:",len(X_test))

# ============================================================
# STEP 3: WEAK MODELS (FOR HARDNESS)
# ============================================================

NUM_WEAK_MODELS=10

prob_matrix=np.zeros((len(X_train),NUM_WEAK_MODELS))

print("\nTraining weak models...")

for i in range(NUM_WEAK_MODELS):

    clf=RandomForestClassifier(
        n_estimators=3,
        max_depth=3,
        min_samples_leaf=10,
        random_state=10+i,
        n_jobs=-1
    )

    clf.fit(X_train,y_train)

    probs=clf.predict_proba(X_train)[:,1]

    prob_matrix[:,i]=probs

# ============================================================
# STEP 4: HARDNESS RANK
# ============================================================

std_prob=prob_matrix.std(axis=1)

rank_0_10=np.clip(
    np.round(
        10-10*(std_prob-std_prob.min())/
        (std_prob.max()-std_prob.min())
    ),
    0,10
).astype(int)

ranking_df=pd.DataFrame({
    "row_id":np.arange(len(X_train)),
    "true_label":y_train,
    "std_prob":std_prob,
    "rank":rank_0_10
})

print("\nRank distribution")
print(ranking_df["rank"].value_counts().sort_index())

# ============================================================
# STEP 5: CREATE BALANCED TRAIN SET
# ============================================================

sensitive_idx = ranking_df[ranking_df.true_label==1].row_id.values
normal_idx = ranking_df[ranking_df.true_label==0].row_id.values

n_sensitive=len(sensitive_idx)

np.random.seed(42)
normal_sample=np.random.choice(normal_idx,n_sensitive,replace=False)

final_idx=np.concatenate([sensitive_idx,normal_sample])

print("\nBalanced training dataset")
print("Sensitive:",n_sensitive)
print("Non-sensitive:",len(normal_sample))
print("Total:",len(final_idx))

X_balanced=X_train.iloc[final_idx]
y_balanced=y_train.iloc[final_idx]

# ============================================================
# FUNCTION TO PRINT METRICS
# ============================================================
def print_metrics(y_true, probs):

    preds = (probs >= 0.5).astype(int)

    auc = roc_auc_score(y_true, probs)
    f1 = f1_score(y_true, preds)

    cm = confusion_matrix(y_true, preds)

    TN, FP, FN, TP = cm.ravel()

    total = TP + TN + FP + FN

    sensitive_total = TP + FN
    normal_total = TN + FP

    precision = TP / (TP + FP + 1e-9)
    recall = TP / (TP + FN + 1e-9)
    specificity = TN / (TN + FP + 1e-9)

    FPR = FP / (FP + TN + 1e-9)
    FNR = FN / (FN + TP + 1e-9)

    accuracy = (TP + TN) / total

    print("AUC:", round(auc,4))
    print("F1 :", round(f1,4))
    print("Accuracy:", round(accuracy*100,2), "%")

    print("\nConfusion Matrix")
    print(cm)

    print("\n===== CLASS DISTRIBUTION =====")
    print("Total Sensitive:", sensitive_total)
    print("Total Normal:", normal_total)

    print("\n===== CONFUSION MATRIX DETAILS =====")

    print("True Positive:", TP, 
          f"({TP/sensitive_total*100:.2f}%)")

    print("False Negative:", FN,
          f"({FN/sensitive_total*100:.2f}%)")

    print("True Negative:", TN,
          f"({TN/normal_total*100:.2f}%)")

    print("False Positive:", FP,
          f"({FP/normal_total*100:.2f}%)")

    print("\n===== PERFORMANCE METRICS =====")

    print("Precision:", f"{precision*100:.2f}%")
    print("Recall (TPR):", f"{recall*100:.2f}%")
    print("Specificity (TNR):", f"{specificity*100:.2f}%")

    print("False Positive Rate (FPR):", f"{FPR*100:.2f}%")
    print("False Negative Rate (FNR):", f"{FNR*100:.2f}%")
# ============================================================
# STEP 6: TRAIN FINAL MODELS
# ============================================================

NUM_FINAL_MODELS=5

for m in range(NUM_FINAL_MODELS):

    print("\n==============================")
    print("FINAL MODEL",m+1)
    print("==============================")

    clf=RandomForestClassifier(

        n_estimators=2500,
        max_depth=None,
        min_samples_leaf=2,
        max_features=None,
        class_weight="balanced",

        random_state=100+m,
        n_jobs=-1
    )

    clf.fit(X_balanced,y_balanced)

    # ----------------------------------
    # TRAIN (BALANCED DATA)
    # ----------------------------------

    print("\n--- TRAIN (Balanced subset) ---")

    probs=clf.predict_proba(X_balanced)[:,1]

    print_metrics(y_balanced,probs)

    # ----------------------------------
    # TRAIN (FULL TRAIN DATA)
    # ----------------------------------

    print("\n--- TRAIN (Full training set) ---")

    probs=clf.predict_proba(X_train)[:,1]

    print_metrics(y_train,probs)

    # ----------------------------------
    # TEST
    # ----------------------------------

    print("\n--- TEST (Full test set) ---")

    probs=clf.predict_proba(X_test)[:,1]

    print_metrics(y_test,probs)

    print("\nClassification Report")
    print(classification_report(y_test,(probs>=0.5)))