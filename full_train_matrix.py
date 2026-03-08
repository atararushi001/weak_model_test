
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report

# ============================================================
# STEP 1: LOAD DATA
# ============================================================

df = pd.read_csv("dataset/features_matrix.csv")

df = df[(df['isGET'] == 1) | (df['isPOST'] == 1)]
df = df[df['flag'].isin(['y', 'n'])]

discard_cols = [
    'reqId','flag',
    'changeInParams',
    'passwordInPath',
    'payInPath',
    'viewInParams'
]

X = df.drop(columns=discard_cols, errors='ignore')
y = df['flag'].map({'y':1,'n':0})

print("Dataset size:", len(X))
print("Sensitive ratio:", y.mean())

# ============================================================
# STEP 2: TRAIN TEST SPLIT
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
# STEP 3: TRAIN WEAK MODELS
# ============================================================

NUM_WEAK_MODELS = 10
prob_matrix = np.zeros((N_train,NUM_WEAK_MODELS))

print("\nTraining weak models...")

for i in range(NUM_WEAK_MODELS):

    clf = RandomForestClassifier(
        n_estimators=10,
        max_depth=3,
        min_samples_leaf=10,
        max_features='sqrt',
        random_state=10+i,
        n_jobs=-1
    )

    clf.fit(X_train,y_train)

    prob_matrix[:,i] = clf.predict_proba(X_train)[:,1]

# ============================================================
# STEP 4: HARDNESS RANKING
# ============================================================

std_prob = prob_matrix.std(axis=1)

rank_0_10 = np.clip(
    np.round(
        10 - 10*(std_prob-std_prob.min())/
        (std_prob.max()-std_prob.min())
    ),
    0,10
).astype(int)

ranking_df = pd.DataFrame({
    "row_id":np.arange(N_train),
    "rank":rank_0_10
})

print("\nRank distribution")
print(ranking_df["rank"].value_counts().sort_index())

# ============================================================
# STEP 5: HARDNESS SAMPLING
# ============================================================

D = 100
x = 1.0

ranking_df = ranking_df.sort_values("rank").reset_index(drop=True)

ranking_df["r"] = np.arange(1,N_train+1)

ranking_df["P_r"] = D*np.exp(-(ranking_df["r"]**x)/N_train)
ranking_df["P_r"] /= ranking_df["P_r"].sum()

# ============================================================
# STEP 6: HARDNESS ENSEMBLE
# ============================================================

RUNS = 5
NUM_MODELS = 20
TRAIN_SIZE = int(0.9*N_train)

f1_scores=[]
auc_scores=[]

training_samples_all_runs=[]

print("\nRunning hardness-aware ensemble...")

for run in range(RUNS):

    print(f"\n================ Run {run+1}/{RUNS} ================")

    np.random.seed(42+run)

    models=[]
    sampled_indices_run=[]

    for i in range(NUM_MODELS):

        sampled_idx = np.random.choice(
            ranking_df["row_id"],
            size=TRAIN_SIZE,
            replace=True,
            p=ranking_df["P_r"]
        )
        
        sampled_indices_run.extend(sampled_idx)

        clf = RandomForestClassifier(
            n_estimators=2500,
            max_depth=None,
            min_samples_leaf=2,
            max_features=None,
            class_weight="balanced",
            random_state=10+i,
            n_jobs=-1
        )

        clf.fit(
            X_train.iloc[sampled_idx],
            y_train.iloc[sampled_idx]
        )

        models.append(clf)

    training_samples_all_runs.append(sampled_indices_run)

    # Ensemble prediction
    ensemble_probs = np.zeros(len(X_test))

    for clf in models:
        ensemble_probs += clf.predict_proba(X_test)[:,1]

    ensemble_probs /= len(models)

    ensemble_preds = (ensemble_probs>=0.5).astype(int)

    f1 = f1_score(y_test,ensemble_preds)
    auc = roc_auc_score(y_test,ensemble_probs)

    f1_scores.append(f1)
    auc_scores.append(auc)

    print("F1 :",round(f1,4))
    print("AUC:",round(auc,4))

# ============================================================
# FINAL RESULTS
# ============================================================

print("\n================ FINAL RESULTS ================")

print("F1 :",np.mean(f1_scores),"+-",np.std(f1_scores))
print("AUC:",np.mean(auc_scores),"+-",np.std(auc_scores))

# ============================================================
# CONFUSION MATRIX ANALYSIS
# ============================================================

cm = confusion_matrix(y_test,ensemble_preds)

TN,FP,FN,TP = cm.ravel()

print("\nConfusion Matrix")
print(cm)

print("\nDetailed Metrics")

print("True Positive:",TP)
print("True Negative:",TN)
print("False Positive:",FP)
print("False Negative:",FN)

precision = TP/(TP+FP)
recall = TP/(TP+FN)
specificity = TN/(TN+FP)

fpr = FP/(FP+TN)
fnr = FN/(FN+TP)

print("\nPrecision:",precision)
print("Recall:",recall)
print("Specificity:",specificity)

print("False Positive Rate:",fpr)
print("False Negative Rate:",fnr)

print("\nClassification Report")
print(classification_report(y_test,ensemble_preds))

# ============================================================
# SAVE PREDICTIONS
# ============================================================

results = pd.DataFrame({
    "true_label":y_test,
    "pred_label":ensemble_preds,
    "probability":ensemble_probs
})

results["correct"]=(results["true_label"]==results["pred_label"]).astype(int)

results.to_csv("test_predictions_full.csv",index=False)

# ============================================================
# SAVE MISCLASSIFIED
# ============================================================

errors = results[results["correct"]==0]

errors.to_csv("misclassified_samples.csv",index=False)

print("\nTotal errors:",len(errors))

# ============================================================
# SAVE TRAINING SAMPLES USED
# ============================================================

training_log = pd.DataFrame()

for i,indices in enumerate(training_samples_all_runs):

    temp = pd.DataFrame({
        "run":i+1,
        "train_row_index":indices
    })

    training_log = pd.concat([training_log,temp])

training_log.to_csv("training_samples_used.csv",index=False)

print("\nTraining sample log saved")

# ============================================================
# PROBABILITY DISTRIBUTION
# ============================================================

plt.figure()

plt.hist(
    ensemble_probs[y_test==0],
    bins=40,
    alpha=0.5,
    label="Normal"
)

plt.hist(
    ensemble_probs[y_test==1],
    bins=40,
    alpha=0.5,
    label="Sensitive"
)

plt.legend()
plt.title("Probability Distribution")

plt.savefig("probability_distribution.png")
plt.close()

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

feat_imp = pd.DataFrame({
    "feature":X_train.columns,
    "importance":models[0].feature_importances_
})

feat_imp = feat_imp.sort_values(
    "importance",
    ascending=False
)

feat_imp.to_csv("feature_importance.csv",index=False)

print("\nTop Important Features")
print(feat_imp.head(20))
