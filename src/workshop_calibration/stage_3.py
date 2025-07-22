# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
# ---

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import brier_score_loss, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import calibration as cal


# %%
# Part 1 : Data Preparation and Splitting

random_state = 6

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

data = arff.loadarff("../../data/dataset_multilabel.arff")
df = pd.DataFrame(data[0])

print(df.head())

numeric_cols = df.select_dtypes(include=[np.number]).columns
X = df[numeric_cols]

label_cols = [
    "25400",
    "29600",
    "30400",
    "33400",
    "17300",
    "19400",
    "34500",
    "38100",
    "49700",
    "50390",
    "55800",
    "57500",
    "59300",
    "37880",
]
y = df[label_cols].astype(int)
print(y.head())

label_counts = y.sum()
print("\nDistribution of labels:")
print(label_counts.to_string())
print(f"Total number of unique labels: {len(label_counts)}")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.90,
    shuffle=False,
    random_state=random_state,
)

X_proper_train, X_cal, y_proper_train, y_cal = train_test_split(
    X_train,
    y_train,
    test_size=0.2,
    shuffle=False,
    random_state=random_state,
)

# %%
# Part 2 : Multi-output classifier non calibrated

base_clf = RandomForestClassifier(random_state=random_state)
multi_clf = MultiOutputClassifier(base_clf)
multi_clf.fit(X_train, y_train)
pred_probs_uncalibrated = multi_clf.predict_proba(X_test)
pred_y_uncalibrated = multi_clf.predict(X_test)

# %%
# Part 3 : Model calibration

calibrated_clfs = []
pred_probs_calibrated = []
pred_y_calibrated = []
for i in range(y.shape[1]):
    clf = CalibratedClassifierCV(base_clf, method="sigmoid", cv=10)
    clf.fit(X_train, y_train.iloc[:, i])
    calibrated_clfs.append(clf)
    pred_probs_calibrated.append(clf.predict_proba(X_test)[:, 1])
    pred_y_calibrated.append(clf.predict(X_test))

# Conversion to numpy matrix
pred_probs_uncalibrated_matrix = np.vstack([p[:, 1] for p in pred_probs_uncalibrated]).T
pred_probs_calibrated_matrix = np.vstack(pred_probs_calibrated).T

pred_y_uncalibrated_matrix = np.column_stack(pred_y_uncalibrated).T
pred_y_calibrated_matrix = np.column_stack(pred_y_calibrated)

# %%
# Part 4 : Compute metrics

brier_scores_uncalibrated = [
    brier_score_loss(y_test.iloc[:, i], pred_probs_uncalibrated_matrix[:, i])
    for i in range(y.shape[1])
]
brier_scores_calibrated = [
    brier_score_loss(y_test.iloc[:, i], pred_probs_calibrated_matrix[:, i])
    for i in range(y.shape[1])
]

accuracy_scores_uncalibrated = [
    accuracy_score(y_test.iloc[:, i], pred_y_uncalibrated_matrix[:, i])
    for i in range(y_test.shape[1])
]
accuracy_scores_calibrated = [
    accuracy_score(y_test.iloc[:, i], pred_y_calibrated_matrix[:, i])
    for i in range(y_test.shape[1])
]

ece_scores_uncalibrated = [
    cal.get_calibration_error(pred_probs_uncalibrated_matrix[:, i], y_test.iloc[:, i])
    for i in range(y.shape[1])
]
ece_scores_calibrated = [
    cal.get_calibration_error(pred_probs_calibrated_matrix[:, i], y_test.iloc[:, i])
    for i in range(y.shape[1])
]

# Summary in a DataFrame
brier_score_df = pd.DataFrame(
    {
        "Label": [f"label_{i}" for i in range(y.shape[1])],
        "Brier Uncalibrated": brier_scores_uncalibrated,
        "Brier Calibrated": brier_scores_calibrated,
    }
)

accuracy_score_df = pd.DataFrame(
    {
        "Label": [f"label_{i}" for i in range(y.shape[1])],
        "Accuracy Uncalibrated": accuracy_scores_uncalibrated,
        "Accuracy Calibrated": accuracy_scores_calibrated,
    }
)

ece_score_df = pd.DataFrame(
    {
        "Label": [f"label_{i}" for i in range(y.shape[1])],
        "ECE Uncalibrated": ece_scores_uncalibrated,
        "ECE Calibrated": ece_scores_calibrated,
    }
)

# %%
# Part 5 : Visualisation

# Plot Accuracy per label before and after calibration
plt.figure(figsize=(10, 5))
accuracy_long = accuracy_score_df.melt(
    id_vars="Label", var_name="Type", value_name="Accuracy"
)
sns.barplot(data=accuracy_long, x="Label", y="Accuracy", hue="Type")
plt.title("Accuracy per Label - Before and After Calibration")
plt.tight_layout()
plt.show()

# Plot Brier Score per label before and after calibration
plt.figure(figsize=(10, 5))
brier_long = brier_score_df.melt(
    id_vars="Label", var_name="Type", value_name="Brier Score"
)
sns.barplot(data=brier_long, x="Label", y="Brier Score", hue="Type")
plt.title("Brier Score per Label - Before and After Calibration")
plt.tight_layout()
plt.show()

# Plot ECE per label before and after calibration
plt.figure(figsize=(10, 5))
ece_long = ece_score_df.melt(id_vars="Label", var_name="Type", value_name="ECE")
sns.barplot(data=ece_long, x="Label", y="ECE", hue="Type")
plt.title("Expected Calibration Error (ECE) per Label - Before and After Calibration")
plt.tight_layout()
plt.show()
