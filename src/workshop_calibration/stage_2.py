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
# Step 0 : Import Libraries
import marimo as mo
import pandas as pd
import numpy as np
import random
from scipy.io import arff
from sklearn.metrics import log_loss, brier_score_loss, f1_score
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from venn_abers import VennAbersCalibrator
import calibration as cal

import warnings

warnings.filterwarnings("ignore")

# %%
# Step 1 : Data Loading
random.seed(1)

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

data = arff.loadarff("../../data/___")  # TODO
df = pd.DataFrame(data[0])

print(df.head(___))  # TODO

# %%
# Step 2 : Data preparation
X = df.drop("___", axis=1)  # TODO
y = df["___"].astype(___).subtract(___)  # TODO

print(X.head(5))
print(y.head(5))

# %%
# Step 3 : Data Splitting for Training, Calibration and Testing
X_train, X_test, y_train, y_test = ___(  # TODO
    ___,  # TODO
    ___,  # TODO
    test_size=___,  # TODO
    shuffle=___,  # TODO
)

X_proper_train, X_cal, y_proper_train, y_cal = train_test_split(
    ___,  # TODO
    ___,  # TODO
    ___,  # TODO
    ___,  # TODO
)

print(X_proper_train.head())
print(X_cal.head())
print(y_proper_train.head())
print(y_cal.head())

# %%
# Step 4 : Define the models to test
clfs = {}
clfs["Naive Bayes"] = GaussianNB()
clfs["SVM"] = SVC(probability=True)
clfs["RF"] = RandomForestClassifier()
clfs["AdaBoost"] = AdaBoostClassifier()
clfs["Logistic"] = LogisticRegression(max_iter=10000)
clfs["Neural Network"] = MLPClassifier(max_iter=10000)

for name_model in clfs.keys():
    print(f"- {name_model}")


# %%
# Step 5 : Define the metrics
def metrics(
    clf,
    X_test,
    y_test,
    acc_list,
    log_loss_list,
    brier_loss_list,
    ece_list,
    VennAbersCalibrator=False,
):
    if VennAbersCalibrator:
        p_pred = clf.___(np.asarray(___))  # TODO
        y_pred = clf.___(np.array(___), one_hot=False)  # TODO
    else:
        p_pred = clf.predict_proba(___)  # TODO
        y_pred = clf.predict(___)  # TODO
    acc_list.append(___(y_test, y_pred, average="___"))  # TODO
    log_loss_list.append(___(y_test, p_pred))  # TODO
    brier_loss_list.append(___(y_test, p_pred))  # TODO
    ece_list.append(___(p_pred, y_test))  # TODO
    return acc_list, log_loss_list, brier_loss_list, ece_list


# %%
# Step 6: Calibrate the models
def run_multiclass_comparison(clf_name, clf):

    print(clf_name + ":")
    log_loss_list = []
    brier_loss_list = []
    acc_list = []
    ece_list = []

    print("base")
    clf.___(__, ___)  # TODO
    acc_list, log_loss_list, brier_loss_list, ece_list = metrics(
        ___,  # TODO
        ___,  # TODO
        ___,  # TODO
        ___,  # TODO
        ___,  # TODO
        ___,  # TODO
        ___,  # TODO
    )

    print("sigmoid")
    clf.___(__, ___)  # TODO
    cal_sigm = ___(__, ___, ___)  # TODO
    cal_sigm.___(__, ___)  # TODO
    acc_list, log_loss_list, brier_loss_list, ece_list = metrics(
        ___,  # TODO
        ___,  # TODO
        ___,  # TODO
        ___,  # TODO
        ___,  # TODO
        ___,  # TODO
        ___,  # TODO
    )

    print("isotonic")
    cal_iso = ___(__, ___, ___)  # TODO
    cal_iso.___(__, ___)  # TODO
    acc_list, log_loss_list, brier_loss_list, ece_list = metrics(
        ___,  # TODO
        ___,  # TODO
        ___,  # TODO
        ___,  # TODO
        ___,  # TODO
        ___,  # TODO
        ___,  # TODO
    )

    print("sigmoid_cv")
    cal_sigm_cv = ___(__, ___, ___)  # TODO
    cal_sigm_cv.___(__, ___)  # TODO
    acc_list, log_loss_list, brier_loss_list, ece_list = metrics(
        ___,  # TODO
        ___,  # TODO
        ___,  # TODO
        ___,  # TODO
        ___,  # TODO
        ___,  # TODO
        ___,  # TODO
    )

    print("isotonic_cv")
    cal_iso_cv = ___(__, ___, ___)  # TODO
    cal_iso_cv.___(__, ___)  # TODO
    acc_list, log_loss_list, brier_loss_list, ece_list = metrics(
        ___,  # TODO
        ___,  # TODO
        ___,  # TODO
        ___,  # TODO
        ___,  # TODO
        ___,  # TODO
        ___,  # TODO
    )

    print("ivap")
    va = ___(__, ___, ___)  # TODO
    va.___(__, ___)  # TODO
    acc_list, log_loss_list, brier_loss_list, ece_list = metrics(
        ___,  # TODO
        ___,  # TODO
        ___,  # TODO
        ___,  # TODO
        ___,  # TODO
        ___,  # TODO
        ___,  # TODO
        VennAbersCalibrator=___,  # TODO
    )

    print("cvap \n")
    va_cv = ___(__, ___, ___)  # TODO
    va_cv.___(__, ___)  # TODO
    acc_list, log_loss_list, brier_loss_list, ece_list = metrics(
        ___,  # TODO
        ___,  # TODO
        ___,  # TODO
        ___,  # TODO
        ___,  # TODO
        ___,  # TODO
        ___,  # TODO
        VennAbersCalibrator=___,  # TODO
    )

    df_ll = pd.DataFrame(
        columns=[
            "Classifier",
            "Uncalibrated",
            "Platt",
            "Isotonic",
            "Platt-CV",
            "Isotonic-CV",
            "IVAP",
            "CVAP",
        ]
    )
    df_ll.loc[0] = [clf_name] + log_loss_list
    df_bl = pd.DataFrame(
        columns=[
            "Classifier",
            "Uncalibrated",
            "Platt",
            "Isotonic",
            "Platt-CV",
            "Isotonic-CV",
            "IVAP",
            "CVAP",
        ]
    )
    df_bl.loc[0] = [clf_name] + brier_loss_list
    df_acc = pd.DataFrame(
        columns=[
            "Classifier",
            "Uncalibrated",
            "Platt",
            "Isotonic",
            "Platt-CV",
            "Isotonic-CV",
            "IVAP",
            "CVAP",
        ]
    )
    df_acc.loc[0] = [clf_name] + acc_list
    df_ece = pd.DataFrame(
        columns=[
            "Classifier",
            "Uncalibrated",
            "Platt",
            "Isotonic",
            "Platt-CV",
            "Isotonic-CV",
            "IVAP",
            "CVAP",
        ]
    )
    df_ece.loc[0] = [clf_name] + ece_list

    return df_bl, df_ll, df_acc, df_ece


# %%
# Step 7 : Compare models on multiclass classification
print("Comparing models for multiclass classification")
results_brier = pd.DataFrame()
results_log = pd.DataFrame()
results_acc = pd.DataFrame()
results_ece = pd.DataFrame()

for ____ in ___:  # TODO
    scratch_b, scratch_l, scratch_acc, scratch_ece = ___(  # TODO
        ___,  # TODO
        ___,  # TODO
    )
    results_brier = pd.concat((___, ___), ignore_index=True)  # TODO
    results_log = pd.concat((___, ___), ignore_index=True)  # TODO
    results_acc = pd.concat((___, ___), ignore_index=True)  # TODO
    results_ece = pd.concat((___, ___), ignore_index=True)  # TODO


# %%
# Step 8 : Define the function to convert the dataframe to a markdown table
def df_to_markdown_table(df, higher_is_better=True):
    # Convert to float and find best indices
    df_float = df.select_dtypes(include=["number"])
    if higher_is_better:
        best_indices = df_float.idxmax(axis=1)
    else:
        best_indices = df_float.idxmin(axis=1)

    # Round and convert to string
    formatted_df = df_float.round(4).astype(str)

    # Bold the best values
    for idx, best_col in enumerate(best_indices):
        formatted_df.iloc[idx, formatted_df.columns.get_loc(best_col)] = (
            f"**{formatted_df.iloc[idx, formatted_df.columns.get_loc(best_col)]}**"
        )

    # Generate markdown table
    headers = [""] + list(formatted_df.columns)
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for idx, row in formatted_df.iterrows():
        line = "| " + str(idx) + " | " + " | ".join(row.values) + " |"
        lines.append(line)

    return "\n".join(lines)


def get_best_metric(df, metric_name, higher_is_better=True):
    values = df.mean()
    ranks = df.rank(axis=1, ascending=not higher_is_better).mean()
    best_value = values.argmax() if higher_is_better else values.argmin()
    best_rank = ranks.argmin()
    formatted_values = values.round(4).astype(str)
    formatted_values[values.index[best_value]] = (
        f"**{formatted_values[values.index[best_value]]}**"
    )
    formatted_ranks = ranks.round(2).astype(str)
    formatted_ranks[ranks.index[best_rank]] = (
        f"**{formatted_ranks[ranks.index[best_rank]]}**"
    )

    # Create markdown table
    headers = ["Method", "Value", "Rank"]
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| --- | --- | --- |")

    for method, val, rank in zip(values.index, formatted_values, formatted_ranks):
        lines.append(f"| {method} | {val} | {rank} |")

    return f"### {metric_name} :\n" + "\n".join(lines)


if "Classifier" in results_acc.columns:
    results_acc.set_index("Classifier", inplace=True)
if "Classifier" in results_brier.columns:
    results_brier.set_index("Classifier", inplace=True)
if "Classifier" in results_ece.columns:
    results_ece.set_index("Classifier", inplace=True)
if "Classifier" in results_log.columns:
    results_log.set_index("Classifier", inplace=True)

# %%
# Step 9 : Display the results
mo.md(
    "## Accuracy Results\n"
    + df_to_markdown_table(results_acc, higher_is_better=True)
    + "\n\n## Brier Loss Results\n"
    + df_to_markdown_table(results_brier, higher_is_better=False)
    + "\n\n## Log Loss Results\n"
    + df_to_markdown_table(results_log, higher_is_better=False)
    + "\n\n## ECE Results\n"
    + df_to_markdown_table(results_ece, higher_is_better=False)
    + "\n\n\n## Summary Statistics\n"
    + f"{get_best_metric(results_acc, 'Accuracy', higher_is_better=True)}\n\n"
    + f"{get_best_metric(results_brier, 'Brier Loss', higher_is_better=False)}\n\n"
    + f"{get_best_metric(results_log, 'Log Loss', higher_is_better=False)}\n\n"
    + f"{get_best_metric(results_ece, 'ECE', higher_is_better=False)}"
)

# %%
