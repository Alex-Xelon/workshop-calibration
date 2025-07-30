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
import random
from scipy.io import arff
from sklearn.metrics import log_loss, brier_score_loss, f1_score
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
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

data = ___  # TODO
df = pd.DataFrame(data[0])

print(___)  # TODO

# %%
# Step 2 : Data preparation
X = ___  # TODO
y = ___  # TODO

print(X.head(5))
print(y.head(5))

# %%
# Step 3 : Data Splitting for Training, Calibration and Testing
X_train, X_test, y_train, y_test = ___  # TODO
X_proper_train, X_cal, y_proper_train, y_cal = ___  # TODO

print(X_train.head(5))
print(X_test.head(5))
print(y_train.head(5))
print(y_test.head(5))

print("\n\n")

print(X_proper_train.head(5))
print(X_cal.head(5))
print(y_proper_train.head(5))
print(y_cal.head(5))

# %%
# Step 4 : Define the models to test
clfs = {}
clfs["SVM"] = SVC(probability=True)
clfs["RF"] = RandomForestClassifier()
clfs["AdaBoost"] = AdaBoostClassifier()
clfs["Logistic"] = LogisticRegression(max_iter=10000)
clfs["Neural Network"] = MLPClassifier(max_iter=10000)

for name_model in clfs.keys():
    print(f"- {name_model}")

# %%
# Step 5 : Example of calibration : Sigmoid and Isotonic
model_example = ___  # TODO

for method in [___]:  # TODO
    print(f"\nCalibrating LogisticRegression with {method} method")
    # Wrap with CalibratedClassifierCV using the chosen method
    calibrated_model = ___  # TODO

    # Predict probabilities and classes on the test set
    probs_cal = ___  # TODO
    preds_cal = ___  # TODO
    print(f"Probs calibration: \n{pd.DataFrame(probs_cal).head(10)}")
    print(f"Preds calibration: \n{pd.DataFrame(preds_cal).head(10)}")

# %%
# Step 6 : Example of metrics : Sigmoid and Isotonic
acc_cal = ___  # TODO
brier_cal = ___  # TODO
logloss_cal = ___  # TODO
ece_cal = ___  # TODO
print(f"Score Accuracy: {acc_cal:.3f}")
print(f"Brier Score: {brier_cal:.3f}")
print(f"Log Loss: {logloss_cal:.3f}")
print(f"ECE: {ece_cal:.3f}")

# %%
# Step 7 : Example of calibration : VennAbersCalibrator
va = ___  # TODO

probs_va = ___  # TODO
preds_va = ___  # TODO
print(f"Probs calibration: \n{pd.DataFrame(probs_va).head(10)}")
print(f"Preds calibration: \n{pd.DataFrame(preds_va).head(10)}")

# %%
# Step 8 : Example of metrics : VennAbersCalibrator
acc_va = ___  # TODO
brier_va = ___  # TODO
logloss_va = ___  # TODO
ece_va = ___  # TODO
print(f"Score Accuracy: {acc_va:.3f}")
print(f"Brier Score: {brier_va:.3f}")
print(f"Log Loss: {logloss_va:.3f}")
print(f"ECE: {ece_va:.3f}")


# %%
# Step 9 : Define the metrics
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
        p_pred = ___  # TODO
        y_pred = ___  # TODO
    else:
        p_pred = ___  # TODO
        y_pred = ___  # TODO
    acc_list.append(___)  # TODO
    log_loss_list.append(___)  # TODO
    brier_loss_list.append(___)  # TODO
    ece_list.append(___)  # TODO
    return acc_list, log_loss_list, brier_loss_list, ece_list


# %%
# Step 10: Calibrate the models
def run_multiclass_comparison(clf_name, clf):

    print(clf_name + ":")
    log_loss_list = []
    brier_loss_list = []
    acc_list = []
    ece_list = []

    print("base")
    ___  # TODO
    acc_list, log_loss_list, brier_loss_list, ece_list = ___  # TODO

    print("sigmoid")
    ___  # TODO
    cal_sigm = ___  # TODO
    ___  # TODO
    acc_list, log_loss_list, brier_loss_list, ece_list = ___  # TODO

    print("isotonic")
    cal_iso = ___  # TODO
    ___  # TODO
    acc_list, log_loss_list, brier_loss_list, ece_list = ___  # TODO

    print("sigmoid_cv")
    cal_sigm_cv = ___  # TODO
    ___  # TODO
    acc_list, log_loss_list, brier_loss_list, ece_list = ___  # TODO

    print("isotonic_cv")
    cal_iso_cv = ___  # TODO
    ___  # TODO
    acc_list, log_loss_list, brier_loss_list, ece_list = ___  # TODO

    print("ivap")
    va = ___  # TODO
    ___  # TODO
    acc_list, log_loss_list, brier_loss_list, ece_list = ___  # TODO

    print("cvap \n")
    va_cv = ___  # TODO
    ___  # TODO
    acc_list, log_loss_list, brier_loss_list, ece_list = ___  # TODO

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
# Step 11 : Compare models on multiclass classification
print("Comparing models for multiclass classification")
results_brier = pd.DataFrame()
results_log = pd.DataFrame()
results_acc = pd.DataFrame()
results_ece = pd.DataFrame()

for ___ in ___:  # TODO
    scratch_b, scratch_l, scratch_acc, scratch_ece = ___  # TODO
    results_brier = ___  # TODO
    results_log = ___  # TODO
    results_acc = ___  # TODO
    results_ece = ___  # TODO


# %%
# Step 12 : Define the function to convert the dataframe to a markdown table
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
# Step 13 : Display the results
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
