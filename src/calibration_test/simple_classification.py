import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.metrics import log_loss, brier_score_loss, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from venn_abers import VennAbersCalibrator
import calibration as cal
from sklearn.calibration import CalibrationDisplay

import warnings

warnings.filterwarnings("ignore")


random_state = 1
np.random.seed(seed=1)

X, y = make_classification(
    n_samples=100000,
    n_features=20,
    n_informative=4,
    n_redundant=2,
    random_state=random_state,
)
X = pd.DataFrame(X)
y = pd.Series(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.99)

X_proper_train, X_cal, y_proper_train, y_cal = train_test_split(
    X_train, y_train, shuffle=False, test_size=0.2
)

clfs = {}
clfs["Naive Bayes"] = GaussianNB()
clfs["SVM"] = SVC(probability=True)
clfs["RF"] = RandomForestClassifier()
clfs["XGB"] = AdaBoostClassifier()
clfs["Logistic"] = LogisticRegression(max_iter=10000)
clfs["Neural Network"] = MLPClassifier(max_iter=10000)

fig = plt.figure(figsize=(36, 27))
gs = GridSpec(12, 9)
colors = plt.get_cmap("Dark2")
markers = ["^", "v", "s", "o", "D", "P", "*"]

grid_positions = [
    (slice(0, 3), slice(0, 3)),
    (slice(0, 3), slice(3, 6)),
    (slice(0, 3), slice(6, 9)),
    (slice(6, 9), slice(0, 3)),
    (slice(6, 9), slice(3, 6)),
    (slice(6, 9), slice(6, 9)),
]

hist_grid_positions = [
    (3, 1),
    (4, 0),
    (4, 1),
    (4, 2),
    (5, 0),
    (5, 1),
    (5, 2),
    (3, 4),
    (4, 3),
    (4, 4),
    (4, 5),
    (5, 3),
    (5, 4),
    (5, 5),
    (3, 7),
    (4, 6),
    (4, 7),
    (4, 8),
    (5, 6),
    (5, 7),
    (5, 8),
    (9, 1),
    (10, 0),
    (10, 1),
    (10, 2),
    (11, 0),
    (11, 1),
    (11, 2),
    (9, 4),
    (10, 3),
    (10, 4),
    (10, 5),
    (11, 3),
    (11, 4),
    (11, 5),
    (9, 7),
    (10, 6),
    (10, 7),
    (10, 8),
    (11, 6),
    (11, 7),
    (11, 8),
]


def run_multiclass_comparison(clf_name, clf):

    print(clf_name + ":")
    log_loss_list = []
    brier_loss_list = []
    acc_list = []
    ece_list = []
    calibration_displays = {}

    print("base")
    clf.fit(X_train, y_train)
    p_pred = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)
    acc_list.append(accuracy_score(y_test, y_pred))
    log_loss_list.append(log_loss(y_test, p_pred))
    brier_loss_list.append(brier_score_loss(y_test, p_pred))
    ece_list.append(cal.get_calibration_error(p_pred, y_test))
    display = CalibrationDisplay.from_estimator(
        clf,
        X_test,
        y_test,
        n_bins=10,
        name="Uncalibrated",
        ax=ax_calibration_curve,
        color=colors(0),
        marker=markers[0],
    )
    calibration_displays["Uncalibrated"] = display

    row, col = hist_grid_positions.pop(0)
    ax = fig.add_subplot(gs[row, col])
    ax.hist(
        p_pred,
        range=(0, 1),
        bins=10,
        label="Uncalibrated",
        color=colors(0),
    )
    ax.set(
        title=f"{clf_name} - Uncalibrated",
        xlabel="Mean predicted probability",
        ylabel="Count",
    )

    print("sigmoid")
    clf.fit(X_proper_train, y_proper_train)
    cal_sigm = CalibratedClassifierCV(clf, method="sigmoid")
    cal_sigm.fit(X_cal, y_cal)
    p_pred = cal_sigm.predict_proba(X_test)[:, 1]
    y_pred = cal_sigm.predict(X_test)
    acc_list.append(accuracy_score(y_test, y_pred))
    log_loss_list.append(log_loss(y_test, p_pred))
    brier_loss_list.append(brier_score_loss(y_test, p_pred))
    ece_list.append(cal.get_calibration_error(p_pred, y_test))
    display = CalibrationDisplay.from_predictions(
        y_test,
        p_pred,
        n_bins=10,
        name="sigmoid",
        ax=ax_calibration_curve,
        color=colors(1),
        marker=markers[1],
    )
    calibration_displays["sigmoid"] = display

    row, col = hist_grid_positions.pop(0)
    ax = fig.add_subplot(gs[row, col])
    ax.hist(
        p_pred,
        range=(0, 1),
        bins=10,
        label="sigmoid",
        color=colors(1),
    )
    ax.set(
        title=f"{clf_name} - sigmoid",
        xlabel="Mean predicted probability",
        ylabel="Count",
    )

    print("isotonic")
    cal_iso = CalibratedClassifierCV(clf, method="isotonic", cv="prefit")
    cal_iso.fit(X_cal, y_cal)
    p_pred = cal_iso.predict_proba(X_test)[:, 1]
    y_pred = cal_iso.predict(X_test)
    acc_list.append(accuracy_score(y_test, y_pred))
    log_loss_list.append(log_loss(y_test, p_pred))
    brier_loss_list.append(brier_score_loss(y_test, p_pred))
    ece_list.append(cal.get_calibration_error(p_pred, y_test))
    display = CalibrationDisplay.from_predictions(
        y_test,
        p_pred,
        n_bins=10,
        name="isotonic",
        ax=ax_calibration_curve,
        color=colors(2),
        marker=markers[2],
    )
    calibration_displays["isotonic"] = display

    row, col = hist_grid_positions.pop(0)
    ax = fig.add_subplot(gs[row, col])
    ax.hist(
        p_pred,
        range=(0, 1),
        bins=10,
        label="isotonic",
        color=colors(2),
    )
    ax.set(
        title=f"{clf_name} - isotonic",
        xlabel="Mean predicted probability",
        ylabel="Count",
    )

    print("sigmoid_cv")
    cal_sigm_cv = CalibratedClassifierCV(clf, method="sigmoid", cv=5)
    cal_sigm_cv.fit(X_train, y_train)
    p_pred = cal_sigm_cv.predict_proba(X_test)[:, 1]
    y_pred = cal_sigm_cv.predict(X_test)
    acc_list.append(accuracy_score(y_test, y_pred))
    log_loss_list.append(log_loss(y_test, p_pred))
    brier_loss_list.append(brier_score_loss(y_test, p_pred))
    ece_list.append(cal.get_calibration_error(p_pred, y_test))
    display = CalibrationDisplay.from_predictions(
        y_test,
        p_pred,
        n_bins=10,
        name="sigmoid_cv",
        ax=ax_calibration_curve,
        color=colors(3),
        marker=markers[3],
    )
    calibration_displays["sigmoid_cv"] = display

    row, col = hist_grid_positions.pop(0)
    ax = fig.add_subplot(gs[row, col])
    ax.hist(
        p_pred,
        range=(0, 1),
        bins=10,
        label="sigmoid_cv",
        color=colors(3),
    )
    ax.set(
        title=f"{clf_name} - sigmoid_cvs",
        xlabel="Mean predicted probability",
        ylabel="Count",
    )

    print("isotonic_cv")
    cal_iso_cv = CalibratedClassifierCV(clf, method="isotonic", cv=5)
    cal_iso_cv.fit(X_train, y_train)
    p_pred = cal_iso_cv.predict_proba(X_test)[:, 1]
    y_pred = cal_iso_cv.predict(X_test)
    acc_list.append(accuracy_score(y_test, y_pred))
    log_loss_list.append(log_loss(y_test, p_pred))
    brier_loss_list.append(brier_score_loss(y_test, p_pred))
    ece_list.append(cal.get_calibration_error(p_pred, y_test))
    display = CalibrationDisplay.from_predictions(
        y_test,
        p_pred,
        n_bins=10,
        name="isotonic_cv",
        ax=ax_calibration_curve,
        color=colors(4),
        marker=markers[4],
    )
    calibration_displays["isotonic_cv"] = display

    row, col = hist_grid_positions.pop(0)
    ax = fig.add_subplot(gs[row, col])
    ax.hist(
        p_pred,
        range=(0, 1),
        bins=10,
        label="isotonic_cv",
        color=colors(4),
    )
    ax.set(
        title=f"{clf_name} - isotonic_cv",
        xlabel="Mean predicted probability",
        ylabel="Count",
    )

    print("ivap")
    va = VennAbersCalibrator(clf, inductive=True, cal_size=0.2, random_state=42)
    va.fit(np.asarray(X_train), np.asarray(y_train))
    p_pred_va = va.predict_proba(np.array(X_test))[:, 1]
    y_pred = va.predict(np.array(X_test), one_hot=False)
    acc_list.append(accuracy_score(y_test, y_pred))
    log_loss_list.append(log_loss(y_test, p_pred_va))
    brier_loss_list.append(brier_score_loss(y_test, p_pred_va))
    ece_list.append(cal.get_calibration_error(p_pred_va, y_test))
    display = CalibrationDisplay.from_predictions(
        y_test,
        p_pred_va,
        n_bins=10,
        name="ivap",
        ax=ax_calibration_curve,
        color=colors(5),
        marker=markers[5],
    )
    calibration_displays["ivap"] = display

    row, col = hist_grid_positions.pop(0)
    ax = fig.add_subplot(gs[row, col])
    ax.hist(
        p_pred_va,
        range=(0, 1),
        bins=10,
        label="ivap",
        color=colors(5),
    )
    ax.set(
        title=f"{clf_name} - ivap", xlabel="Mean predicted probability", ylabel="Count"
    )

    print("cvap")
    va_cv = VennAbersCalibrator(clf, inductive=False, n_splits=5)
    va_cv.fit(np.asarray(X_train), np.asarray(y_train))
    p_pred_cv = va_cv.predict_proba(np.asarray(X_test))[:, 1]
    y_pred = va_cv.predict(np.array(X_test), one_hot=False)
    acc_list.append(accuracy_score(y_test, y_pred))
    log_loss_list.append(log_loss(y_test, p_pred_cv))
    brier_loss_list.append(brier_score_loss(y_test, p_pred_cv))
    ece_list.append(cal.get_calibration_error(p_pred_cv, y_test))
    display = CalibrationDisplay.from_predictions(
        y_test,
        p_pred_cv,
        n_bins=10,
        name="cvap",
        ax=ax_calibration_curve,
        color=colors(6),
        marker=markers[6],
    )
    calibration_displays["cvap"] = display

    row, col = hist_grid_positions.pop(0)
    ax = fig.add_subplot(gs[row, col])
    ax.hist(
        p_pred,
        range=(0, 1),
        bins=10,
        label="cvap",
        color=colors(6),
    )
    ax.set(
        title=f"{clf_name} - cvap", xlabel="Mean predicted probability", ylabel="Count"
    )

    print("")

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


results_brier = pd.DataFrame()
results_log = pd.DataFrame()
results_acc = pd.DataFrame()
results_ece = pd.DataFrame()

for clf_name in clfs:
    ax_calibration_curve = fig.add_subplot(gs[grid_positions.pop(0)])
    scratch_b, scratch_l, scratch_acc, scratch_ece = run_multiclass_comparison(
        clf_name, clfs[clf_name]
    )
    results_brier = pd.concat((results_brier, scratch_b), ignore_index=True)
    results_log = pd.concat((results_log, scratch_l), ignore_index=True)
    results_acc = pd.concat((results_acc, scratch_acc), ignore_index=True)
    results_ece = pd.concat((results_ece, scratch_ece), ignore_index=True)
    ax_calibration_curve.grid()
    ax_calibration_curve.set_title(f"{clf_name} Calibration Curves")


results_acc.set_index("Classifier", inplace=True)
print("Accuracy Results:")
print(results_acc.round(3))

results_brier.set_index("Classifier", inplace=True)
print("\nBrier Loss Results:")
print(results_brier.round(4))

results_log.set_index("Classifier", inplace=True)
print("\nLog Loss Results:")
print(results_log.round(4))

results_ece.set_index("Classifier", inplace=True)
print("\nECE Results:")
print(results_ece.round(4))

print("\nMean Accuracy across methods:")
print(results_acc.mean())

print("\nMean rank of Accuracy across classifiers:")
print(results_acc.rank(axis=1, ascending=False).mean())

print("\nMean Brier Loss across methods:")
print(results_brier.mean())

print("\nMean rank of Brier Loss across classifiers:")
print(results_brier.rank(axis=1).mean())

print("\nMean Log Loss across methods:")
print(results_log.mean())

print("\nMean rank of Log Loss across classifiers:")
print(results_log.rank(axis=1).mean())

print("\nMean ECE across methods:")
print(results_ece.mean())

print("\nMean rank of ECE across classifiers:")
print(results_ece.rank(axis=1).mean())

plt.tight_layout()
os.makedirs("plots", exist_ok=True)
plt.savefig("plots/calibration_simple_classification.png")
