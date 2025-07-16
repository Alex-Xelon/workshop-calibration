import pandas as pd
import numpy as np
from sklearn.metrics import log_loss, brier_score_loss, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.datasets import make_classification
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


# data = arff.loadarff('data/dataset_186_satimage.arff')
# df = pd.DataFrame(data[0])
# df['class'] = ([int(str(i)[2]) for i in df['class'].values])
# df['class'] = df['class'].astype('int')-1


# df_train_cal, df_test = train_test_split(df, test_size = 2000, random_state = 42, shuffle = False)
# df_proper_train, df_cal = train_test_split(df_train_cal, test_size = 2000, random_state = 42, shuffle = False)


# X_train = df_train_cal.drop('class', axis=1)
# y_train = df_train_cal['class']

# X_proper_train = df_proper_train.drop('class', axis=1)
# y_proper_train = df_proper_train['class']

# X_cal = df_cal.drop('class', axis=1)
# y_cal = df_cal['class']

# X_test = df_test.drop('class', axis=1)
# y_test = df_test['class'].replace(6, 5)


random_state = 1
np.random.seed(seed=1)

X, y = make_classification(
    n_samples=100000,
    n_features=20,
    n_informative=4,
    n_redundant=2,
    random_state=random_state,
    n_classes=6,
)
X = pd.DataFrame(X)
y = pd.Series(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.99)

X_proper_train, X_cal, y_proper_train, y_cal = train_test_split(
    X_train, y_train, shuffle=False, test_size=0.2
)

lb = LabelBinarizer()
y_test_binary = lb.fit_transform(y_test)

clfs = {}
clfs["Naive Bayes"] = GaussianNB()
clfs["SVM"] = SVC(probability=True)
clfs["RF"] = RandomForestClassifier()
clfs["XGB"] = AdaBoostClassifier()
clfs["Logistic"] = LogisticRegression(max_iter=10000)
clfs["Neural Network"] = MLPClassifier(max_iter=10000)


def run_multiclass_comparison(clf_name, clf):

    print(clf_name + ":")
    log_loss_list = []
    brier_loss_list = []
    acc_list = []
    ece_list = []

    print("base")
    clf.fit(X_train, y_train)
    p_pred = clf.predict_proba(X_test)
    y_pred = clf.predict(X_test)
    acc_list.append(accuracy_score(y_test, y_pred))
    log_loss_list.append(log_loss(y_test, p_pred))
    brier_loss_list.append(brier_score_loss(y_test_binary, p_pred))
    ece_list.append(cal.get_calibration_error(p_pred, y_test))

    print("sigmoid")
    clf.fit(X_proper_train, y_proper_train)
    cal_sigm = CalibratedClassifierCV(clf, method="sigmoid", cv="prefit")
    cal_sigm.fit(X_cal, y_cal)
    p_pred = cal_sigm.predict_proba(X_test)
    y_pred = cal_sigm.predict(X_test)
    acc_list.append(accuracy_score(y_test, y_pred))
    log_loss_list.append(log_loss(y_test, p_pred))
    brier_loss_list.append(brier_score_loss(y_test_binary, p_pred))
    ece_list.append(cal.get_calibration_error(p_pred, y_test))

    print("isotonic")
    cal_iso = CalibratedClassifierCV(clf, method="isotonic", cv="prefit")
    cal_iso.fit(X_cal, y_cal)
    p_pred = cal_iso.predict_proba(X_test)
    y_pred = cal_iso.predict(X_test)
    acc_list.append(accuracy_score(y_test, y_pred))
    log_loss_list.append(log_loss(y_test, p_pred))
    brier_loss_list.append(brier_score_loss(y_test_binary, p_pred))
    ece_list.append(cal.get_calibration_error(p_pred, y_test))

    print("sigmoid_cv")
    cal_sigm_cv = CalibratedClassifierCV(clf, method="sigmoid", cv=5)
    cal_sigm_cv.fit(X_train, y_train)
    p_pred = cal_sigm_cv.predict_proba(X_test)
    y_pred = cal_sigm_cv.predict(X_test)
    acc_list.append(accuracy_score(y_test, y_pred))
    log_loss_list.append(log_loss(y_test, p_pred))
    brier_loss_list.append(brier_score_loss(y_test_binary, p_pred))
    ece_list.append(cal.get_calibration_error(p_pred, y_test))

    print("isotonic_cv")
    cal_iso_cv = CalibratedClassifierCV(clf, method="isotonic", cv=5)
    cal_iso_cv.fit(X_train, y_train)
    p_pred = cal_iso_cv.predict_proba(X_test)
    y_pred = cal_iso_cv.predict(X_test)
    acc_list.append(accuracy_score(y_test, y_pred))
    log_loss_list.append(log_loss(y_test, p_pred))
    brier_loss_list.append(brier_score_loss(y_test_binary, p_pred))
    ece_list.append(cal.get_calibration_error(p_pred, y_test))

    print("ivap")
    va = VennAbersCalibrator(clf, inductive=True, cal_size=0.2, random_state=42)
    va.fit(np.asarray(X_train), np.asarray(y_train))
    p_pred_va = va.predict_proba(np.array(X_test))
    y_pred = va.predict(np.array(X_test), one_hot=False)
    acc_list.append(accuracy_score(y_test, y_pred))
    log_loss_list.append(log_loss(y_test, p_pred_va))
    brier_loss_list.append(brier_score_loss(y_test_binary, p_pred_va))
    ece_list.append(cal.get_calibration_error(p_pred_va, y_test))

    print("cvap")
    va_cv = VennAbersCalibrator(clf, inductive=False, n_splits=5)
    va_cv.fit(np.asarray(X_train), np.asarray(y_train))
    p_pred_cv = va_cv.predict_proba(np.asarray(X_test))
    y_pred = va_cv.predict(np.array(X_test), one_hot=False)
    acc_list.append(accuracy_score(y_test, y_pred))
    log_loss_list.append(log_loss(y_test, p_pred_cv))
    brier_loss_list.append(brier_score_loss(y_test_binary, p_pred_cv))
    ece_list.append(cal.get_calibration_error(p_pred_cv, y_test))

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
    scratch_b, scratch_l, scratch_acc, scratch_ece = run_multiclass_comparison(
        clf_name, clfs[clf_name]
    )
    results_brier = pd.concat((results_brier, scratch_b), ignore_index=True)
    results_log = pd.concat((results_log, scratch_l), ignore_index=True)
    results_acc = pd.concat((results_acc, scratch_acc), ignore_index=True)
    results_ece = pd.concat((results_ece, scratch_ece), ignore_index=True)


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
