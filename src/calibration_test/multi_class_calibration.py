import marimo

__generated_with = "0.14.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
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
    return (
        AdaBoostClassifier,
        CalibratedClassifierCV,
        GaussianNB,
        LogisticRegression,
        MLPClassifier,
        RandomForestClassifier,
        SVC,
        VennAbersCalibrator,
        arff,
        brier_score_loss,
        cal,
        f1_score,
        log_loss,
        mo,
        np,
        pd,
        train_test_split,
    )


@app.cell
def _(arff, pd, random, train_test_split):
    random.seed(1)

    data = arff.loadarff("data/dataset_multiclass.arff")
    df = pd.DataFrame(data[0])

    X = df.drop("Class", axis=1)
    y = df["Class"].astype(int).subtract(1)

    print(y.value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.90,
        shuffle=False,
    )

    X_proper_train, X_cal, y_proper_train, y_cal = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        shuffle=False,
    )

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)

    print(X_train.head())
    print(y_train.head())
    print(X_test.head())
    print(y_test.head())
    print(X_proper_train.head())
    print(y_proper_train.head())
    print(X_cal.head())
    print(y_cal.head())

    return (
        X_cal,
        X_proper_train,
        X_test,
        X_train,
        y_cal,
        y_proper_train,
        y_test,
        y_train,
    )


@app.cell
def _(
    AdaBoostClassifier,
    GaussianNB,
    LogisticRegression,
    MLPClassifier,
    RandomForestClassifier,
    SVC,
):
    clfs = {}
    clfs["Naive Bayes"] = GaussianNB()
    clfs["SVM"] = SVC(probability=True)
    clfs["RF"] = RandomForestClassifier()
    clfs["XGB"] = AdaBoostClassifier()
    clfs["Logistic"] = LogisticRegression(max_iter=10000)
    clfs["Neural Network"] = MLPClassifier(max_iter=10000)
    return (clfs,)


@app.cell
def _(
    CalibratedClassifierCV,
    VennAbersCalibrator,
    X_cal,
    X_proper_train,
    X_test,
    X_train,
    brier_score_loss,
    cal,
    f1_score,
    log_loss,
    np,
    pd,
    y_cal,
    y_proper_train,
    y_test,
    y_train,
):
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
        acc_list.append(f1_score(y_test, y_pred, average="weighted"))
        log_loss_list.append(log_loss(y_test, p_pred))
        brier_loss_list.append(brier_score_loss(y_test, p_pred))
        ece_list.append(cal.get_calibration_error(p_pred, y_test))

        print("sigmoid")
        clf.fit(X_proper_train, y_proper_train)
        cal_sigm = CalibratedClassifierCV(clf, method="sigmoid", cv="prefit")
        cal_sigm.fit(X_cal, y_cal)
        p_pred = cal_sigm.predict_proba(X_test)
        y_pred = cal_sigm.predict(X_test)
        acc_list.append(f1_score(y_test, y_pred, average="weighted"))
        log_loss_list.append(log_loss(y_test, p_pred))
        brier_loss_list.append(brier_score_loss(y_test, p_pred))
        ece_list.append(cal.get_calibration_error(p_pred, y_test))

        print("isotonic")
        cal_iso = CalibratedClassifierCV(clf, method="isotonic", cv="prefit")
        cal_iso.fit(X_cal, y_cal)
        p_pred = cal_iso.predict_proba(X_test)
        y_pred = cal_iso.predict(X_test)
        acc_list.append(f1_score(y_test, y_pred, average="weighted"))
        log_loss_list.append(log_loss(y_test, p_pred))
        brier_loss_list.append(brier_score_loss(y_test, p_pred))
        ece_list.append(cal.get_calibration_error(p_pred, y_test))

        print("sigmoid_cv")
        cal_sigm_cv = CalibratedClassifierCV(clf, method="sigmoid", cv=5)
        cal_sigm_cv.fit(X_train, y_train)
        p_pred = cal_sigm_cv.predict_proba(X_test)
        y_pred = cal_sigm_cv.predict(X_test)
        acc_list.append(f1_score(y_test, y_pred, average="weighted"))
        log_loss_list.append(log_loss(y_test, p_pred))
        brier_loss_list.append(brier_score_loss(y_test, p_pred))
        ece_list.append(cal.get_calibration_error(p_pred, y_test))

        print("isotonic_cv")
        cal_iso_cv = CalibratedClassifierCV(clf, method="isotonic", cv=5)
        cal_iso_cv.fit(X_train, y_train)
        p_pred = cal_iso_cv.predict_proba(X_test)
        y_pred = cal_iso_cv.predict(X_test)
        acc_list.append(f1_score(y_test, y_pred, average="weighted"))
        log_loss_list.append(log_loss(y_test, p_pred))
        brier_loss_list.append(brier_score_loss(y_test, p_pred))
        ece_list.append(cal.get_calibration_error(p_pred, y_test))

        print("ivap")
        va = VennAbersCalibrator(clf, inductive=True, cal_size=0.2)
        va.fit(np.asarray(X_train), np.asarray(y_train))
        p_pred_va = va.predict_proba(np.array(X_test))
        y_pred = va.predict(np.array(X_test), one_hot=False)
        acc_list.append(f1_score(y_test, y_pred, average="weighted"))
        log_loss_list.append(log_loss(y_test, p_pred_va))
        brier_loss_list.append(brier_score_loss(y_test, p_pred_va))
        ece_list.append(cal.get_calibration_error(p_pred_va, y_test))

        print("cvap")
        va_cv = VennAbersCalibrator(clf, inductive=False, n_splits=5)
        va_cv.fit(np.asarray(X_train), np.asarray(y_train))
        p_pred_cv = va_cv.predict_proba(np.asarray(X_test))
        y_pred = va_cv.predict(np.array(X_test), one_hot=False)
        acc_list.append(f1_score(y_test, y_pred, average="weighted"))
        log_loss_list.append(log_loss(y_test, p_pred_cv))
        brier_loss_list.append(brier_score_loss(y_test, p_pred_cv))
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

    return (run_multiclass_comparison,)


@app.cell
def _(clfs, pd, run_multiclass_comparison):
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

    return results_acc, results_brier, results_ece, results_log


@app.cell
def _(results_acc, results_brier, results_ece, results_log):
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
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Accuracy Results\n"
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
            + f"{get_best_metric(results_ece, 'ECE', higher_is_better=False)}
        """
    )
    return


if __name__ == "__main__":
    app.run()
