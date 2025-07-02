import marimo

__generated_with = "0.14.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    import pandas as pd
    from sklearn.datasets import make_multilabel_classification
    from sklearn.model_selection import train_test_split
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.multioutput import MultiOutputClassifier
    from sklearn.metrics import brier_score_loss, accuracy_score
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.ensemble import RandomForestClassifier

    return (
        CalibratedClassifierCV,
        MultiOutputClassifier,
        RandomForestClassifier,
        accuracy_score,
        brier_score_loss,
        make_multilabel_classification,
        np,
        pd,
        plt,
        sns,
        train_test_split,
    )


@app.cell
def _(make_multilabel_classification, np, pd, train_test_split):
    # Simuler un dataset multi-label
    random_state = 1
    np.random.seed(seed=0)

    X, y = make_multilabel_classification(
        n_samples=10000,
        n_features=20,
        n_classes=5,
        n_labels=2,
        random_state=random_state,
    )
    y = pd.DataFrame(y, columns=[f"label_{i}" for i in range(y.shape[1])])

    # Découpage des données
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_test, X_train, y, y_test, y_train


@app.cell
def _(MultiOutputClassifier, RandomForestClassifier, X_test, X_train, y_train):
    # Classifieur multi-sortie non calibré
    base_clf = RandomForestClassifier(random_state=42)
    multi_clf = MultiOutputClassifier(base_clf)
    multi_clf.fit(X_train, y_train)
    pred_probs_uncalibrated = multi_clf.predict_proba(X_test)
    pred_y_uncalibrated = multi_clf.predict(X_test)
    return base_clf, pred_probs_uncalibrated, pred_y_uncalibrated


@app.cell
def _(
    CalibratedClassifierCV,
    X_test,
    X_train,
    base_clf,
    np,
    pred_probs_uncalibrated,
    pred_y_uncalibrated,
    y,
    y_train,
):
    # Calibration par sortie
    calibrated_clfs = []
    pred_probs_calibrated = []
    pred_y_calibrated = []
    for i in range(y.shape[1]):
        clf = CalibratedClassifierCV(base_clf, method="isotonic", cv=3)
        clf.fit(X_train, y_train.iloc[:, i])
        calibrated_clfs.append(clf)
        pred_probs_calibrated.append(clf.predict_proba(X_test)[:, 1])
        pred_y_calibrated.append(clf.predict(X_test))

    # Conversion en matrice numpy
    pred_probs_uncalibrated_matrix = np.vstack(
        [p[:, 1] for p in pred_probs_uncalibrated]
    ).T
    pred_probs_calibrated_matrix = np.vstack(pred_probs_calibrated).T

    pred_y_uncalibrated_matrix = np.column_stack(pred_y_uncalibrated).T
    pred_y_calibrated_matrix = np.column_stack(pred_y_calibrated)
    return (
        pred_probs_calibrated_matrix,
        pred_probs_uncalibrated_matrix,
        pred_y_calibrated_matrix,
        pred_y_uncalibrated_matrix,
    )


@app.cell
def _(np):
    def adaptive_ece(probs, labels, n_bins=10):
        """
        Compute the Expected Calibration Error using adaptive binning.

        Each bin contains approximately the same number of samples.
        """
        probs = np.asarray(probs)
        labels = np.asarray(labels)
        assert probs.shape == labels.shape

        # Sort by predicted probability
        sorted_indices = np.argsort(probs)
        sorted_probs = probs[sorted_indices]
        sorted_labels = labels[sorted_indices]

        # Bin edges to split the sorted predictions into equal-sized bins
        bin_edges = np.linspace(0, len(probs), n_bins + 1, dtype=int)

        ece = 0.0
        for i in range(n_bins):
            start = bin_edges[i]
            end = bin_edges[i + 1]
            if end > start:  # avoid empty bins
                bin_probs = sorted_probs[start:end]
                bin_labels = sorted_labels[start:end]
                avg_conf = np.mean(bin_probs)
                acc = np.mean(bin_labels)
                weight = len(bin_probs) / len(probs)
                ece += np.abs(avg_conf - acc) * weight

        return ece

    return (adaptive_ece,)


@app.cell
def _(
    accuracy_score,
    adaptive_ece,
    brier_score_loss,
    pd,
    pred_probs_calibrated_matrix,
    pred_probs_uncalibrated_matrix,
    pred_y_calibrated_matrix,
    pred_y_uncalibrated_matrix,
    y,
    y_test,
):
    # Calcul des Brier scores
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
        adaptive_ece(pred_probs_uncalibrated_matrix[:, i], y_test.iloc[:, i])
        for i in range(y.shape[1])
    ]
    ece_scores_calibrated = [
        adaptive_ece(
            pred_probs_calibrated_matrix[:, i],
            y_test.iloc[:, i],
        )
        for i in range(y.shape[1])
    ]

    # Résumé dans un DataFrame
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
    return accuracy_score_df, brier_score_df, ece_score_df


@app.cell
def _(accuracy_score_df, brier_score_df, ece_score_df, plt, sns):
    # Visualisation
    plt.figure(figsize=(10, 5))
    brier_long = brier_score_df.melt(
        id_vars="Label", var_name="Type", value_name="Brier Score"
    )
    sns.barplot(data=brier_long, x="Label", y="Brier Score", hue="Type")
    plt.title("Brier Score per Label - Before and After Calibration")
    plt.tight_layout()
    plt.show()

    # Plot Accuracy per label before and after calibration
    plt.figure(figsize=(10, 5))
    accuracy_long = accuracy_score_df.melt(
        id_vars="Label", var_name="Type", value_name="Accuracy"
    )
    sns.barplot(data=accuracy_long, x="Label", y="Accuracy", hue="Type")
    plt.title("Accuracy per Label - Before and After Calibration")
    plt.tight_layout()
    plt.show()

    # Plot ECE per label before and after calibration
    plt.figure(figsize=(10, 5))
    ece_long = ece_score_df.melt(id_vars="Label", var_name="Type", value_name="ECE")
    sns.barplot(data=ece_long, x="Label", y="ECE", hue="Type")
    plt.title(
        "Expected Calibration Error (ECE) per Label - Before and After Calibration"
    )
    plt.tight_layout()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
