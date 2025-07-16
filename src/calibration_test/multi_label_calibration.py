import marimo

__generated_with = "0.14.11"
app = marimo.App(width="medium")


@app.cell
def _():
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

    return (
        CalibratedClassifierCV,
        MultiOutputClassifier,
        RandomForestClassifier,
        accuracy_score,
        arff,
        brier_score_loss,
        cal,
        np,
        pd,
        plt,
        sns,
        train_test_split,
    )


@app.cell
def _(arff, np, pd, train_test_split):
    # Load and prepare the Water Quality dataset for multi-label classification
    # The dataset contains water quality measurements and binary indicators for 14 different quality parameters

    random_state = 6

    data = arff.loadarff("data/dataset_multilabel.arff")
    df = pd.DataFrame(data[0])

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    X = df[numeric_cols]
    print(X.head())

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

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", 50)

    print("\nX_train head:")
    print(X_train.head().to_string())

    print("\ny_train head:")
    print(y_train.head().to_string())

    print("\nX_test head:")
    print(X_test.head().to_string())

    print("\ny_test head:")
    print(y_test.head().to_string())

    print("\nX_proper_train head:")
    print(X_proper_train.head().to_string())

    print("\ny_proper_train head:")
    print(y_proper_train.head().to_string())

    print("\nX_cal head:")
    print(X_cal.head().to_string())

    print("\ny_cal head:")
    print(y_cal.head().to_string())

    return X_test, X_train, random_state, y, y_test, y_train


@app.cell
def _(
    MultiOutputClassifier,
    RandomForestClassifier,
    X_test,
    X_train,
    random_state,
    y_train,
):
    # Multi-output classifier non calibrated
    base_clf = RandomForestClassifier(random_state=random_state)
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
    # Calibration by output
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
def _(
    accuracy_score,
    brier_score_loss,
    cal,
    pd,
    pred_probs_calibrated_matrix,
    pred_probs_uncalibrated_matrix,
    pred_y_calibrated_matrix,
    pred_y_uncalibrated_matrix,
    y,
    y_test,
):
    # Compute Brier scores
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
        cal.get_calibration_error(
            pred_probs_uncalibrated_matrix[:, i], y_test.iloc[:, i]
        )
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
    return accuracy_score_df, brier_score_df, ece_score_df


@app.cell
def _(accuracy_score_df, brier_score_df, ece_score_df, plt, sns):
    # Visualisation
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
    plt.title(
        "Expected Calibration Error (ECE) per Label - Before and After Calibration"
    )
    plt.tight_layout()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
