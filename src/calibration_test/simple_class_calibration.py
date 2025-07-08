import marimo

__generated_with = "0.14.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.naive_bayes import GaussianNB
    from sklearn.calibration import CalibratedClassifierCV, calibration_curve
    from sklearn.metrics import brier_score_loss, log_loss, f1_score
    from sklearn.frozen import FrozenEstimator
    from venn_abers import VennAbersCalibrator
    import calibration as cal
    from sklearn.preprocessing import StandardScaler
    from sklearn.utils import resample
    import warnings

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    return (
        CalibratedClassifierCV,
        FrozenEstimator,
        GaussianNB,
        LogisticRegression,
        RandomForestClassifier,
        SVC,
        StandardScaler,
        VennAbersCalibrator,
        brier_score_loss,
        cal,
        calibration_curve,
        f1_score,
        log_loss,
        np,
        os,
        pd,
        plt,
        resample,
        train_test_split,
    )


@app.cell
def _(StandardScaler, pd, resample, train_test_split):
    # Étape 1 : Génération d'un jeu de données binaire
    print("Génération d'un jeu de données synthétique")

    random_state = 42

    df = pd.read_csv(
        "hf://datasets/mihaicata/diabetes/all_data_processed.tsv", sep="\t"
    )
    inputs_expanded = (
        df["inputs"]
        .astype(str)
        .str.replace(" ", "")
        .str.replace("[", "")
        .str.replace("]", "")
        .str.split(",", expand=True)
    )

    inputs_expanded.columns = [f"input_{i}" for i in range(inputs_expanded.shape[1])]
    inputs_expanded = inputs_expanded.dropna(axis=1)
    df = pd.concat([inputs_expanded, df["label"]], axis=1)

    # Équilibrage des classes par sur-échantillonnage
    majority = df[df.label == df.label.value_counts().idxmax()]
    minority = df[df.label == df.label.value_counts().idxmin()]
    minority_upsampled = resample(
        minority, replace=True, n_samples=len(majority), random_state=42
    )

    df = pd.concat([majority, minority_upsampled])

    X = df.drop(columns=["label"])
    y = df["label"]

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    print(df.head(10))

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, shuffle=True, test_size=0.99, random_state=random_state, stratify=y
    )

    X_proper_train, X_cal, y_proper_train, y_cal = train_test_split(
        X_train,
        y_train,
        shuffle=True,
        test_size=0.2,
        random_state=random_state,
        stratify=y_train,
    )
    return (
        X_cal,
        X_proper_train,
        X_test,
        X_train,
        random_state,
        y_cal,
        y_proper_train,
        y_test,
        y_train,
    )


@app.cell
def _(LogisticRegression, RandomForestClassifier, SVC, random_state):
    def define_model():
        # Étape 2 : Définir les modèles à tester
        print("Définition des modèles à tester")
        models = {
            "Logistic Regression": LogisticRegression(
                max_iter=5000, random_state=random_state
            ),
            "Random Forest": RandomForestClassifier(n_estimators=10),
            "SVM (probability = False)": SVC(probability=False),
        }
        for name in models.keys():
            print(f"- {name}")
        results = {}
        return models, results

    models, results = define_model()
    return models, results


@app.cell
def _(
    X_test,
    X_train,
    brier_score_loss,
    cal,
    f1_score,
    log_loss,
    models,
    np,
    results,
    y_test,
    y_train,
):
    # Étape 3 : Évaluer chaque modèle sans calibration, a l'exception de SVM sans probabilités
    for name, model in models.items():
        print(f"\n Entraînement du modèle : {name}")
        model.fit(X_train, y_train)
        if name == "SVM (probability = False)":
            # Utilisation de decision_function et application de la sigmoïde
            decision = model.decision_function(X_test)
            probs = 1 / (1 + np.exp(-decision))
        else:
            probs = model.predict_proba(X_test)[:, 1]
        acc = f1_score(y_test, model.predict(X_test))
        brier = brier_score_loss(y_test, probs)
        logloss = log_loss(y_test, probs)
        ece = cal.get_calibration_error(probs, y_test)
        print(f"Score Accuracy: {acc:.3f}")
        print(f"Brier Score: {brier:.3f}")
        print(f"Log Loss: {logloss:.3f}")
        print(f"Expected Calibration Error: {ece:.3f}")
        results[name] = {
            "model": model,
            "accuracy": acc,
            "probs": probs,
            "brier": brier,
            "logloss": logloss,
            "ece": ece,
        }
    return


@app.cell
def _(
    CalibratedClassifierCV,
    FrozenEstimator,
    VennAbersCalibrator,
    X_cal,
    X_proper_train,
    X_test,
    brier_score_loss,
    cal,
    f1_score,
    log_loss,
    models,
    np,
    results,
    y_cal,
    y_proper_train,
    y_test,
):
    # Étape 4 : Calibration explicite avec Platt(sigmoid) et Isotonic
    def calibration():
        for name in models.keys():
            for method in ["sigmoid", "isotonic"]:
                name_method = f"{name} + {method}"
                base_model = models[name].fit(X_proper_train, y_proper_train)
                calibrated_model = CalibratedClassifierCV(
                    estimator=FrozenEstimator(base_model), method=method
                )
                print(f"\n Calibration de {name} avec méthode {method}")
                calibrated_model.fit(X_cal, y_cal)
                probs = calibrated_model.predict_proba(X_test)[:, 1]
                y_pred = calibrated_model.predict(X_test)
                acc = f1_score(y_test, y_pred)
                brier = brier_score_loss(y_test, probs)
                logloss = log_loss(y_test, probs)
                ece = cal.get_calibration_error(probs, y_test)
                print(f"Score Accuracy: {acc:.3f}")
                print(f"Brier Score: {brier:.3f}")
                print(f"Log Loss: {logloss:.3f}")
                print(f"Expected Calibration Error: {ece:.3f}")
                print(probs.max())
                results[name_method] = {
                    "probs": probs,
                    "accuracy": acc,
                    "brier": brier,
                    "logloss": logloss,
                    "ece": ece,
                }
            name_method = f"{name} + Venn-ABERS"
            print(f"\n Calibration de {name} avec méthode Venn-ABERS")
            base_model = models[name].fit(X_proper_train, y_proper_train)
            if name == "SVM (probability = False)":
                # Utiliser decision_function + sigmoïde pour obtenir des "pseudo-probas"
                p_cal = 1 / (1 + np.exp(-base_model.decision_function(X_cal)))
                p_test = 1 / (1 + np.exp(-base_model.decision_function(X_test)))
                # VennAbersCalibrator attend des tableaux 2D
                p_cal = np.vstack([1 - p_cal, p_cal]).T
                p_test = np.vstack([1 - p_test, p_test]).T
            else:
                p_cal = base_model.predict_proba(X_cal)
                p_test = base_model.predict_proba(X_test)
            va = VennAbersCalibrator()
            probs = va.predict_proba(p_cal=p_cal, y_cal=np.array(y_cal), p_test=p_test)[
                :, 1
            ]
            y_pred = va.predict(p_cal=p_cal, y_cal=np.array(y_cal), p_test=p_test)[:, 1]
            acc = f1_score(y_test, y_pred)
            brier = brier_score_loss(y_test, probs)
            logloss = log_loss(y_test, probs)
            ece = cal.get_calibration_error(probs, y_test)
            print(f"Score Accuracy: {acc:.3f}")
            print(f"Brier Score: {brier:.3f}")
            print(f"Log Loss: {logloss:.3f}")
            print(f"Expected Calibration Error: {ece:.3f}")
            print(probs.max())
            results[name_method] = {
                "probs": probs,
                "accuracy": acc,
                "brier": brier,
                "logloss": logloss,
                "ece": ece,
            }
        return results

    calibration()
    return


@app.cell
def _(calibration_curve, os, pd, plt, results, y_test):
    # Step 5: Plot calibration curves
    def plot():
        print("\n Génération des courbes de calibration 1/2")
        os.makedirs("plots", exist_ok=True)
        plt.figure(figsize=(8, 6))
        for name in [
            "Random Forest",
            "Random Forest + sigmoid",
            "Random Forest + isotonic",
            "Random Forest + Venn-ABERS",
        ]:
            res = results[name]
            prob_true, prob_pred = calibration_curve(y_test, res["probs"], n_bins=10)
            plt.plot(prob_pred, prob_true, marker="o", label=name)

        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("Probabilité prédite")
        plt.ylabel("Fréquence observée")
        plt.title("Courbes de calibration")
        plt.legend(loc="best")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        print("\n Génération des courbes de calibration 2/2")
        plt.figure(figsize=(8, 6))
        for name in [
            "SVM (probability = False)",
            "SVM (probability = False) + sigmoid",
            "SVM (probability = False) + isotonic",
            "SVM (probability = False) + Venn-ABERS",
        ]:
            res = results[name]
            prob_true, prob_pred = calibration_curve(y_test, res["probs"], n_bins=10)
            plt.plot(prob_pred, prob_true, marker="o", label=name)

        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("Probabilité prédite")
        plt.ylabel("Fréquence observée")
        plt.title("Courbes de calibration")
        plt.legend(loc="best")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        score_df = pd.DataFrame.from_dict(
            {
                name: {
                    "Accuracy": res["accuracy"],
                    "Brier Score": res["brier"],
                    "Log Loss": res["logloss"],
                    "ECE": res["ece"],
                }
                for name, res in results.items()
            },
            orient="index",
        ).sort_index()

        pd.set_option("display.max_rows", None)
        pd.set_option("display.width", 0)

        print("\nSummary of model scores:")
        print(score_df)
        return

    plot()
    return


@app.cell
def _(
    CalibratedClassifierCV,
    FrozenEstimator,
    GaussianNB,
    VennAbersCalibrator,
    X_test,
    X_train,
    brier_score_loss,
    cal,
    f1_score,
    log_loss,
    np,
    pd,
    train_test_split,
    y_test,
    y_train,
):
    def _():
        # Prepare results storage
        metrics = ["accuracy", "brier", "log loss", "ece"]
        methods = [
            "Uncalibrated",
            "IVAP",
            "CVAP",
            "Prefit",
            "Isotonic",
            "Isotonic prefit",
            "Sigmoid",
            "Sigmoid prefit",
        ]
        results = {m: [] for m in metrics}

        # Uncalibrated
        clf = GaussianNB()
        clf.fit(X_train, y_train)
        clf_prob = clf.predict_proba(X_test)[:, 1]
        results["accuracy"].append(f1_score(y_test, clf.predict(X_test)))
        results["brier"].append(brier_score_loss(y_test, clf_prob))
        results["log loss"].append(log_loss(y_test, clf_prob))
        results["ece"].append(cal.get_calibration_error(clf_prob, y_test))

        # IVAP
        va = VennAbersCalibrator(
            estimator=GaussianNB(), inductive=True, cal_size=0.2, shuffle=False
        )
        va.fit(X_train, y_train)
        va_inductive_prob = va.predict_proba(X_test)[:, 1]
        results["accuracy"].append(f1_score(y_test, va.predict(X_test)[:, 1]))
        results["brier"].append(brier_score_loss(y_test, va_inductive_prob))
        results["log loss"].append(log_loss(y_test, va_inductive_prob))
        results["ece"].append(cal.get_calibration_error(va_inductive_prob, y_test))

        # CVAP
        va = VennAbersCalibrator(estimator=GaussianNB(), inductive=False, n_splits=2)
        va.fit(X_train, y_train)
        va_cv_prob = va.predict_proba(X_test)[:, 1]
        results["accuracy"].append(f1_score(y_test, va.predict(X_test)[:, 1]))
        results["brier"].append(brier_score_loss(y_test, va_cv_prob))
        results["log loss"].append(log_loss(y_test, va_cv_prob))
        results["ece"].append(cal.get_calibration_error(va_cv_prob, y_test))

        # Prefit
        X_train_proper, X_cal, y_train_proper, y_cal = train_test_split(
            X_train, y_train, test_size=0.2, shuffle=False
        )
        clf.fit(X_train_proper, y_train_proper)
        p_cal = clf.predict_proba(X_cal)
        p_test = clf.predict_proba(X_test)
        va = VennAbersCalibrator()
        va_prefit_prob = va.predict_proba(
            p_cal=p_cal, y_cal=np.array(y_cal), p_test=p_test
        )[:, 1]
        y_pred = va.predict(p_cal=p_cal, y_cal=np.array(y_cal), p_test=p_test)[:, 1]
        results["accuracy"].append(f1_score(y_test, y_pred))
        results["brier"].append(brier_score_loss(y_test, va_prefit_prob))
        results["log loss"].append(log_loss(y_test, va_prefit_prob))
        results["ece"].append(cal.get_calibration_error(va_prefit_prob, y_test))

        # Isotonic (cv=5)
        iso = CalibratedClassifierCV(GaussianNB(), method="isotonic", cv=5)
        iso.fit(X_train, y_train)
        iso_prob = iso.predict_proba(X_test)[:, 1]
        results["accuracy"].append(f1_score(y_test, iso.predict(X_test)))
        results["brier"].append(brier_score_loss(y_test, iso_prob))
        results["log loss"].append(log_loss(y_test, iso_prob))
        results["ece"].append(cal.get_calibration_error(iso_prob, y_test))

        # Isotonic prefit
        clf.fit(X_train_proper, y_train_proper)
        iso_prefit = CalibratedClassifierCV(FrozenEstimator(clf), method="isotonic")
        iso_prefit.fit(X_cal, y_cal)
        iso_prefit_prob = iso_prefit.predict_proba(X_test)[:, 1]
        results["accuracy"].append(f1_score(y_test, iso_prefit.predict(X_test)))
        results["brier"].append(brier_score_loss(y_test, iso_prefit_prob))
        results["log loss"].append(log_loss(y_test, iso_prefit_prob))
        results["ece"].append(cal.get_calibration_error(iso_prefit_prob, y_test))

        # Sigmoid (cv=5)
        sig = CalibratedClassifierCV(GaussianNB(), method="sigmoid", cv=5)
        sig.fit(X_train, y_train)
        sig_prob = sig.predict_proba(X_test)[:, 1]
        results["accuracy"].append(f1_score(y_test, sig.predict(X_test)))
        results["brier"].append(brier_score_loss(y_test, sig_prob))
        results["log loss"].append(log_loss(y_test, sig_prob))
        results["ece"].append(cal.get_calibration_error(sig_prob, y_test))

        # Sigmoid prefit
        clf.fit(X_train_proper, y_train_proper)
        sig_prefit = CalibratedClassifierCV(FrozenEstimator(clf), method="sigmoid")
        sig_prefit.fit(X_cal, y_cal)
        sig_prefit_prob = sig_prefit.predict_proba(X_test)[:, 1]
        results["accuracy"].append(f1_score(y_test, sig_prefit.predict(X_test)))
        results["brier"].append(brier_score_loss(y_test, sig_prefit_prob))
        results["log loss"].append(log_loss(y_test, sig_prefit_prob))
        results["ece"].append(cal.get_calibration_error(sig_prefit_prob, y_test))

        # Display results
        df_loss = pd.DataFrame(results, index=methods).T.round(3)
        return print(df_loss)

    _()
    return


if __name__ == "__main__":
    app.run()
