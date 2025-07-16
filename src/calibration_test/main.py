import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, log_loss, f1_score
from sklearn.frozen import FrozenEstimator
from venn_abers import VennAbersCalibrator
import calibration as cal
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

df = pd.read_csv("hf://datasets/mihaicata/diabetes/all_data_processed.tsv", sep="\t")

# Nettoyage et conversion robuste de la colonne 'inputs'
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

# Mélange du DataFrame pour éviter l'ordre par classe
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.head(10)

X = df.drop(columns=["label"])
y = df["label"]

# Standardisation des données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, shuffle=True, test_size=0.99, random_state=42, stratify=y
)

X_proper_train, X_cal, y_proper_train, y_cal = train_test_split(
    X_train, y_train, shuffle=True, test_size=0.2, random_state=42, stratify=y
)

# Étape 2 : Définir les modèles à tester
print("Définition des modèles à tester")
models = {
    "Logistic Regression": LogisticRegression(max_iter=5000),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "SVM (probability = False)": SVC(probability=False),
}
for name in models.keys():
    print(f"- {name}")
results = {}

# Étape 3 : Évaluer chaque modèle sans calibration, a l'exception de SVM sans probabilités
for name, model in models.items():
    print(f"\n Entraînement du modèle : {name}")
    model.fit(X_train, y_train)
    if name == "SVM (probability = False)":
        # Utilisation de decision_function et application de la sigmoïde
        decision = model.decision_function(X_test)
        probs = 1 / (1 + np.exp(-decision))
        y_pred = (probs > 0.5).astype(int)
    else:
        probs = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
    acc = f1_score(y_test, y_pred)
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
            results[name_method] = {
                "probs": probs,
                "accuracy": acc,
                "brier": brier,
                "logloss": logloss,
                "ece": ece,
            }
        name_method = f"{name} + Venn-ABERS"
        print(f"\n Calibration de {name} avec méthode Venn-ABERS")
        if name == "SVM (probability = False)":
            base_model = SVC(probability=False)
            base_model.fit(X_proper_train, y_proper_train)
            # Utiliser decision_function + sigmoïde pour obtenir des "pseudo-probas"
            p_cal = 1 / (1 + np.exp(-base_model.decision_function(X_cal)))
            p_test = 1 / (1 + np.exp(-base_model.decision_function(X_test)))
            # VennAbersCalibrator attend des tableaux 2D
            p_cal = np.vstack([1 - p_cal, p_cal]).T
            p_test = np.vstack([1 - p_test, p_test]).T
        else:
            base_model = models[name]
            base_model.fit(X_proper_train, y_proper_train)
            p_cal = base_model.predict_proba(X_cal)
            p_test = base_model.predict_proba(X_test)
        va = VennAbersCalibrator()
        y_cal_arr = np.array(y_cal)
        probs = va.predict_proba(p_cal=p_cal, y_cal=y_cal_arr, p_test=p_test)[:, 1]
        y_pred = (probs > 0.5).astype(int)
        acc = f1_score(y_test, y_pred)
        brier = brier_score_loss(y_test, probs)
        logloss = log_loss(y_test, probs)
        ece = cal.get_calibration_error(probs, y_test)
        print(f"Score Accuracy: {acc:.3f}")
        print(f"Brier Score: {brier:.3f}")
        print(f"Log Loss: {logloss:.3f}")
        print(f"Expected Calibration Error: {ece:.3f}")
        results[name_method] = {
            "probs": probs,
            "accuracy": acc,
            "brier": brier,
            "logloss": logloss,
            "ece": ece,
        }

    return results


calibration()


# Step 5: Plot calibration curves
def plot():
    print("\n Génération des courbes de calibration 1/2")
    os.makedirs("plots", exist_ok=True)
    plt.figure(figsize=(10, 8))
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
    plt.figure(figsize=(10, 8))
    for name in [
        "SVM (probability = False)",
        "SVM (probability = False + sigmoid)",
        "SVM (probability = False + isotonic)",
        "SVM (probability = False + Venn-ABERS)",
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
    ).sort_index()  # Trie par ordre alphabétique de name

    # Forcer l'affichage de toutes les lignes
    pd.set_option("display.max_rows", None)
    pd.set_option(
        "display.width", 0
    )  # Affiche sur toute la largeur disponible dans marimo

    print("\nSummary of model scores:")
    print(score_df)
    return


plot()
