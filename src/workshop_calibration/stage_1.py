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
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, log_loss, f1_score
from sklearn.frozen import FrozenEstimator
from venn_abers import VennAbersCalibrator
import xgboost as xgb
import calibration as cal
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from scipy.io import arff
import warnings

warnings.filterwarnings("ignore")

# %%
# Step 1 : Load the dataset
print("Loading the dataset")

random_state = 28
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

data = arff.loadarff("../../data/dataset_simpleclass.arff")
df = pd.DataFrame(data[0])

print(df.head(10))

# Balancing the classes by oversampling
majority = df[df.label == df.label.value_counts().idxmax()]
minority = df[df.label == df.label.value_counts().idxmin()]
minority_upsampled = resample(
    minority,
    replace=True,
    n_samples=len(majority),
    random_state=random_state,
)

df = pd.concat([majority, minority_upsampled])
df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

X = df.drop(columns=["label"])
y = df["label"].astype(int).values

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    shuffle=True,
    test_size=0.95,
    random_state=random_state,
    stratify=y,
)

X_proper_train, X_cal, y_proper_train, y_cal = train_test_split(
    X_train,
    y_train,
    shuffle=True,
    test_size=0.2,
    random_state=random_state,
    stratify=y_train,
)


# %%
# Step 2 : Define the models to test
print("Models to test")
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=5000,
        random_state=random_state,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=10,
        random_state=random_state,
    ),
    "XGBoost": xgb.XGBClassifier(
        tree_method="hist",
        random_state=random_state,
    ),
}
results = {}

for name_model in models.keys():
    print(f"- {name_model}")

# %%
# Step 3 : Evaluate each model without calibration
for name, model in models.items():
    print(f"\n Entraînement du modèle : {name}")
    model.fit(X_train, y_train)
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


# %%
# Step 4 : Explicit calibration with Platt (sigmoid), Isotonic and Venn-ABERS
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
    return


calibration()


# %%
# Step 5: Plot calibration curves
def plot():
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

    plt.figure(figsize=(8, 6))
    for name in [
        "XGBoost",
        "XGBoost + sigmoid",
        "XGBoost + isotonic",
        "XGBoost + Venn-ABERS",
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

    return score_df


plot()


# %%
# Step 6 : Compare the different calibration methods
def compare_methods():
    # Prepare results storage
    metrics = ["accuracy", "brier", "log loss", "ece"]
    methods = [
        "Uncalibrated",
        "Isotonic",
        "Isotonic prefit",
        "Sigmoid",
        "Sigmoid prefit",
        "Prefit Venn-Abers",
        "IVAP",
        "CVAP",
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

    # Isotonic (cv=5)
    iso = CalibratedClassifierCV(GaussianNB(), method="isotonic", cv=5)
    iso.fit(X_train, y_train)
    iso_prob = iso.predict_proba(X_test)[:, 1]
    results["accuracy"].append(f1_score(y_test, iso.predict(X_test)))
    results["brier"].append(brier_score_loss(y_test, iso_prob))
    results["log loss"].append(log_loss(y_test, iso_prob))
    results["ece"].append(cal.get_calibration_error(iso_prob, y_test))

    # Isotonic prefit
    clf.fit(X_proper_train, y_proper_train)
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
    clf.fit(X_proper_train, y_proper_train)
    sig_prefit = CalibratedClassifierCV(FrozenEstimator(clf), method="sigmoid")
    sig_prefit.fit(X_cal, y_cal)
    sig_prefit_prob = sig_prefit.predict_proba(X_test)[:, 1]
    results["accuracy"].append(f1_score(y_test, sig_prefit.predict(X_test)))
    results["brier"].append(brier_score_loss(y_test, sig_prefit_prob))
    results["log loss"].append(log_loss(y_test, sig_prefit_prob))
    results["ece"].append(cal.get_calibration_error(sig_prefit_prob, y_test))

    # Prefit
    clf.fit(X_proper_train, y_proper_train)
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

    # IVAP
    va = VennAbersCalibrator(
        estimator=GaussianNB(),
        inductive=True,
        cal_size=0.2,
    )
    va.fit(X_train, y_train)
    va_inductive_prob = va.predict_proba(X_test)[:, 1]
    results["accuracy"].append(f1_score(y_test, va.predict(X_test)[:, 1]))
    results["brier"].append(brier_score_loss(y_test, va_inductive_prob))
    results["log loss"].append(log_loss(y_test, va_inductive_prob))
    results["ece"].append(cal.get_calibration_error(va_inductive_prob, y_test))

    # CVAP
    va = VennAbersCalibrator(
        estimator=GaussianNB(),
        inductive=False,
        n_splits=2,
    )
    va.fit(X_train, y_train)
    va_cv_prob = va.predict_proba(X_test)[:, 1]
    results["accuracy"].append(f1_score(y_test, va.predict(X_test)[:, 1]))
    results["brier"].append(brier_score_loss(y_test, va_cv_prob))
    results["log loss"].append(log_loss(y_test, va_cv_prob))
    results["ece"].append(cal.get_calibration_error(va_cv_prob, y_test))

    print(
        "Summary of the results for the different calibration methods (base model: GaussianNB):"
    )
    df_loss = pd.DataFrame(results, index=methods).T.round(3)
    return df_loss


compare_methods()

# %%
