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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, log_loss, f1_score
from venn_abers import VennAbersCalibrator
import xgboost as xgb
import calibration as cal
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from scipy.io import arff
import warnings

warnings.filterwarnings("ignore")

# %%
# Step 1: Data Loading
print("Loading the dataset")

random_state = 28
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

data = arff.loadarff("../../data/___")  # TODO
df = pd.DataFrame(data[0])

print(df.head(___))  # TODO

# %%
# Step 2 : Balancing the classes by oversampling
majority = df[df.___ == df.label.value_counts().idxmax()]  # TODO
minority = df[df.label == df.label.___.___()]  # TODO
minority_upsampled = ___(  # TODO
    ___,  # TODO
    replace=True,
    n_samples=len(___),  # TODO
    random_state=random_state,
)

df = pd.concat([___, ___])  # TODO
df = df.___(frac=___, random_state=random_state).reset_index(drop=___)  # TODO

print(df.head(10))

# %%
# Step 3 : Feature Preparation
X = df.drop(columns=["___"])  # TODO
y = df["___"].astype(___).values  # TODO
print(X.head(10))
print(pd.DataFrame(y, columns=["label"]).head(10))

# %%
# Step 4 : Feature Scaling
scaler = ___()  # TODO
X = scaler.fit_transform(___)  # TODO
print(pd.DataFrame(X).head(10))

# %%
# Step 5 : Data Splitting for Training and Testing
X_train, X_test, y_train, y_test = train_test_split(
    ___,  # TODO
    ___,  # TODO
    shuffle=___,  # TODO
    test_size=___,  # TODO
    random_state=random_state,
    stratify=___,  # TODO
)
print(pd.DataFrame(X_train).head(5))
print(pd.Series(y_train).head(5))
print(pd.DataFrame(X_test).head(5))
print(pd.Series(y_test).head(5))

# %%
# Step 6 : Data Splitting for Proper Training and Calibration
X_proper_train, X_cal, y_proper_train, y_cal = train_test_split(
    ___,  # TODO
    ___,  # TODO
    shuffle=___,  # TODO
    test_size=___,  # TODO
    random_state=random_state,
    stratify=___,  # TODO
)
print(pd.DataFrame(X_proper_train).head(5))
print(pd.Series(y_proper_train).head(5))
print(pd.DataFrame(X_cal).head(5))
print(pd.Series(y_cal).head(5))

# %%
# Step 7 : Model Definition
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
# Step 8 : Example of model training
print("Example of model training and evaluation")
model_example = ___(  # TODO
    ___,  # TODO
    ___,  # TODO
)
model_example.fit(___, ___)  # TODO
probs_example = model_example.predict_proba(___)[:, ___]  # TODO
preds_example = model_example.predict(___)  # TODO

print(probs_example[:5])
print(preds_example[:5])

# %%
# Step 9 : Example of model evaluation
acc_example = f1_score(___, ___)  # TODO
brier_example = brier_score_loss(___, ___)  # TODO
logloss_example = log_loss(___, ___)  # TODO
ece_example = cal.get_calibration_error(___, ___)  # TODO

print(f"Score Accuracy: {acc_example:.3f}")
print(f"Brier Score: {brier_example:.3f}")
print(f"Log Loss: {logloss_example:.3f}")
print(f"Expected Calibration Error: {ece_example:.3f}")

# %%
# Step 10 : Model Evaluation
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
# Step 11 : Example of model calibration : Sigmoid and Isotonic
model_example.fit(X_proper_train, y_proper_train)

for method in ["___", "___"]:  # TODO
    print(f"\nCalibrating LogisticRegression with {method} method")
    # Wrap with CalibratedClassifierCV using the chosen method
    calibrated_model = CalibratedClassifierCV(
        estimator=___,  # TODO
        method=___,  # TODO
        cv="prefit",
    )
    calibrated_model.fit(___, ___)  # TODO

    # Predict probabilities and classes on the test set
    probs_cal = calibrated_model.predict_proba(___)[:, 1]  # TODO
    preds_cal = calibrated_model.predict(___)  # TODO
    print(f"Probs calibration: {probs_cal[:5]}")
    print(f"Preds calibration: {preds_cal[:5]}")

    # Evaluate
    acc_cal = f1_score(___, ___)  # TODO
    brier_cal = brier_score_loss(___, ___)  # TODO
    logloss_cal = log_loss(___, ___)  # TODO
    ece_cal = cal.get_calibration_error(___, ___)  # TODO
    print(f"Score Accuracy: {acc_cal:.3f}")
    print(f"Brier Score: {brier_cal:.3f}")
    print(f"Log Loss: {logloss_cal:.3f}")
    print(f"Expected Calibration Error: {ece_cal:.3f}")

# %%
# Step 12 : Example of model calibration : Venn-ABERS (part 1 : calibration)
print("\nCalibrating LogisticRegression with Venn-ABERS method")
# Fit the model on the proper training set
model_example.fit(___, ___)  # TODO
# Get predicted probabilities for calibration and test sets
p_cal = model_example.predict_proba(___)  # TODO
p_test = model_example.predict_proba(___)  # TODO
# Calibrate and predict with VennAbersCalibrator
va = VennAbersCalibrator()
probs_va = va.predict_proba(p_cal=___, y_cal=np.array(___), p_test=___)[:, 1]  # TODO
preds_va = va.predict(p_cal=___, y_cal=np.array(___), p_test=___)[:, 1]  # TODO
print(f"Probs calibration: {probs_va[:5]}")
print(f"Preds calibration: {preds_va[:5]}")

# %%
# Step 13 : Example of model calibration : Venn-ABERS (part 2 : evaluation)
acc_va = f1_score(___, ___)  # TODO
brier_va = brier_score_loss(___, ___)  # TODO
logloss_va = log_loss(___, ___)  # TODO
ece_va = cal.get_calibration_error(___, ___)  # TODO
print(f"Score Accuracy: {acc_va:.3f}")
print(f"Brier Score: {brier_va:.3f}")
print(f"Log Loss: {logloss_va:.3f}")
print(f"Expected Calibration Error: {ece_va:.3f}")

# %%
# Step 14 : Model Calibration
def calibration():
    for name in models.keys():
        for method in ["sigmoid", "isotonic"]:
            print(f"\n Calibration de {name} avec méthode {method}")
            base_model = models[name].fit(X_proper_train, y_proper_train)
            calibrated_model = CalibratedClassifierCV(
                estimator=base_model, method=method, cv="prefit"
            )
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
            name_method = f"{name} + {method}"
            results[name_method] = {
                "probs": probs,
                "accuracy": acc,
                "brier": brier,
                "logloss": logloss,
                "ece": ece,
            }
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
        name_method = f"{name} + Venn-ABERS"
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
# Step 15: Plot Results
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
# Step 16 : Run metrics function
def run_metrics(clf, X_test, y_test, results, probs=None, preds=None, va=False):
    if probs is None and preds is None:
        probs = clf.predict_proba(X_test)[:, 1]
        preds = clf.predict(X_test)
    if va:
        probs = clf.predict_proba(X_test)[:, 1]
        preds = clf.predict(X_test)[:, 1]
    acc = f1_score(y_test, preds)
    brier = brier_score_loss(y_test, probs)
    logloss = log_loss(y_test, probs)
    ece = cal.get_calibration_error(probs, y_test)
    results["accuracy"].append(acc)
    results["brier"].append(brier)
    results["log loss"].append(logloss)
    results["ece"].append(ece)
    return


# %%
# Step 17 : Model Comparison
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
    clf.fit(___, ___)  # TODO
    run_metrics(___, ___, ___, ___)  # TODO

    # Isotonic (cv=5)
    iso = CalibratedClassifierCV(GaussianNB(), method="___", cv=___)  # TODO
    iso.fit(___, ___)  # TODO
    run_metrics(___, ___, ___, ___)  # TODO

    # Isotonic prefit
    clf = ___  # TODO
    clf.fit(___, ___)  # TODO
    iso_prefit = CalibratedClassifierCV(___, method="___", cv="___")  # TODO
    iso_prefit.fit(___, ___)  # TODO
    run_metrics(___, ___, ___, ___)  # TODO

    # Sigmoid (cv=5)
    sig = CalibratedClassifierCV(GaussianNB(), method="___", cv=___)  # TODO
    sig.fit(___, ___)  # TODO
    run_metrics(___, ___, ___, ___)  # TODO

    # Sigmoid prefit
    clf = GaussianNB()
    clf.fit(___, ___)  # TODO
    sig_prefit = CalibratedClassifierCV(___, method="___", cv="___")  # TODO
    sig_prefit.fit(___, ___)  # TODO
    run_metrics(___, ___, ___, ___)  # TODO

    # Prefit
    clf = ___  # TODO
    clf.fit(___, ___)  # TODO
    p_cal = clf.predict_proba(___)  # TODO
    p_test = clf.predict_proba(___)  # TODO
    va = VennAbersCalibrator()
    va_prefit_prob = va.predict_proba(
        p_cal=p_cal, y_cal=np.array(y_cal), p_test=p_test
    )[:, 1]
    y_pred = va.predict(p_cal=p_cal, y_cal=np.array(y_cal), p_test=p_test)[:, 1]
    run_metrics(___, ___, ___, ___, ___, ___)  # TODO

    # IVAP
    va = VennAbersCalibrator(
        estimator=GaussianNB(),
        inductive=___,  # TODO
        cal_size=___,  # TODO
        random_state=0,
    )
    va.fit(___, ___)  # TODO
    run_metrics(___, ___, ___, ___, ___, ___)  # TODO

    # CVAP
    va = VennAbersCalibrator(
        estimator=GaussianNB(),
        inductive=___,  # TODO
        n_splits=___,  # TODO
        random_state=0,
    )
    va.fit(___, ___)  # TODO
    run_metrics(___, ___, ___, ___, ___, ___)  # TODO

    print(
        "Summary of the results for the different calibration methods (base model: GaussianNB):"
    )
    df_loss = pd.DataFrame(results, index=methods).T.round(3)
    return df_loss


compare_methods()

# %%
