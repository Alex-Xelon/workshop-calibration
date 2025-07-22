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

random_state = 1
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

# Load ARFF file (replace with your actual file path)
data = arff.loadarff("../../data/dataset_stage_1.arff")
df = pd.DataFrame(data[0])

print(df.head(10))

# %%
# Step 2 : Balancing the classes by oversampling
# Assume the label column is named 'label'
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

print(df.head(10))

# %%
# Step 3 : Feature Preparation
X = df.drop(columns=["label"])
y = df["label"].astype(int).values
print(X.head(10))
print(pd.Series(y).head(10))

# %%
# Step 4 : Feature Scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)
print(pd.DataFrame(X).head(10))

# %%
# Step 5 : Data Splitting for Training and Testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    shuffle=True,
    test_size=0.2,
    random_state=random_state,
    stratify=y,
)
print(pd.DataFrame(X_train).head(5))
print(pd.Series(y_train).head(5))
print(pd.DataFrame(X_test).head(5))
print(pd.Series(y_test).head(5))

# %%
# Step 6 : Data Splitting for Proper Training and Calibration
X_proper_train, X_cal, y_proper_train, y_cal = train_test_split(
    X_train,
    y_train,
    shuffle=True,
    test_size=0.2,
    random_state=random_state,
    stratify=y_train,
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
model_example = LogisticRegression(
    max_iter=5000,
    random_state=random_state,
)
model_example.fit(X_proper_train, y_proper_train)
probs_example = model_example.predict_proba(X_test)[:, 1]
preds_example = model_example.predict(X_test)

print(probs_example[:5])
print(preds_example[:5])

# %%
# Step 9 : Example of model evaluation
acc_example = f1_score(y_test, preds_example)
brier_example = brier_score_loss(y_test, probs_example)
logloss_example = log_loss(y_test, probs_example)
ece_example = cal.get_calibration_error(probs_example, y_test)

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
for method in ["sigmoid", "isotonic"]:
    print(f"\nCalibrating LogisticRegression with {method} method")
    # Wrap with CalibratedClassifierCV using the chosen method
    calibrated_model = CalibratedClassifierCV(
        estimator=model_example, method=method, cv="prefit"
    )
    calibrated_model.fit(X_cal, y_cal)
    # Predict probabilities and classes on the test set
    probs_cal = calibrated_model.predict_proba(X_test)[:, 1]
    preds_cal = calibrated_model.predict(X_test)
    # Evaluate
    acc_cal = f1_score(y_test, preds_cal)
    brier_cal = brier_score_loss(y_test, probs_cal)
    logloss_cal = log_loss(y_test, probs_cal)
    ece_cal = cal.get_calibration_error(probs_cal, y_test)
    print(f"Score Accuracy: {acc_cal:.3f}")
    print(f"Brier Score: {brier_cal:.3f}")
    print(f"Log Loss: {logloss_cal:.3f}")
    print(f"Expected Calibration Error: {ece_cal:.3f}")

# %%
# Step 12 : Example of model calibration : Venn-ABERS
print("\nCalibrating LogisticRegression with Venn-ABERS method")
# Fit the model on the proper training set
model_example.fit(X_proper_train, y_proper_train)
# Get predicted probabilities for calibration and test sets
p_cal = model_example.predict_proba(X_cal)
p_test = model_example.predict_proba(X_test)
# Calibrate and predict with VennAbersCalibrator
va = VennAbersCalibrator()
probs_va = va.predict_proba(p_cal=p_cal, y_cal=np.array(y_cal), p_test=p_test)[:, 1]
preds_va = va.predict(p_cal=p_cal, y_cal=np.array(y_cal), p_test=p_test)[:, 1]
# Evaluate
acc_va = f1_score(y_test, preds_va)
brier_va = brier_score_loss(y_test, probs_va)
logloss_va = log_loss(y_test, probs_va)
ece_va = cal.get_calibration_error(probs_va, y_test)
print(f"Score Accuracy: {acc_va:.3f}")
print(f"Brier Score: {brier_va:.3f}")
print(f"Log Loss: {logloss_va:.3f}")
print(f"Expected Calibration Error: {ece_va:.3f}")


# %%
# Step 13 : Model Calibration
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
# Step 14: Plot Results
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
# Step 15 : Run metrics function
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
# Step 16 : Model Comparison
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
    run_metrics(clf, X_test, y_test, results)

    # Isotonic (cv=5)
    iso = CalibratedClassifierCV(GaussianNB(), method="isotonic", cv=5)
    iso.fit(X_train, y_train)
    run_metrics(iso, X_test, y_test, results)

    # Isotonic prefit
    clf = GaussianNB()
    clf.fit(X_proper_train, y_proper_train)
    iso_prefit = CalibratedClassifierCV(clf, method="isotonic", cv="prefit")
    iso_prefit.fit(X_cal, y_cal)
    run_metrics(iso_prefit, X_test, y_test, results)

    # Sigmoid (cv=5)
    sig = CalibratedClassifierCV(GaussianNB(), method="sigmoid", cv=5)
    sig.fit(X_train, y_train)
    run_metrics(sig, X_test, y_test, results)

    # Sigmoid prefit
    clf = GaussianNB()
    clf.fit(X_proper_train, y_proper_train)
    sig_prefit = CalibratedClassifierCV(clf, method="sigmoid", cv="prefit")
    sig_prefit.fit(X_cal, y_cal)
    run_metrics(sig_prefit, X_test, y_test, results)

    # Prefit
    clf = GaussianNB()
    clf.fit(X_proper_train, y_proper_train)
    p_cal = clf.predict_proba(X_cal)
    p_test = clf.predict_proba(X_test)
    va = VennAbersCalibrator()
    va_prefit_prob = va.predict_proba(
        p_cal=p_cal, y_cal=np.array(y_cal), p_test=p_test
    )[:, 1]
    y_pred = va.predict(p_cal=p_cal, y_cal=np.array(y_cal), p_test=p_test)[:, 1]
    run_metrics(va, X_test, y_test, results, va_prefit_prob, y_pred)

    # IVAP
    va = VennAbersCalibrator(
        estimator=GaussianNB(),
        inductive=True,
        cal_size=0.2,
    )
    va.fit(X_train, y_train)
    run_metrics(va, X_test, y_test, results, va=True)

    # CVAP
    va = VennAbersCalibrator(
        estimator=GaussianNB(),
        inductive=False,
        n_splits=2,
    )
    va.fit(X_train, y_train)
    run_metrics(va, X_test, y_test, results, va=True)

    print(
        "Summary of the results for the different calibration methods (base model: GaussianNB):"
    )
    df_loss = pd.DataFrame(results, index=methods).T.round(3)
    return df_loss


compare_methods()

# %%
