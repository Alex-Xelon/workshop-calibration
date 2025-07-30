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
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# %%
# Step 1 : Data loading

random_state = 6

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

data = ___  # TODO
df = pd.DataFrame(data[0])

print(___)  # TODO

# %%
# Step 2 : Data Preparation
X = ___  # TODO

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
y = ___  # TODO

print(X.head())
print(y.head())

# %%
# Step 3 : Data Splitting
X_train, X_test, y_train, y_test = ___  # TODO

X_proper_train, X_cal, y_proper_train, y_cal = ___  # TODO

print(f"\n X_train :\n{X_train.head()}")
print(f"shape : {X_train.shape}")
print(f"\n y_train :\n{y_train.head()}")
print(f"shape : {y_train.shape}")
print(f"\n X_test :\n{X_test.head()}")
print(f"shape : {X_test.shape}")
print(f"\n y_test :\n{y_test.head()}")
print(f"shape : {y_test.shape}")
print(f"\n X_proper_train :\n{X_proper_train.head()}")
print(f"shape : {X_proper_train.shape}")
print(f"\n y_proper_train :\n{y_proper_train.head()}")
print(f"shape : {y_proper_train.shape}")
print(f"\n X_cal :\n{X_cal.head()}")
print(f"shape : {X_cal.shape}")
print(f"\n y_cal :\n{y_cal.head()}")
print(f"shape : {y_cal.shape}")

# %%
# Step 4 : Multi-output classifier non calibrated
base_clf = ___  # TODO
multi_clf = ___  # TODO

pred_probs_uncalibrated = ___  # TODO
pred_y_uncalibrated = ___  # TODO

print(f"\n pred_probs_uncalibrated :\n{pred_probs_uncalibrated}")
print(f"\n pred_y_uncalibrated :\n{pred_y_uncalibrated}")


# %%
# Step 5 : Model calibration
calibrated_clfs = []
pred_probs_calibrated = []
pred_y_calibrated = []
for ___ in range(___):
    ___  # TODO

# Conversion to numpy matrix
pred_probs_uncalibrated_matrix = ___  # TODO
pred_probs_calibrated_matrix = ___  # TODO

pred_y_uncalibrated_matrix = ___  # TODO
pred_y_calibrated_matrix = ___  # TODO

print(f"pred_probs_uncalibrated_matrix:\n{pred_probs_uncalibrated_matrix}")
print(f"shape: {pd.DataFrame(pred_probs_uncalibrated_matrix).shape}")
print(f"pred_probs_calibrated_matrix:\n{pred_probs_calibrated_matrix}")
print(f"shape: {pd.DataFrame(pred_probs_calibrated_matrix).shape}")
print(f"pred_y_uncalibrated_matrix:\n{pred_y_uncalibrated_matrix}")
print(f"shape: {pd.DataFrame(pred_y_uncalibrated_matrix).shape}")
print(f"pred_y_calibrated_matrix:\n{pred_y_calibrated_matrix}")
print(f"shape: {pd.DataFrame(pred_y_calibrated_matrix).shape}")

# %%
# Step 6 : Compute metrics

brier_scores_uncalibrated = [___]  # TODO
brier_scores_calibrated = [___]  # TODO

accuracy_scores_uncalibrated = [___]  # TODO
accuracy_scores_calibrated = [___]  # TODO

ece_scores_uncalibrated = [___]  # TODO
ece_scores_calibrated = [___]  # TODO

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

# %%
# Step 7 : Visualisation

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
plt.title("Expected Calibration Error (ECE) per Label - Before and After Calibration")
plt.tight_layout()
plt.show()
