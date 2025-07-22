# Calibration exercises

This workshop is organized into three progressive stages, each focusing on a distinct aspect of model calibration and reliable AI predictions. In every stage, you will encounter sections of code with intentional gaps to be completed. Carefully follow the provided instructions and implement each step sequentially to build meaningful and functional solutions.

### **Directory Structure :**
- [simple_class_calibration.py](src/calibration_test/simple_class_calibration.py) – Binary classification calibration
- [multi_class_calibration.py](src/calibration_test/multi_class_calibration.py) – Multi-class calibration
- [multi_label_calibration.py](src/calibration_test/multi_label_calibration.py) – Multi-label calibration

## Stage 1 : [Binary Calibration (Simple Calibration)](src/calibration_test/simple_class_calibration.py)

### **Objective :**
Calibrate a binary classification model so that its predicted probabilities reflect reality.

### **Instructions :**
### Step 0 : Import Libraries
- Execute the import cell to ensure all required libraries are loaded.

### Step 1 : Data Loading
- Load the dataset from the `dataset_simpleclass.arff` file.
- Print the first 10 rows of the dataset.

### Step 2 : Class Balancing by Oversampling
- Identify the majority and minority classes in your DataFrame `df` using the `label` column.
- Use the `resample` function to upsample the minority class so that it has the same number of samples as the majority class.
- Concatenate the upsampled minority class with the majority class to create a balanced DataFrame.
- Shuffle the entire balanced DataFrame using `sample()` and reset its index with `reset_index()`, removing the old index.

### Step 3 : Feature and Label Preparation
- Separate the features (`X`) by dropping the `label` column from the DataFrame.
- Extract the labels (`y`) as an integer numpy array from the `label` column.

### Step 4 : Feature Scaling
- Use `StandardScaler` to fit and transform the features (`X`).

### Step 5 : Data Splitting for Training and Testing
- Use `train_test_split` to split the data into training and test sets, with 5% of the data for training and 95% for testing. Shuffle the data and stratify by the label.

### Step 6 : Data Splitting for Proper Training and Calibration
- Use `train_test_split` again to split the training set into a proper training set (80%) and a calibration set (20%), shuffling and stratifying by the label.

### Step 7 : Model Training and Evaluation (Uncalibrated)
- For each model in the `models` dictionary :
    - Fit the model on the training data, predict the probabilities for the positive class on the test set.
    - Compute the F1 score (`f1_score`), Brier score (`brier_score_loss`), log loss (`log_loss`), and Expected Calibration Error (`cal.get_calibration_error`) between the true labels and the model's predictions (predicted classes or probabilities as appropriate).

### Step 8 : Explicit Calibration with Platt (Sigmoid) and Isotonic
- For each model in the `models` dictionary, iterate through the calibration methods "sigmoid" and "isotonic".
- For each combination, fit the base model using the proper training set and use the trained model as the estimator for a `CalibratedClassifierCV` with the chosen calibration method.
- Fit this calibrated model on the calibration set, predict probabilities for the positive class and classes on the test set.
- Calculate the F1 score (`f1_score`), Brier score (`brier_score_loss`), log loss (`log_loss`), and Expected Calibration Error (`cal.get_calibration_error`) by comparing the true labels with the model's predictions.

### Step 9 : Explicit Calibration with Venn-Abers
- For each model in the `models` dictionary, fit the base model on the proper training set, use the trained model to predict probabilities on both the calibration set and the test set.
- Calibrate `VennAbersCalibrator` using the predicted probabilities and true labels from the calibration set and predicted probabilities from the test set.
- After calibration, use the Venn-Abers calibrator to predict calibrated probabilities and classes for the test set.
- Calculate the F1 score (`f1_score`), Brier score (`brier_score_loss`), log loss (`log_loss`), and Expected Calibration Error (`cal.get_calibration_error`) by comparing the true test labels with the model's predictions.

### Step 10 : Uncalibrated Model
- Fit a `GaussianNB` classifier on the training data.
- Predict probabilities and classes on the test set.
- Compute and store the F1 score, Brier score, log loss, and ECE for the predictions.

### Step 11 : IVAP Calibration
   - Perform Venn-Abers calibration using the inductive approach by reserving 20% of the training data for calibration.
   - Predict probabilities and classes on the test set.
   - Compute and store the F1 score, Brier score, log loss, and ECE.

### Step 12 : CVAP Calibration
   - Perform Venn-Abers calibration using the cross-validation approach with 2 splits, making sure that the calibration is done in a non-inductive (i.e., not using a held-out calibration set) manner.
   - Predict probabilities and classes on the test set.
   - Compute and store the F1 score, Brier score, log loss, and ECE.

### Step 13 : Prefit Venn-Abers Calibration
   - Randomly split the training data so that 20% is used as the proper training set and the remaining 80% is reserved for calibration.
   - Fit the classifier on the proper training set and predict probabilities for the calibration and test sets.
   - Predict probabilities and classes on the test set.
   - Compute and store the F1 score, Brier score, log loss, and ECE.

### Step 14 : Isotonic Calibration (cv=5)
   - Apply `CalibratedClassifierCV` using the isotonic calibration method with 5-fold cross-validation.
   - Fit the calibrated classifier on the calibration set and predict probabilities and classes on the test set.
   - Compute and store the F1 score, Brier score, log loss, and ECE.

### Step 15 : Isotonic Prefit Calibration
   - Fit the classifier on the proper training set
   - Use the fitted classifier as the estimator for a `CalibratedClassifierCV` with the isotonic method
   - Fit the calibrated classifier on the calibration set and predict probabilities for the calibration and test sets.
   - Predict probabilities and classes on the test set.
   - Compute and store the F1 score, Brier score, log loss, and ECE.

### Step 16 : Sigmoid Calibration (cv=5)
   - Apply `CalibratedClassifierCV` using the sigmoid calibration method with 5-fold cross-validation.
   - Fit the calibrated model on the training set and predict probabilities and classes on the test set.
   - Compute and store the F1 score, Brier score, log loss, and ECE.

### Step 17 : Sigmoid Prefit Calibration
   - Fit the classifier on the proper training set
   - Use the fitted classifier as the estimator for a `CalibratedClassifierCV` with the sigmoid method
   - Fit the calibrated model on the calibration set and predict probabilities and classes on the test set.
   - Compute and store the F1 score, Brier score, log loss, and ECE.

---

## Stage 2 : [Multi-class Calibration](src/calibration_test/multi_class_calibration.py)

### **Objective :**
Adapt calibration techniques to a multi-class classification problem.

### **Instructions :**
### Step 1 : Data Splitting for Training and Calibration
- Complete the arguments of the `train_test_split` function so that the binary dataset is split into train and test sets without shuffling, allocating 10% of the data for training and 90% for testing.
- Use the `train_test_split` function again to split the data for calibration, this time setting the test_size to 0.8 (for the calibration set) and train_size to 0.2 (for the proper training set), and do not shuffle the data.

### Step 2 : Base Model Training and Evaluation
- For base model, fit the model on the training set and predict the probabilities on the test set and the classes on the test set.
- Store the f1 score with average "weighted" in acc_list, log loss in log_loss_list, brier score in brier_loss_list and ece in ece_list for the base model.

### Step 3 : Sigmoid Calibration
- For sigmoid calibration, prefit the model on the proper training set, calibrate the model with method "sigmoid" and cv="prefit" on the calibration set and predict the probabilities on the test set and the classes on the test set.
- Store the f1 score with average "weighted" in acc_list, log loss in log_loss_list, brier score in brier_loss_list and ece in ece_list for the sigmoid calibrated model.

### Step 4 : Isotonic Calibration
- For isotonic calibration, prefit the model on the proper training set, calibrate the model with method "isotonic" and cv="prefit" on the calibration set and predict the probabilities on the test set and the classes on the test set.
- Store the f1 score with average "weighted" in acc_list, log loss in log_loss_list, brier score in brier_loss_list and ece in ece_list for the isotonic calibrated model.

### Step 5 : Isotonic Calibration with Cross-Validation
- For isotonic calibration with cross-validation, prefit the model on the proper training set, calibrate the model with method "isotonic" and cv=5 on the calibration set and predict the probabilities on the test set and the classes on the test set.
- Store the f1 score with average "weighted" in acc_list, log loss in log_loss_list, brier score in brier_loss_list and ece in ece_list for the isotonic calibrated model with cross-validation.

### Step 6 : Sigmoid Calibration with Cross-Validation
- For sigmoid calibration with cross-validation, prefit the model on the proper training set, calibrate the model with method "sigmoid" and cv=5 on the calibration set and predict the probabilities on the test set and the classes on the test set.
- Store the f1 score with average "weighted" in acc_list, log loss in log_loss_list, brier score in brier_loss_list and ece in ece_list for the sigmoid calibrated model with cross-validation.

### Step 7 : IVAP Calibration
- For IVAP calibration, prefit the model on the proper training set, calibrate the model on the calibration set with inductive true and size of calibration set of 0.2 and predict the probabilities on the test set and the classes on the test set.
- Store the f1 score with average "weighted" in acc_list, log loss in log_loss_list, brier score in brier_loss_list and ece in ece_list for the Venn-Abers calibrated model.

### Step 8 : CVAP Calibration with Cross-Validation
- For CVAP calibration with cross-validation, prefit the model on the proper training set, calibrate the model on the calibration set with inductive false and 5 splits, then predict the probabilities on the test set and the classes on the test set.
- Store the f1 score with average "weighted" in acc_list, log loss in log_loss_list, brier score in brier_loss_list and ece in ece_list for the Venn-Abers calibrated model with cross-validation.

### Step 9 : Calibration Comparison
- For each classifier name in the dictionary `clfs`, iterate over the classifiers and run the calibration comparison for each classifier in your list.
- Concatenate `results_brier`, `results_log_loss`, `results_acc`, and `results_ece` with the new `scratch_brier`, `scratch_log_loss`, `scratch_acc`, and `scratch_ece`, ignoring the index.

### Step 10 : Calibration Comparison and Results Aggregation
- For each classifier name in the dictionary `clfs`, iterate over the classifiers and run the calibration comparison for each classifier in your list.
- Concatenate `results_brier`, `results_log_loss`, `results_acc`, and `results_ece` with the new `scratch_brier`, `scratch_log_loss`, `scratch_acc`, and `scratch_ece`, ignoring the index.

---

## Stage 3 : [Multi-label Calibration](src/calibration_test/multi_label_calibration.py)

### **Objective :**
Calibrate a multi-label classification model where each sample can belong to multiple classes.

### **Instructions :**
### Step 1 : Data Preparation and Splitting
- Split the data into training and test sets using `train_test_split`, allocating 90% for testing and 10% for training, without shuffling, and set a random seed for reproducibility.
- Further split the training set into a "proper training" set and a "calibration" set using `train_test_split` again, with 80% for proper training and 20% for calibration, also without shuffling and with the same random seed.
- Print the heads of all resulting splits (X_train, y_train, X_test, y_test, X_proper_train, y_proper_train, X_cal, y_cal) to verify the splits.

### Step 2 : Base Model Training and Evaluation
- Instantiate a base classifier using `RandomForestClassifier` with the specified `random_state`.
- Wrap the base classifier with `MultiOutputClassifier` to enable multi-label classification.
- Fit the multi-label classifier on the training data.
- Predict the class probabilities and labels for the test set.

### Step 3 : Calibration by Output
- For each label in y, create a CalibratedClassifierCV with method="sigmoid" and cv=10 and fit each calibrated classifier on the corresponding column of y_train
- Predict the probabilities and labels for the test set.
- Convert the lists of predicted probabilities and labels to numpy matrices for further evaluation.

---
