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
- Assign to `majority` the subset of `df` where the `label` column equals the most frequent value in `df.label` (i.e., the majority class).
- Assign to `minority` the subset of `df` where the `label` column equals the least frequent value in `df.label` (i.e., the minority class).
- Use the `resample` function to upsample the minority class so that it has the same number of samples as the majority class.
- Concatenate the upsampled minority class with the majority class to create a balanced DataFrame.
- Shuffle the **entire** balanced DataFrame using `sample()` and reset its index with `reset_index()`, removing the old index.

### Step 3 : Feature and Label Preparation
- Separate the features (`X`) by dropping the `label` column from the DataFrame.
- Extract the labels (`y`) as an integer numpy array from the `label` column.

### Step 4 : Feature Scaling
- Use `StandardScaler` to fit and transform the features (`X`).

### Step 5 : Data Splitting for Training and Testing
- Use `train_test_split` to split the data into training and test sets, with 5% of the data for training and 95% for testing. Shuffle the data and stratify by the label.

### Step 6 : Data Splitting for Proper Training and Calibration
- Use `train_test_split` again to split the training set into a proper training set (80%) and a calibration set (20%), shuffling and stratifying by the training labels.

### Step 7 : Model Definition
- Execute the following cell to define a `models` dictionary with models to test.

### Step 8 : Example of model training
- Fit the model on the training data, predict the probabilities for the positive class and the classes on the test set.

### Step 9 : Example of model evaluation
- Compute the F1 score (`f1_score`), Brier score (`brier_score_loss`), log loss (`log_loss`), and Expected Calibration Error (`cal.get_calibration_error`) between the true labels and the model's predictions (predicted classes or probabilities as appropriate).

### Step 10 : Model Evaluation
- Execute the following cell to extend the example of model training and evaluation to all models in the `models` dictionary. For each model, the cell will:
  - Fit the model on the training set (`X_train`, `y_train`).
  - Predict probabilities and classes on the test set (`X_test`).
  - Compute and print the F1 score, Brier score, log loss, and Expected Calibration Error (ECE) for each model.
  - Store the results for each model in the `results` dictionary.

### Step 11 : Example of model calibration : Sigmoid and Isotonic
- For each calibration method ("sigmoid" and "isotonic"), create a `CalibratedClassifierCV` using the already-fitted `model_example` as the estimator, set `method` to the current calibration method, and `cv="prefit"`.
- Fit the calibrated model on the calibration set.
- Use the calibrated model to predict probabilities for the positive class (`predict_proba`) and predicted classes (`predict`) on the test set.
- Compute the F1 score (`f1_score`), Brier score (`brier_score_loss`), log loss (`log_loss`), and Expected Calibration Error (`cal.get_calibration_error`) using the true test labels and the predictions (predicted classes or probabilities as appropriate).

### Step 12 : Example of model calibration : Venn-Abers
- Fit the model on the proper training set.
- Get predicted probabilities for calibration and test sets (`predict_proba`).
- Calibrate and predict with VennAbersCalibrator using the predicted probabilities and true labels from the calibration set and predicted probabilities from the test set.
- Compute the F1 score (`f1_score`), Brier score (`brier_score_loss`), log loss (`log_loss`), and Expected Calibration Error (`cal.get_calibration_error`) using the true test labels and the predictions (predicted classes or probabilities as appropriate).

### Step 13 : Model Calibration
- Execute the following cell to extend the example of model calibration to all models in the `models` dictionary.

### Step 14 : Plot Results
- Execute the following cell to plot the results of the calibration methods.

### Step 15 : Run metrics function
- Execute the following cell to run the metrics function used in the previous step

### Step 16 : Model Comparison
- **Uncalibrated** :
   - Fit a `GaussianNB` classifier on the training set and fulfill the run_metrics function
- **Isotonic** :
   - Apply `CalibratedClassifierCV` using the isotonic calibration method with 5-fold cross-validation
   - Fit the calibrated classifier on the training set and fulfill the run_metrics function
- **Isotonic prefit** :
   - Fit the classifier on the proper training set
   - Use the prefitted classifier as the estimator for a `CalibratedClassifierCV` with the isotonic method
   - Fit the calibrated classifier on the calibration set and fulfill the run_metrics function
- **Sigmoid** :
   - Apply `CalibratedClassifierCV` using the sigmoid calibration method with 5-fold cross-validation
   - Fit the calibrated classifier on the training set and fulfill the run_metrics function.
- **Sigmoid prefit** :
   - Fit a `GaussianNB` classifier on the proper training set
   - Use the prefitted classifier as the estimator for a `CalibratedClassifierCV` with the sigmoid method
   - Fit the calibrated classifier on the calibration set and fulfill the run_metrics function
- **Prefit Venn-Abers** :
   - Fit a `GaussianNB` classifier on the proper training set
   - Predict probabilities for the calibration and test sets
   - Fulfill the run_metrics function
- **IVAP** :
   - Perform Venn-Abers calibration using the inductive approach by reserving 20% of the training data for calibration.
   - Fit the calibrated classifier on the training set and fulfill the run_metrics function
- **CVAP** :
   - Perform Venn-Abers calibration using the cross-validation approach with 2 splits, making sure that the calibration is done in a non-inductive manner.
   - Fit the calibrated classifier on the training set and fulfill the run_metrics function.

---

## Stage 2 : [Multi-class Calibration](src/calibration_test/multi_class_calibration.py)

### **Objective :**
Adapt calibration techniques to a multi-class classification problem.

### **Instructions :**
### Step 0 : Import Libraries
- Execute the import cell to ensure all required libraries are loaded.

### Step 1 : Data Loading
- Load the dataset from the `dataset_stage_2.arff` file.
- Print the first 10 rows of the dataset.

### Step 2 : Data preparation
- Separate the features (`X`) by dropping the `Class` column from the DataFrame.
- Extract the labels (`y`) by converting the `Class` column to integers and subtracting 1.

### Step 3 : Data Splitting for Training, Calibration and Testing
- Complete the arguments of the `train_test_split` function so that the dataset is split into train and test sets without shuffling, allocating 10% of the data for training and 90% for testing.
- Use the `train_test_split` function again to split the training data into a proper training set (80%) and a calibration set (20%), without shuffling.

### Step 4 : Define the models
- Run the following cell to define the models to test

### Step 5 : Define the metrics
- Determine probabilities and classes for the test set using the `predict_proba` and `predict` methods of the classifier either with or without Venn-Abers calibration.
- Calculate the F1 score for multi-class classification on the test set using the weighted approach to ensure your method is suitable for imbalanced class distribution.
- Calculate the log loss, Brier score and ECE for the test set using the `log_loss`, `brier_score_loss` and `cal.get_calibration_error` functions.

### Step 6 : Calibrate the models
- **Base model** :
   - Fit the model on the training set using `fit` and fulfill the metrics fonction to calculate the metrics for the base model.
- **Sigmoid** :
   - Fit the model on the proper training set using `fit`
   - Calibrate the model with method "sigmoid" and cv="prefit" on the calibration set using `CalibratedClassifierCV` and fit the calibrated model on the calibration set
   - Fulfill the metrics fonction with the test set
- **Isotonic** :
   - Calibrate the model with method "isotonic" and cv="prefit" on the calibration set using `CalibratedClassifierCV` and fit the calibrated model on the calibration set
   - Fulfill the metrics fonction with the test set
- **Sigmoid cv** :
   - Use `CalibratedClassifierCV` with method "sigmoid" and 5-fold cross-validation to calibrate the model, then fit the calibrated model on the calibration set
   - Fulfill the metrics fonction with the test set
- **Isotonic cv** :
   - Use `CalibratedClassifierCV` with method "isotonic" and 5-fold cross-validation to calibrate the model, then fit the calibrated model on the calibration set
   - Fulfill the metrics fonction with the test set
- **IVAP** :
   - Use `VennAbersCalibrator` to calibrate the model with inductive false and 5 splits, then fit the calibrated model on the training set
   - Fulfill the metrics fonction with the test set with VennAbersCalibrator parameter set to True
- **CVAP** :
   - Use `VennAbersCalibrator` to calibrate the model with inductive false and 5 splits, then fit the calibrated model on the training set
   - Fulfill the metrics fonction with the test set with VennAbersCalibrator parameter set to False

### Step 7 : Calibration Comparison
- Loop through each classifier `clf_name` in the `clfs` dictionary.
- For each classifier, run the calibration comparison function `run_multiclass_comparison` with the correct arguments.
- Add the results from each classifier to the overall results DataFrames, making sure to ignore the index when combining them.

### Step 8 : Plot Results
- Execute the following cell to plot the results of the calibration methods.

### Step 9 : Run metrics function
- Execute the following cell to run the metrics function used in the previous step

---

## Stage 3 : [Multi-label Calibration](src/calibration_test/multi_label_calibration.py)

### **Objective :**
Calibrate a multi-label classification model where each sample can belong to multiple classes.

### **Instructions :**
### Step 0 : Import Libraries
- Execute the import cell to ensure all required libraries are loaded.

### Step 1 : Data loading
- Load the dataset from the dataset path for stage 3.
- Print the first 10 rows of the dataset.

### Step 2 : Data Preparation
- Select all numeric columns from the DataFrame using `select_dtypes()`, retrieve them with the `columns` attribute and assign them to `X`.
- Use label_cols to assign the labels `y` as an integer numpy array.
- Print the first 5 rows of `X` and `y` to verify the data preparation.

### Step 3 : Data Splitting
- Split the data into training and test sets using `train_test_split`, allocating 90% for testing and 10% for training, without shuffling, and set a random seed for reproducibility.
- Further split the training set into a "proper training" set and a "calibration" set using `train_test_split` again, with 80% for proper training and 20% for calibration, also without shuffling and with the same random seed.

### Step 4 : Base Model Training and Evaluation
- Instantiate a base classifier using `RandomForestClassifier` with the specified `random_state`.
- Wrap the base classifier with `MultiOutputClassifier` to enable multi-label classification.
- Fit the multi-label classifier on the training data using `fit` method.
- Predict the class probabilities and labels for the test set using `predict_proba` and `predict` methods.

### Step 5 : Model Calibration
- For each label in y, create a `CalibratedClassifierCV` with sigmoid method and cross-validation with 10 folds
- Fit each calibrated classifier on the corresponding column of y_train and add the calibrated classifier to the `calibrated_clfs` list.
- Predict the probabilities and labels for the test set using `predict_proba` and `predict` methods.
- Convert the lists of predicted probabilities and labels to numpy matrices for further evaluation using `np.vstack` and `np.column_stack`.

### Step 6 : Compute metrics
- Run the following cell to compute the metrics for the uncalibrated and calibrated models.

### Step 7 : Visualisation
- Execute the following cell to plot the results of the calibration methods.

---
