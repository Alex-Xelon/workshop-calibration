# Calibration exercises

This workshop is organized into three progressive stages, each focusing on a distinct aspect of model calibration and reliable AI predictions. In every stage, you will encounter sections of code with intentional gaps to be completed. Carefully follow the provided instructions and implement each step sequentially to build meaningful and functional solutions.

### **Directory Structure :**
- [simple_class_calibration.py](src/workshop_calibration/stage_1.py) – Binary classification calibration
- [multi_class_calibration.py](src/workshop_calibration/stage_2.py) – Multi-class calibration
- [multi_label_calibration.py](src/workshop_calibration/stage_3.py) – Multi-label calibration

## Stage 1 : Binary Classification Calibration

- **Workshop to complete:** [src/workshop_calibration/stage_1.py](src/workshop_calibration/stage_1.py)
- **Related course:** [Calibration Tutorial - Calibration for Binary Classification](calibration_tutorial.md#calibration-for-binary-classification-simple-calibration)

### **Objective :**
Calibrate a binary classification model so that its predicted probabilities reflect reality.

### **Instructions :**
### Step 0 : Import Libraries
- Execute the import cell to ensure all required libraries are loaded.

### Step 1 : Data Loading
- Load the dataset from the `dataset_stage_1.arff` file.
- Print the first **10** rows of the dataset.

### Step 2 : Class Balancing by Oversampling
- Assign to `majority` the subset of `df` where the `label` column equals the most frequent value in `df.label` (i.e., the majority class). To determine the majority class, you can count the number of instances for each label in the `label` column and select the class with the highest count.
- Assign to `minority` the subset of `df` where the `label` column equals the least frequent value in `df.label` (i.e., the minority class). To determine the minority class, you can count the number of instances for each label in the `label` column and select the class with the lowest count.
- Use the `resample` function to upsample the minority class, so that it has the same number of samples as the majority class. Set `replace=True` to allow for oversampling, `n_samples` to the length of the majority class and `random_state` using `random_state` variable.
- Concatenate the upsampled minority class with the majority class to create a balanced DataFrame using `pd.concat`.
- Shuffle the **entire** balanced DataFrame using `sample()`, reset its index with `reset_index()` removing the old index and set the random state to `random_state` variable.

### Step 3 : Feature and Label Preparation
- Separate the features (`X`) by dropping the `label` column from the DataFrame using `drop` method.
- Extract the labels (`y`) as an **integer** numpy array from the `label` column using `astype` method and `values` attribute.

### Step 4 : Feature Scaling
- Use `StandardScaler` to fit and transform the features (`X`) using `fit_transform` method.

### Step 5 : Data Splitting for Training and Testing
- Use `train_test_split` to split the data into training and test sets, with 5% of the data for training and 95% for testing. Shuffle the data and stratify by the label. Set `random_state` to `random_state` variable.

### Step 6 : Data Splitting for Proper Training and Calibration
- Use `train_test_split` again to split the training set into a proper training set (80%) and a calibration set (20%), shuffling and stratifying by the **training labels**. Set `random_state` to `random_state` variable.

### Step 7 : Model Definition
- Execute the following cell to define a `models` dictionary with models to test.

### Step 8 : Example of model training
- Instantiate the model using the `LogisticRegression` class. Set `max_iter` to 5000 and `random_state` to `random_state` variable.
- Fit the model on the **training data**, predict the probabilities for the positive class and the classes on the **test set** using `fit`, `predict_proba` and `predict` methods.
> Note : The `predict_proba` method returns a numpy array of shape (n_samples, n_classes) where each row corresponds to a sample and each column corresponds to a class. To get the probabilities for the positive class, you need to select the second column of the array.

### Step 9 : Example of model evaluation
- Compute the F1 score (`f1_score`), Brier score (`brier_score_loss`), log loss (`log_loss`), and Expected Calibration Error (`get_calibration_error` method from `cal` object) between the true labels and the model's predictions (predicted classes or probabilities as appropriate).

### Step 10 : Model Evaluation
- Execute the following cell to extend the example of model training and evaluation to all models in the `models` dictionary. For each model, the cell will:
  - Fit the model on the training set (`X_train`, `y_train`).
  - Predict probabilities and classes on the test set (`X_test`).
  - Compute and print the F1 score, Brier score, log loss, and Expected Calibration Error (ECE) for each model.
  - Store the results for each model in the `results` dictionary.

### Step 11 : Example of model calibration : Sigmoid and Isotonic
- For each calibration method ("sigmoid" and "isotonic"), create a `CalibratedClassifierCV` using the already-fitted `model_example` as the estimator, set `method` to the current calibration method, and `cv="prefit"`.
- Fit the calibrated model on the **calibration set**.
- Use the calibrated model to predict probabilities for the positive class (`predict_proba` method) and predicted classes (`predict` method) on the **test set**.
> Note : The `predict_proba` method returns a numpy array of shape (n_samples, n_classes) where each row corresponds to a sample and each column corresponds to a class. To get the probabilities for the positive class, you need to select the second column of the array.
- Compute the F1 score (`f1_score`), Brier score (`brier_score_loss`), log loss (`log_loss`), and Expected Calibration Error (`get_calibration_error` method from `cal` object) using the true test labels and the predictions (predicted classes or probabilities as appropriate).

### Step 12 : Example of model calibration : Venn-Abers (part 1 : calibration)
- Fit the model on the **proper training set**.
- Get predicted probabilities for calibration and test sets (`predict_proba`).
- Calibrate VennAbersCalibrator
- Predict probabilities for the positive class and true labels using `predict_proba` and `predict` methods with predicted probabilities and true labels from the calibration set and predicted probabilities from the test set.
> Note : The `predict` and `predict_proba` methods of `VennAbersCalibrator` return a numpy array of shape (n_samples, n_classes). To get the probabilities for the positive class, you need to select the second column of the array.

### Step 13 : Example of model calibration : Venn-Abers (part 2 : evaluation)
- Compute the F1 score (`f1_score`), Brier score (`brier_score_loss`), log loss (`log_loss`), and Expected Calibration Error (`get_calibration_error` method from `cal` object) using the true test labels and the predictions (predicted classes or probabilities as appropriate).

### Step 14 : Model Calibration
- Execute the following cell to extend the example of model calibration to all models in the `models` dictionary.

### Step 15 : Plot Results
- Execute the following cell to plot the results of the calibration methods.

### Step 16 : Run metrics function
- Execute the following cell to run the metrics function used in the previous step

### Step 17 : Model Comparison
- **Uncalibrated** :
   - Fit a `GaussianNB` classifier on the **training set** and fulfill the run_metrics function with the **test set**
- **Isotonic** :
   - Apply `CalibratedClassifierCV` using the **isotonic** calibration method with **5-fold** cross-validation
   - Fit the calibrated classifier on the **training set** and fulfill the run_metrics function
- **Isotonic prefit** :
   - Instantiate a new `GaussianNB` classifier and fit it on the **proper training set**
   - Use the prefitted classifier as the estimator for a `CalibratedClassifierCV` with the **isotonic** method
   - Fit the calibrated classifier on the **calibration set** and fulfill the run_metrics function
> Note : Use a new `GaussianNB` classifier to avoid kernel memory issues.
- **Sigmoid** :
   - Apply `CalibratedClassifierCV` using the **sigmoid** calibration method with **5-fold** cross-validation
   - Fit the calibrated classifier on the **training set** and fulfill the run_metrics function.
- **Sigmoid prefit** :
   - Instantiate a new `GaussianNB` classifier and fit it on the **proper training set**
   - Use the prefitted classifier as the estimator for a `CalibratedClassifierCV` with the **sigmoid** method
   - Fit the calibrated classifier on the **calibration set** and fulfill the run_metrics function
- **Prefit Venn-Abers** :
   - Instantiate a new `GaussianNB` classifier and fit it on the **proper training set**
   - Predict probabilities for the **calibration** and **test sets** using `predict_proba` method
   - Create a `VennAbersCalibrator` object
   - Predict probabilities for the positive class and true labels using `predict_proba` and `predict` methods with predicted probabilities and true labels from the calibration set and predicted probabilities from the test set.
   - Fulfill the run_metrics function
   > Note : The `predict` and `predict_proba` methods of `VennAbersCalibrator` return a numpy array of shape (n_samples, n_classes). To get the probabilities for the positive class, you need to select the second column of the array.
- **IVAP** :
   - Perform Venn-Abers calibration with a `GaussianNB` classifier using the **inductive** approach by reserving **20%** of the training data for calibration. Set random_state parameter to 0.
   - Fit the calibrated classifier on the training set and fulfill the run_metrics function
- **CVAP** :
   - Perform Venn-Abers calibration with a `GaussianNB` classifier using the **cross-validation** approach with **5 splits**, making sure that the calibration is done in a non-inductive manner. Set `random_state` to 0.
   - Fit the calibrated classifier on the training set and fulfill the run_metrics function.

---

## Stage 2 : Multi-class Calibration

- **Workshop to complete:** [src/workshop_calibration/stage_2.py](src/workshop_calibration/stage_2.py)
- **Related course:** [Calibration Tutorial - Calibration for Multi-class Models](calibration_tutorial.md#calibration-for-multi-class-models)

### **Objective :**
Adapt calibration techniques to a multi-class classification problem.

### **Instructions :**
### Step 0 : Import Libraries
- Execute the import cell to ensure all required libraries are loaded.

### Step 1 : Data Loading
- Load the dataset from the `dataset_stage_2.arff` file using `loadarff` method from `arff` library.
- Print the first 10 rows of the dataset.

### Step 2 : Data preparation
- Separate the features (`X`) by dropping the `Class` column from the DataFrame.
- Extract the labels (`y`) by converting the `Class` column to integers and subtracting **1** using `astype` method and `subtract` method.

### Step 3 : Data Splitting for Training, Calibration and Testing
- Complete the arguments of the `train_test_split` function so that the dataset is split into train and test sets without shuffling, allocating 10% of the data for training and 90% for testing.
- Use the `train_test_split` function again to split the training data into a proper training set (80%) and a calibration set (20%), without shuffling.

### Step 4 : Define the models
- Run the following cell to define the models to test

### Step 5 : Example of calibration : Sigmoid and Isotonic
- Instantiate the model using the `RandomForestClassifier` class.
- Fit the model on the **proper training set**.
- For each calibration method ("**sigmoid**" and "**isotonic**"), create a `CalibratedClassifierCV` using the already-fitted `model_example` as the estimator, set `method` to the current calibration method, and `cv="prefit"`.
- Fit the calibrated model on the **calibration set**.
- Use the calibrated model to predict probabilities for the positive class and predicted classes on the **test set** using `predict_proba` and `predict` methods.

### Step 6 : Example of metrics : Sigmoid and Isotonic
- Compute the F1 score (`f1_score`) with **weighted** average, Brier score (`brier_score_loss`), log loss (`log_loss`), and Expected Calibration Error (`get_calibration_error` method from `cal` object) using the true test labels and the predictions (predicted classes or probabilities as appropriate).

### Step 7 : Example of calibration : Venn-Abers
- Create a `VennAbersCalibrator` instance using a `RandomForestClassifier` as the estimator, making sure to set its `random_state` parameter to the value of the `random_state` variable. Configure the calibrator to use the **cross-validation** (CVAP) approach with **5 splits**, and ensure that the calibration is performed in a **non-inductive** manner.
- Fit the calibrated model on the **training set**.
- Predict probabilities for the positive class and true labels methods with `predict_proba` and `predict` methods on **test set** and one_hot parameter set to False.

### Step 8 : Example of metrics : Venn-Abers
- Compute the F1 score (`f1_score`) with **weighted** average, Brier score (`brier_score_loss`), log loss (`log_loss`), and Expected Calibration Error (`get_calibration_error` method from `cal` object) using the true test labels and the predictions (predicted classes or probabilities as appropriate).

### Step 9 : Define the metrics
- Determine probabilities and classes for the **test set** using the `predict_proba` and `predict` methods of the classifier either with or without Venn-Abers calibration. Set `one_hot` parameter to False if Venn-Abers calibration is used.
- Calculate the F1 score for multi-class classification on the test set using the weighted approach to ensure your method is suitable for imbalanced class distribution.
- Calculate the log loss, Brier score and ECE for the test set using the `log_loss`, `brier_score_loss` and `get_calibration_error` method from `cal` object.

### Step 10 : Calibrate the models
- **Base model** :
   - Fit the model on the **training set** using `fit` and fulfill the metrics fonction to calculate the metrics for the base model.
- **IVAP** :
   - Use `VennAbersCalibrator` to calibrate the model with **inductive** approach and set the size of the calibration to 0.2, then fit the calibrated model on the **training set**
   - Fulfill the metrics fonction with the **test set** with VennAbersCalibrator parameter set to True
- **CVAP** :
   - Use `VennAbersCalibrator` to calibrate the model with **cross-validation** approach and **5 splits**, then fit the calibrated model on the **training set**
   - Fulfill the metrics fonction with the **test set** with VennAbersCalibrator parameter set to False
- **Sigmoid cv** :
   - Use `CalibratedClassifierCV` with method **sigmoid** and **5-fold** cross-validation to calibrate the model, then fit the calibrated model on the **training set**
   - Fulfill the metrics fonction with the test set
- **Isotonic cv** :
   - Use `CalibratedClassifierCV` with method **isotonic** and **5-fold** cross-validation to calibrate the model, then fit the calibrated model on the **training set**
   - Fulfill the metrics fonction with the test set
- **Sigmoid** :
   - Fit the model on the proper training set using `fit`
   - Calibrate the **prefitted** model with method **sigmoid** using `CalibratedClassifierCV` and fit the calibrated model on the **calibration set**
   - Fulfill the metrics fonction with the test set
- **Isotonic** :
   - Calibrate the **prefitted** model with method **isotonic** using `CalibratedClassifierCV` and fit the calibrated model on the **calibration set**
   - Fulfill the metrics fonction with the test set

### Step 7 : Calibration Comparison
- Loop through each classifier `clf_name` in the `clfs` dictionary.
- For each classifier, run the calibration comparison function `run_multiclass_comparison` with the correct arguments.
- Add the results from each classifier to the overall results DataFrames, making sure to ignore the index when combining them.

### Step 8 : Plot Results
- Execute the following cell to plot the results of the calibration methods.

### Step 9 : Run metrics function
- Execute the following cell to run the metrics function used in the previous step

---

## Stage 3 : Multi-label Calibration

- **Workshop to complete:** [src/workshop_calibration/stage_3.py](src/workshop_calibration/stage_3.py)
- **Related course:** [Calibration Tutorial - Calibration for Multi-label Models](calibration_tutorial.md#calibration-for-multi-label-models)

### **Objective :**
Calibrate a multi-label classification model where each sample can belong to multiple classes.

### **Instructions :**
### Step 0 : Import Libraries
- Execute the import cell to ensure all required libraries are loaded.

### Step 1 : Data loading
- Load the dataset from the dataset path for stage 3 using `loadarff` method from `arff` library.
- Print the first 10 rows of the dataset.

### Step 2 : Data Preparation
- Select all numeric columns from the DataFrame using `select_dtypes()` and assign them to `X`.
- Use label_cols to assign the labels `y` and convert values to integers.

### Step 3 : Data Splitting
- Split the data into training and test sets using `train_test_split`, allocating 90% for testing and 10% for training, without shuffling, and set the random_state parameter to `random_state` variable.
- Further split the training set into a **proper training** set and a **calibration** set using `train_test_split` again, with 80% for proper training and 20% for calibration, also without shuffling and with the same random_state parameter.

### Step 4 : Base Model Training and Evaluation
- Instantiate a base classifier using `RandomForestClassifier` with the specified `random_state`.
- Wrap the base classifier with `MultiOutputClassifier` to enable multi-label classification.
- Fit the multi-label classifier on the **training data**.
- Predict the class probabilities and labels for the **test set**.

### Step 5 : Model Calibration
- For each label in y, create a `CalibratedClassifierCV` with **sigmoid** method and **10-fold** cross-validation.
- Fit each calibrated classifier on the corresponding column of the training set and add the calibrated classifier to the `calibrated_clfs` list.
- Predict the probabilities and labels for the test set.
- Convert the lists of predicted probabilities and labels to numpy matrices for further evaluation using `np.vstack` and `np.column_stack`.
> Note: You can use the transpose to flip the matrix and obtain the correct format if needed.

### Step 6 : Compute metrics
- For each label in the **test set**, compute the Brier score for both the uncalibrated and calibrated models and store the results in the `brier_scores_uncalibrated` and `brier_scores_calibrated` lists.
- For each label, compute the accuracy for both the uncalibrated and calibrated models and store the results in the `accuracy_scores_uncalibrated` and `accuracy_scores_calibrated` lists.
- For each label, compute the Expected Calibration Error (ECE) for both the uncalibrated and calibrated models and store the results in the `ece_scores_uncalibrated` and `ece_scores_calibrated` lists.


### Step 7 : Visualisation
- Execute the following cell to plot the results of the calibration methods.

---
