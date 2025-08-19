# **Model Calibration: A Key to Trustworthy AI Predictions**

## **What is Model Calibration?**

Model calibration refers to how well a model's predicted probabilities reflect the true likelihood of outcomes. In a **perfectly calibrated model**, whenever it predicts an event with probability *p*, that event actually occurs about *p*% of the time. For example, if a binary classifier tags 100 images as "cat" with 60% confidence each, roughly 60 of those images should truly contain a cat if the model is calibrated. In other words, when a model says it's 70% confident about something, we expect it to be correct 70% of the time in those cases.

This concept is distinct from accuracy. A model could have high accuracy yet be *miscalibrated* – meaning its confidence levels don’t match reality. Consider a fair coin toss scenario: one model always predicts 50% chance of heads, and another model fluctuates between predicting 30% or 70% chance of heads on each toss. Both might achieve ~50% accuracy overall on many tosses, but the first model is perfectly calibrated (when it says 50%, half the tosses are heads) whereas the second is not. For the second model, when it said 30%, the actual frequency of heads was 50%, and similarly its 70% predictions were correct only 50% of the time. This mismatch between predicted probability and observed frequency is exactly what poor calibration looks like.

In essence, calibration measures the **alignment between confidence and correctness**. A well-calibrated classifier’s probability outputs can be interpreted as reliable confidence levels. If we group predictions into bins (say all instances where the model predicted ~80% probability of “cat”), about 80% of those instances should indeed be cats for a calibrated model. This property is crucial whenever decisions or further calculations rely on the probability values rather than just the hard class labels.

## **Why is Model Calibration Important?**

Calibration is an often overlooked but **critical aspect of trustworthy AI**. Even if a model’s accuracy is high, using uncalibrated confidence scores can be misleading or dangerous in real-world applications. Some key benefits of good calibration include:

- **Richer Decision-Making:** Probabilities offer more nuance than binary predictions. With calibrated probabilities, we can set risk-based thresholds (e.g. only act if confidence > 90%) or combine outputs from multiple models in a principled way. This is only reliable if those probabilities truly reflect likelihoods. For instance, in medical diagnosis, a model might be 95% confident a patient is healthy – if it’s well-calibrated, that 95% means something concrete about actual risk, whereas an overconfident but miscalibrated model could dangerously understate the true risk.

- **User Trust and Transparency:** When model outputs are presented to end-users or stakeholders, well-calibrated probabilities improve transparency. Users can trust a prediction marked "90% confidence" if historically about 90% of such predictions have been correct. This is crucial for *Trustworthy AI*, as calibrated models communicate uncertainty honestly rather than being blindly overconfident or underconfident.

- **Fair Model Comparison:** Calibration metrics (like Brier score or ECE, discussed later) enable us to compare models on how **reliable** their probabilities are, not just how many predictions they get right. In scenarios with imbalanced data or when the cost of false confidence is high, a slightly less accurate but well-calibrated model may be preferable to a more accurate but miscalibrated one. Calibration thus provides another lens to evaluate model performance beyond accuracy alone.

- **Improved Downstream Integration:** Many AI systems feed model probabilities into downstream processes – for example, a probability might be used in an automated decision rule, or combined with human judgment. Calibrated probabilities ensure these downstream uses are based on numbers that meaningfully correspond to real-world frequencies, making the entire pipeline more robust and interpretable.

Well-calibrated models are therefore *more trustworthy* and *more useful*. They provide a realistic sense of uncertainty, which is invaluable in critical applications (healthcare, finance, autonomous systems, etc.) where knowing how much to trust a prediction is as important as the prediction itself.

---

## **Calibration for Binary Classification (Simple Calibration)**

Binary classification (two classes, e.g. positive/negative) is the simplest case to illustrate model calibration. Many binary classifiers output a raw score (e.g. a logistic regression’s sigmoid output, or a tree’s vote proportion) which is intended to represent a probability. However, different algorithms have different natural calibration characteristics. For example, logistic regression trained with log-loss tends to produce well-calibrated probabilities by default, whereas naive Bayes often pushes predictions to 0 or 1 (overconfident) due to its assumptions. Random forests, on the other hand, tend to be under-confident (predicting probabilities closer to the middle like 0.2–0.8) because they average over many trees.

To check calibration of a binary classifier, the most common tool is the **calibration plot** (also known as a reliability diagram). This plot bins predicted probabilities and compares them with actual outcome frequencies. The ideal calibration plot is a diagonal 45° line (identity function) – meaning predicted probability equals true frequency in each bin. Deviations from the diagonal indicate miscalibration (over- or under-confidence in certain ranges). Along with visual inspection, we can quantify calibration. A popular metric is the **Brier score**, essentially the mean squared error between predicted probabilities and actual outcomes (0 for perfect calibration, lower is better). Another is the **Expected Calibration Error (ECE)**, which computes the weighted average gap between prediction and outcome frequency across all bins (often expressed as a percentage difference). These metrics allow us to say, for instance, “Model A’s probabilities are more aligned with reality than Model B’s.”

### **How do we fix calibration?**

The process of *calibration* typically means taking an already-trained classifier and learning a **calibration function** to adjust its output probabilities to better match observed outcomes. This is usually done on a separate **calibration dataset** (hold-out data not used in training, or via cross-validation), to avoid simply overfitting the training data.

Three widely used calibration techniques for binary classifiers are *Platt scaling*, *isotonic regression* and *Venn-Abers Predictors* :

- **Platt Scaling (Sigmoid Calibration):**
  This method fits a logistic regression (sigmoid) to the model’s output scores vs the true labels. Essentially, it learns a S-shaped curve to map raw outputs to calibrated probabilities. Platt scaling is simple and works well even with smaller datasets, but it does assume the calibration function is sigmoidal (which may not hold if the true calibration curve is not S-shaped). It was originally popularized for calibrating SVM outputs, and in practice it learns two parameters ($\alpha$, $\beta$) such that:

$$\text{calibrated\_p} = \sigma(\alpha \cdot \text{score} + \beta)$$

- **Isotonic Regression:**
  This is a non-parametric calibration method that fits a free-form, non-decreasing function to the reliability diagram. It makes no strong assumption on shape; it can adjust to any monotonic curve mapping predicted scores to true probabilities. The flexibility comes with a risk: isotonic regression can **overfit** if there are not enough calibration points, since it can essentially draw a piecewise-constant line through the data. It's usually recommended when you have a fairly large calibration set (often > 1000 samples). Isotonic will never predict probabilities that decrease as the score increases (it preserves the rank ordering of predictions, just adjusts their values).

**With or Without prefit**

- **Without prefit** (e.g. **cv=5**): the model is retrained inside `CalibratedClassifierCV`, and calibration is performed on validation folds from the training set.

    ```python
    calibrator = CalibratedClassifierCV(
        RandomForestClassifier(),
        method="isotonic",
        cv=5
    )
    calibrator.fit(X_train, y_train)
    ```

- **With prefit**: you first train the model on a small proper training set, and then calibrate it on a larger calibration set. This is more flexible and avoids retraining.

    ```python
    model = RandomForestClassifier().fit(X_train_proper, y_train_proper)
    calibrator = CalibratedClassifierCV(
        model,
        method="isotonic",
        cv="prefit"
    )
    calibrator.fit(X_cal, y_cal)
    ```

- **Venn-Abers Predictors**

Venn-Abers methods are rooted in conformal prediction and provide probability intervals with theoretical validity guarantees. There are three main variants:

1. **Prefit Venn-Abers**: You train a model yourself and give it as input, along with its predicted probabilities on calibration and test data. It adjusts those predictions using the calibration set.

    ```python
    va = VennAbersCalibrator()
    probs = va.predict_proba(p_cal=p_cal, y_cal=y_cal, p_test=p_test)
    ```

2. **IVAP (Inductive Venn-Abers Predictor)**: You give a base estimator, and the `VennAbersCalibrator` internally splits the training set between training and calibration.

    ```python
    va = VennAbersCalibrator(
        estimator=RandomForestClassifier(),
        inductive=True,
        cal_size=0.2
    )
    va.fit(X_train, y_train)
    ```

3. **CVAP (Cross-validated Venn-Abers Predictor)**: Instead of a single calibration split, the training set is split via cross-validation. This provides more stable calibration.

    ```python
    va = VennAbersCalibrator(
        estimator=RandomForestClassifier(),
        inductive=False,
        n_splits=5
    )
    va.fit(X_train, y_train)
    ```

> These methods outperform Platt or Isotonic calibration when data is small or distributions are complex.

While there are many calibration methods available including **Platt scaling** (sigmoid), **isotonic regression**, and more specialized techniques like **temperature scaling** (especially for deep neural networks), *beta calibration*, and ensemble-based approaches, **Platt and isotonic regression remain the most widely used and practical in real-world applications**.
Some modern libraries also provide advanced options such as **Venn-Abers methods**, which offer theoretical guarantees and probability intervals. However, these are more complex and typically reserved for research or specialized scenarios.

### **Example – Calibrating a Classifier**

Suppose we have a dataset and we train a random forest classifier for a binary task. We find its probability outputs are not well aligned with actual outcomes (perhaps the calibration plot shows an S-curve). We can calibrate it as follows:

1. **Hold out a calibration set:** Split the training data into two parts – a smaller portion to train the model, and a larger portion (e.g. 80%) to calibrate. The model is first trained on the proper training set.
2. **Fit the calibrator:** Using the model's predictions on the calibration set, fit a calibration model (sigmoid or isotonic). Scikit-learn provides a convenient wrapper for this.
3. **Use calibrated model for predictions:** Now when we predict on new test data, we'll pass the original model's output through the calibration mapping to get adjusted probabilities.

**In code, using scikit-learn's `CalibratedClassifierCV` class:**

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

# Assume X_proper_train, y_proper_train is our proper training data, and we set aside X_cal, y_cal for calibration.
base_model = RandomForestClassifier(n_estimators=100).fit(X_proper_train, y_proper_train)  # train the model
calibrator = CalibratedClassifierCV(base_model, method="isotonic", cv="prefit")
calibrator.fit(X_cal, y_cal)  # fit calibration on held-out data

# Now use the calibrated model to predict probabilities on test set:
probs = calibrator.predict_proba(X_test)
```

In the code above, `cv="prefit"` tells sklearn that we have already trained the base classifier (`base_model`) and we are providing an explicit calibration set (`X_cal`, `y_cal`). Alternatively, if we set `cv=5` (for example), `CalibratedClassifierCV` would internally perform 5 cross-validation folds on `X_train`, `y_train` to derive calibration data – but using a dedicated set (if available) often yields better calibration.

**Visualizing the effect:** The impact of calibration can be seen in calibration plots. Below is an example calibration curves and metrics comparing a well-calibrated model vs. a miscalibrated one, before and after applying calibration.

#### **Calibration curves (reliability diagrams)**

Calibration curves show how well a classifier's predicted probabilities match the actual observed frequencies of the positive class.
On the plot, the x-axis represents the predicted probability and the y-axis shows the true fraction of positives among samples assigned that probability.
A perfectly calibrated model will have its curve follow the diagonal line (where predicted probability equals observed frequency). Deviations from the diagonal indicate over- or under-confidence in the model's probability estimates.

> The **dashed gray line** represents **perfect calibration** (ideal alignment of predicted probability with actual frequency). The **blue curve** shows a **Logistic Regression model**, which stays very close to the diagonal – it is almost perfectly calibrated. The **orange curve** is an **uncalibrated Gaussian Naive Bayes** model; notice how it deviates significantly, indicating the model's predicted probabilities do not match true frequencies (e.g. when NB predicts around 0.1, the actual fraction of positives is much higher, showing under-confidence, whereas at higher predicted probabilities it becomes over-confident). After applying calibration, the Naive Bayes model's curves (**green curve for isotonic calibration** and **red curve for sigmoid/Platt calibration**) move much closer to the diagonal. Calibration has adjusted the probabilities to be more realistic – low predictions are bumped up and high predictions toned down in this case – resulting in probabilities we can trust more.

![Exemple de courbes de calibration](/assets/calibration_curves.png)


#### **Calibration metrics**

In practice, after calibration we also expect numerical improvements in calibration metrics: for example, the Naive Bayes model’s Brier score would drop significantly once calibrated, and its ECE would also decrease, reflecting that its probabilities are now more in line with actual outcomes. Log-loss (cross-entropy) is also a proper scoring rule that is sensitive to calibration – a lower log-loss usually indicates better calibration for a given accuracy.

- **Brier Score:**
  Measures the mean squared difference between predicted probabilities and actual outcomes (0 or 1). Lower is better; a perfectly calibrated model achieves the lowest possible Brier score for its accuracy.
  $$ \text{Brier Score} = \frac{1}{N} \sum_{i=1}^N (p_i - y_i)^2 $$
  where $p_i$ is the predicted probability and $y_i$ is the true label.

- **Log Loss (Cross-Entropy):**
  Penalizes confident but wrong predictions more heavily. It is sensitive to both calibration and discrimination. Lower log loss means better calibrated and more confident predictions.
  $$ \text{Log Loss} = -\frac{1}{N} \sum_{i=1}^N [y_i \log(p_i) + (1-y_i)\log(1-p_i)] $$

- **Expected Calibration Error (ECE):**
  Measures the average absolute difference between predicted probability and observed frequency, typically by binning predictions. Lower ECE means predicted probabilities are closer to actual frequencies.
  $$ \text{ECE} = \frac{1}{N} \sum_{i=1}^N \left| p_i - y_i \right| $$

> **Low Brier score** and **low log loss** indicate both good calibration and good accuracy.
> **Low ECE** specifically indicates that the predicted probabilities are honest and reliable.

![Exemple de tableaux de métriques de calibration](/assets/calibration_metrics_table.png)

Keep in mind calibration usually does not improve accuracy and can sometimes slightly decrease it. Calibration is about fixing probabilities, often trading a tiny bit of accuracy for much better honest confidence estimates. This trade-off is usually worthwhile when probability estimates will be used in decisions.

> **Note:** Recalibrating an already calibrated model (for example, applying Platt or isotonic calibration multiple times, or calibrating a model that was already well-calibrated) can actually degrade performance. Each calibration step fits a mapping to the predicted probabilities, and if the model is already well-calibrated, further calibration can introduce unnecessary distortion, leading to worse calibration metrics and less reliable probabilities. It's generally best to calibrate only once, and only if your model's probabilities are misaligned with observed outcomes. Always check calibration plots and metrics before and after calibration to ensure it is actually improving your model.

---

## **Calibration for Multi-class Models**

Calibrating **multi-class classifiers** (with more than two classes) is a bit more involved, but the core idea remains: predicted class probabilities should align with actual frequencies. For a multi-class model, you want that for all samples predicted with (for example) 70% probability for class A, about 70% actually belong to class A (and similarly for other classes). However, unlike the binary case, the model outputs a probability distribution across multiple classes that always sums to 1. Calibration must ensure each class probability is meaningful.

A straightforward approach to multi-class calibration is to **reduce it to multiple binary calibrations** (one per class). In fact, scikit-learn’s calibration for multi-class works this way: if you call `CalibratedClassifierCV` on a multi-class classifier, it will internally transform the problem into **one-vs-rest** format. It learns calibration maps for each class’s “against the rest” probability estimates. Essentially, for each class it considers all instances either as “this class” or “not this class” and fits a sigmoid or isotonic calibrator to those probabilities. During prediction, it ensures the adjusted probabilities are renormalized to sum to 1. This one-vs-rest scheme is a simple extension that often works well in practice.

Another popular technique, especially for deep neural networks, is **Temperature Scaling**. This is a special case of Platt scaling that uses a single scalar parameter to **soften or sharpen the probability distribution** for all classes simultaneously (by dividing the logits by a learned temperature > 0) without changing the predicted class ranks. Temperature scaling is convenient for multi-class deep models because it requires tuning just one parameter on a validation set (finding the temperature that minimizes, say, negative log-likelihood). It can significantly reduce overconfidence in modern neural nets. However, temperature scaling assumes a simple scaling of all scores is sufficient; if different classes or score ranges need different adjustments, more flexible methods might be needed. Note that the practical implementation and tuning of temperature scaling is not covered in this workshop.

Beyond these, there are advanced methods specifically for multi-class calibration, such as **Venn-Abers** for multi-class. `Venn–Abers predictors` extend the idea of calibration with guarantees (they output probability intervals) and have formulations for multi-class settings. These methods are more complex and typically used in research; for most practical purposes, one-vs-rest Platt or isotonic calibration via libraries is the go-to solution.

### **Example – Calibrating a Multi-class Model**
Suppose we have a classifier that predicts three classes (A, B, C). We can use `CalibratedClassifierCV` similarly as before:

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

# y has 3 classes here
model = RandomForestClassifier().fit(X_train, y_train)
calibrated_model = CalibratedClassifierCV(model, method="sigmoid", cv="prefit")
# calibrate on calibration set
calibrated_model.fit(X_cal, y_cal)
probs = calibrated_model.predict_proba(X_test)
```

Here probs will be an N×3 array of calibrated probabilities for the 3 classes. Under the hood, this effectively calibrated the model’s output for each class against all others. We could also have used cv=5 without a separate set, which would calibrate using cross-validation splits of the training data.

To evaluate calibration in multi-class problems, we extend the same metrics used for binary classification.

**Brier score**

For $N$ observations and $C$ possible classes, the **multi-class Brier score** generalizes the binary case by summing the squared differences between the predicted probability and the true label indicator for each class, then averaging over all samples and classes:

$$
\text{Brier Score} = \frac{1}{N} \sum_{i=1}^N \sum_{c=1}^C \left( \hat{p}_{ic} - y_{ic} \right)^2
$$

- $y_{ic}$ is 1 if observation $i$ belongs to class $c$, and 0 otherwise.
- $\hat{p}_{ic}$ is the predicted probability that observation $i$ belongs to class $c$.

**Log loss**

The **multi-class log loss** is the average log loss across all classes, weighted by the class frequency:

$$
\text{Log Loss} = -\frac{1}{N} \sum_{i=1}^N \sum_{c=1}^C y_{ic} \log(\hat{p}_{ic})
$$

**Expected calibration error**

Expected calibration error can be extended to multi-class by averaging the per-class calibration errors (possibly weighted by class frequency).

$$
\text{ECE} = \frac{1}{N} \sum_{i=1}^N \sum_{c=1}^C \left| \hat{p}_{ic} - y_{ic} \right|
$$

It's worth noting that if a model was already well-calibrated in multi-class (e.g. a multinomial logistic regression often is), adding an unnecessary calibration step won't help and could even slightly degrade performance. Calibration should be used when you have evidence (via plots or metrics) of miscalibration. Many complex models (like modern deep networks) do exhibit miscalibration, often being overconfident in the predicted class, which is why calibrating them (even with a simple temperature scaling) is becoming a standard practice in classification tasks.

---

## **Calibration for Multi-label Models**

**Multi-label classification** involves predicting multiple labels for each instance (e.g. an image might have *both* "cat" and "dog" labels if it contains both animals). This is effectively a collection of several binary classification problems – one for each label – since each label can be present or absent independently. Therefore, calibration in multi-label settings typically boils down to ensuring each label's probability is calibrated **independently**.

For example, if you have 5 possible labels for an image, the model might output 5 probabilities (one per label). We want that for the subset of images where the model says label 1 has a 90% probability, label 1 is indeed present ~90% of the time; similarly for label 2, label 3, etc. Each label has its own calibration curve. Importantly, multi-label probabilities do not need to sum to 1 (each is an independent probability of that label being true), so we can calibrate each output without worrying about a sum constraint.

The simplest approach is to **calibrate each label’s classifier separately** using binary calibration methods. If using a scikit-learn `MultiOutputClassifier` (which wraps a binary classifier for each output), you could calibrate each sub-classifier. In practice, one can do something like:

```python
from sklearn.multioutput import MultiOutputClassifier
from sklearn.calibration import CalibratedClassifierCV

base_clf = RandomForestClassifier()  # base model for each label
multi_clf = MultiOutputClassifier(base_clf).fit(X_train, Y_train)  # Y_train has multiple label columns

# Calibrate each label-specific classifier
calibrated_clfs = []
for label_idx in range(Y_train.shape[1]):
    clf_label = CalibratedClassifierCV(base_clf, method="sigmoid", cv=5)
    clf_label.fit(X_calib, Y_calib.iloc[:, label_idx])  # fit on that label's column
    calibrated_clfs.append(clf_label)

# Now calibrated_clfs[i] can predict probabilities for label i
calibrated_probs = [clf.predict_proba(X_test)[:,1] for clf in calibrated_clfs]
```

In this snippet, we trained a single `RandomForest` for all labels (`MultiOutputClassifier` does this under the hood by cloning the model for each label). We then calibrated each label's model using its portion of the calibration set (`Y_calib.iloc[:, label_idx]` gives the true outcomes for label i on the calibration data). We used sigmoid (Platt) calibration above; we could also try isotonic if we have enough data for each label. After this, we have a list of calibrated classifiers, one per label, and we gather their predicted probabilities on the test set. We would interpret and use each label's probability separately.

The result of multi-label calibration is that **for each label** you can trust the predicted probability. For instance, suppose an image tagging model says there's a `0.30` (30%) probability of the "dog" tag and `0.80` (80%) for the "cat" tag on a particular image. If calibrated, over many images, about 30% of those tagged with 30% "dog" confidence should actually have dogs, and about 80% of those tagged with 80% "cat" confidence should have cats. This reliability is crucial if, say, you only want to take action on labels that exceed a certain confidence (you wouldn't want to miss too many real positives or include too many false alarms due to miscalibrated scores).

One challenge in multi-label settings is that some labels may be rare, making calibration harder. In such cases, you might need more data or use simpler calibration (sigmoid might be preferable over isotonic for rare labels, which have sparse data). Also, while calibrating each label independently is standard, it ignores any correlation between labels. In advanced research, there are methods to calibrate the joint predictions in multi-label, but these are beyond the scope of most practical applications.

---

## **Conclusion**

Model calibration is a critical step towards **trustworthy AI** because it aligns a model's confidence with reality. In this course, we discussed what calibration means and why it matters, then explored how to calibrate models for binary, multi-class, and multi-label classification. By using methods like Platt scaling and isotonic regression and Venn-Abers predictors, we can adjust a model's outputs to make better decisions and provide reliable confidence estimates. The take-home message is: a well-calibrated model not only *predicts well*, but also *knows when it's likely to be right or wrong* – and that is invaluable for building AI systems that people can trust.
