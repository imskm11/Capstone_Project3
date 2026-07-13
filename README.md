# Part 3 – Advanced Modeling: Ensembles, Hyperparameter Tuning, and Machine Learning Pipeline

## Project Overview

This project is the third phase of the Machine Learning assignment and focuses on building advanced classification models, comparing multiple algorithms, tuning model hyperparameters, and creating a complete production-ready machine learning pipeline.

The objective is not only to build accurate predictive models but also to compare their robustness, evaluate their generalization performance using cross-validation, perform hyperparameter tuning, analyze feature importance, study the impact of removing unimportant features, generate learning curves, and serialize the best-performing model for future deployment.

The project uses the cleaned dataset generated in Part 1 and the preprocessed feature matrix created in Part 2.

---

# Objectives

The main objectives of this project are:

- Build an unconstrained Decision Tree classifier.
- Build a controlled Decision Tree classifier to reduce overfitting.
- Compare Gini and Entropy splitting criteria.
- Train a Random Forest classifier.
- Analyze feature importance scores.
- Perform a Feature Ablation Study.
- Train a Gradient Boosting classifier.
- Compare models using 5-Fold Stratified Cross Validation.
- Tune the Random Forest using GridSearchCV.
- Build a complete Scikit-Learn Pipeline.
- Generate a Manual Learning Curve.
- Serialize the best trained model using Joblib.
- Recommend the most suitable model based on experimental results.

---

# Dataset

Dataset Used:

```
cleaned_data.csv
```

This dataset was generated in Part 1 after performing data cleaning and preprocessing.

The dataset contains information related to social media usage, lifestyle, sleep patterns, anxiety levels, screen time, and digital wellbeing.

---

# Feature and Target Variables

### Feature Matrix (X)

The feature matrix contains all independent variables except the target variable.

Examples include:

- daily_screen_hours
- daily_notifications
- avg_sleep_hours
- anxiety_score_0to27
- fomo_1to10
- minutes_to_first_check_after_waking

along with the remaining encoded predictor variables.

---

### Classification Target (y_clf)

The classification target was created in Part 2 by converting the continuous regression target into a binary variable.

The target was generated using the median value of the regression target.

```python
y_clf = (y_reg > y_reg.median()).astype(int)
```

Class 1 represents participants whose target value is greater than the dataset median, whereas Class 0 represents participants whose target value is less than or equal to the median.

---

# Machine Learning Workflow

The complete workflow followed in this project is shown below.

```
Clean Dataset
       │
       ▼
Feature Engineering
       │
       ▼
Train-Test Split
       │
       ▼
Decision Tree
       │
       ▼
Controlled Decision Tree
       │
       ▼
Gini vs Entropy
       │
       ▼
Random Forest
       │
       ▼
Gradient Boosting
       │
       ▼
Feature Importance
       │
       ▼
Feature Ablation
       │
       ▼
Cross Validation
       │
       ▼
GridSearchCV
       │
       ▼
Learning Curve
       │
       ▼
Best Pipeline
       │
       ▼
Model Serialization
```

---

# Python Libraries Used

The following libraries were used throughout this project.

- pandas
- numpy
- matplotlib
- scikit-learn
- joblib

Major Machine Learning classes used include:

- DecisionTreeClassifier
- RandomForestClassifier
- GradientBoostingClassifier
- LogisticRegression
- StandardScaler
- SimpleImputer
- GridSearchCV
- StratifiedKFold
- Pipeline

---

# Project Structure

```
Project Folder
│
├── cleaned_data.csv
├── Advanced_ML.ipynb
├── best_model.pkl
├── README.md

```

---

# Evaluation Metrics Used

The following metrics were used to evaluate different models.

- Training Accuracy
- Testing Accuracy
- ROC-AUC Score
- Cross Validation Mean ROC-AUC
- Cross Validation Standard Deviation
- Feature Importance
- Learning Curve Analysis

These metrics provide a comprehensive understanding of both model accuracy and its ability to generalize to unseen data.

---

# Task 1 – Decision Tree Baseline

The first classification model trained in this project was a default **DecisionTreeClassifier** with no restrictions on tree depth (`max_depth=None`).

The model was trained using the scaled training dataset obtained from Part 2.

## Model Performance

| Metric | Value |
|---------|-------|
| Training Accuracy | **1.0000 (100%)** |
| Testing Accuracy | **0.5643 (56.43%)** |

## Interpretation

The baseline Decision Tree achieved **100% training accuracy**, indicating that it perfectly memorized the training data.

However, the testing accuracy dropped significantly to **56.43%**, which indicates poor generalization to unseen data.

This behavior is a classic example of **overfitting**.

An unconstrained Decision Tree continues splitting until the training samples are perfectly separated, causing the model to learn noise and random variations instead of meaningful patterns.

Decision Trees are often described as **high-variance models** because each split is chosen greedily based on the current data, and once a split is made it is never reconsidered. Small changes in the training data can therefore produce a very different tree.

---

# Task 2 – Controlled Decision Tree

To reduce overfitting, a second Decision Tree model was trained using controlled hyperparameters.

```python
DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=20,
    random_state=42
)
```

The objective was to reduce model complexity while improving its ability to generalize.

## Role of max_depth

The **max_depth** parameter limits how deep the tree is allowed to grow.

Restricting the tree depth prevents the model from creating excessively specific decision rules that fit only the training data.

A smaller tree generally has:

- Lower variance
- Better generalization
- Slightly higher bias

This represents the classic bias-variance trade-off.

---

## Role of min_samples_split

The **min_samples_split** parameter specifies the minimum number of samples required before a node can be split.

Setting this parameter to **20** prevents the model from creating branches based on very small subsets of the data.

This reduces sensitivity to noisy observations and helps improve model stability.

---

# Performance Comparison

| Model | Training Accuracy | Testing Accuracy |
|--------|------------------:|-----------------:|
| Default Decision Tree | **1.0000** | **0.5643** |
| Controlled Decision Tree | **0.6466** | **0.6214** |

---

## Comparison

The controlled Decision Tree significantly reduced the gap between training and testing performance.

Although the training accuracy decreased from **100%** to **64.66%**, the testing accuracy increased from **56.43%** to **62.14%**.

This indicates that the controlled model generalized better to unseen data and was less affected by overfitting.

The reduction in the train-test performance gap demonstrates that limiting tree depth and preventing unnecessary splits improved the model's robustness.

Overall, the controlled Decision Tree is a more reliable classifier than the unconstrained Decision Tree for this dataset.


---

# Task 3 – Gini vs Entropy Comparison

To investigate the effect of different splitting criteria, two Decision Tree classifiers were trained using the same hyperparameters (`max_depth=5`) while changing only the impurity measure.

The following criteria were compared:

- **Gini Impurity**
- **Entropy (Information Gain)**

Both models were trained using identical training data and evaluated on the same testing dataset.

---

## Gini Impurity Formula

The Gini Impurity measures how often a randomly selected sample would be incorrectly classified if labels were assigned according to the class distribution in a node.

\[
\text{Gini} = 1 - \sum p_i^2
\]

where:

- \(p_i\) = probability of class *i*.

A Gini value of **0** means that all samples in the node belong to a single class, making it a perfectly pure node.

---

## Entropy Formula

Entropy measures the amount of uncertainty or disorder in a node.

\[
\text{Entropy} = - \sum p_i \log_2(p_i)
\]

where:

- \(p_i\) = probability of class *i*.

Entropy is minimized when all observations belong to one class and maximized when classes are evenly distributed.

---

## Model Performance

| Criterion | Test Accuracy |
|-----------|--------------:|
| Gini | **0.6214** |
| Entropy | **0.6307** |

---

## Interpretation

Both impurity measures produced similar classification performance.

However, the Decision Tree using the **Entropy** criterion achieved a slightly higher testing accuracy (**63.07%**) than the **Gini** criterion (**62.14%**).

This suggests that Information Gain selected slightly better decision boundaries for this dataset.

The improvement is relatively small, indicating that both criteria are suitable choices. In practice, Gini is often preferred because it is computationally faster, whereas Entropy may produce marginally better splits for certain datasets.

For this project, the **Entropy-based Decision Tree** provided the best performance among the two criteria and was therefore considered the better choice.

---

## Summary

- Gini Impurity measures node impurity using squared class probabilities.
- Entropy measures uncertainty using information theory.
- A Gini value of **0** indicates a perfectly pure node.
- Entropy achieved slightly higher testing accuracy on this dataset.
- The performance difference was small, showing that both impurity measures are effective for binary classification.


---

# Task 4 – Random Forest Classifier

A **Random Forest Classifier** was trained to improve predictive performance by combining the outputs of multiple Decision Trees.

Unlike a single Decision Tree, Random Forest is an ensemble learning algorithm that aggregates predictions from many trees, making the final model more robust and less prone to overfitting.

The model was trained using the following hyperparameters:

```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)
```

---

## Model Performance

| Metric | Value |
|---------|------:|
| Training Accuracy  | **0.8505** |
| Testing Accuracy   | **0.6300** |
| ROC-AUC Score      | **0.6735** |

---

## Interpretation

The Random Forest classifier achieved a **training accuracy of 85.05%** and a **testing accuracy of 63.00%**.

Compared with the Decision Tree models, the Random Forest produced a smaller gap between training and testing performance, indicating improved generalization and reduced overfitting.

The **ROC-AUC score of 0.6735** indicates that the model has a moderate ability to distinguish between the two classes. Although the improvement is not dramatic, the ensemble approach makes the predictions more stable than a single Decision Tree.

---

# Top 5 Most Important Features

Random Forest automatically estimates the contribution of each feature using the average reduction in Gini Impurity across all trees.

The five most important features identified by the model are shown below.

| Rank | Feature | Importance Score |
|------|-------------------------------|----------------:|
| 1 | daily_screen_hours  | **0.172018** |
| 2 | daily_notifications | **0.101370** |
| 3 | avg_sleep_hours     | **0.065828** |
| 4 | anxiety_score_0to27 | **0.058841** |
| 5 | minutes_to_first_check_after_waking | **0.049309** |

---

## Feature Importance Interpretation

The most influential feature was **daily_screen_hours**, with an importance score of **0.172018**.

This indicates that the model relied heavily on the number of hours spent on screens when making classification decisions.

The second most important feature was **daily_notifications**, suggesting that users receiving a large number of notifications showed different behavioural patterns compared to users with fewer notifications.

Other influential variables such as **average sleep hours**, **anxiety score**, and **time taken to check the phone after waking up** also contributed meaningfully to the model's predictions.

Overall, these results suggest that both digital behaviour and mental wellbeing indicators are important predictors in this dataset.

---

# How Random Forest Computes Feature Importance

Random Forest computes feature importance by measuring the **average reduction in Gini Impurity** produced by each feature across all trees in the forest.

Whenever a feature is selected to split a node, the impurity of that node decreases.

The total reduction in impurity contributed by a feature is accumulated across all trees and then normalized to produce its final importance score.

Therefore, features that consistently produce large reductions in impurity receive higher importance values.

---

# Difference Between Feature Importance and Linear Regression Coefficients

Feature importance values in Random Forest should **not** be interpreted in the same way as coefficients in Linear Regression.

### Linear Regression Coefficients

- Show both the magnitude and direction of the relationship.
- Positive coefficients increase the predicted value.
- Negative coefficients decrease the predicted value.

### Random Forest Feature Importance

- Indicates only how useful a feature is for making predictions.
- Does **not** indicate whether the relationship is positive or negative.
- Higher importance means the feature contributes more to reducing classification error.

Therefore, feature importance measures predictive usefulness rather than the direction of influence.

---

# Bagging Concept in Random Forest

Random Forest is based on the **Bagging (Bootstrap Aggregating)** technique.

The algorithm creates multiple Decision Trees instead of relying on a single tree.

Each tree is trained using a **bootstrap sample**, meaning the training data is sampled **with replacement**.

As a result:

- Every tree receives a slightly different training dataset.
- Some observations appear multiple times.
- Some observations may not appear at all.

In addition, at every split, Random Forest considers only a **random subset of features** instead of evaluating every available feature.

For a classification problem, approximately **√(number of features)** are considered at each split.

This randomness makes individual trees less correlated with one another.

Finally, the predictions from all trees are combined using **majority voting**, producing a more stable and reliable classifier.

The averaging process significantly reduces variance compared with a single deep Decision Tree while maintaining good predictive performance.

---

## Summary

- Random Forest reduced overfitting compared with a single Decision Tree.
- The model achieved **63.00% testing accuracy** and a **ROC-AUC of 0.6735**.
- **daily_screen_hours** was the most influential feature.
- Feature importance measures predictive contribution rather than coefficient direction.
- Bootstrap sampling and random feature selection help Random Forest produce more robust and stable predictions.


---

# Task 4a – Gradient Boosting Classifier

A **Gradient Boosting Classifier** was trained as another ensemble learning technique to compare its performance with Decision Trees and Random Forest.

Unlike Random Forest, which builds many trees independently, Gradient Boosting builds trees sequentially. Each new tree attempts to correct the prediction errors made by the previous trees, gradually improving the overall model performance.

The following hyperparameters were used:

```python
GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
```

---

## Model Performance

| Metric | Value |
|---------|------:|
| Training Accuracy | **0.6911** |
| Testing Accuracy  | **0.6279** |
| ROC-AUC Score     | **0.6843** |

---

## Interpretation

The Gradient Boosting classifier achieved a **training accuracy of 69.11%** and a **testing accuracy of 62.79%**.

The model produced a **ROC-AUC score of 0.6843**, which is slightly higher than the Random Forest model.

Compared with the Decision Tree models, Gradient Boosting showed better generalization because the difference between training and testing accuracy was relatively small.

The sequential learning strategy allows Gradient Boosting to reduce prediction errors gradually, making it an effective ensemble method for this classification task.

---

# Why Gradient Boosting Performs Well

Gradient Boosting builds trees one after another instead of training them independently.

Each tree learns from the mistakes of the previous model by giving more attention to observations that were previously misclassified.

This sequential error-correction process enables Gradient Boosting to achieve strong predictive performance while using relatively shallow trees.

Because of this approach, Gradient Boosting often produces higher predictive accuracy than a single Decision Tree.

---

# Task 4b – Feature Ablation Study

A Feature Ablation Study was conducted to determine whether the least important variables identified by the Random Forest model were actually contributing to prediction performance.

The five features with the lowest importance scores were removed from both the training and testing datasets.

A second Random Forest classifier was then trained using the same hyperparameters as the original model.

The objective was to compare the predictive performance of the full model with that of the reduced model.

---

## Feature Ablation Results

| Model | ROC-AUC |
|--------|--------:|
| Full Random Forest | **0.6735** |
| Reduced Random Forest | **0.6688** |

---

## Interpretation

The Random Forest model trained using all available features achieved a **ROC-AUC score of 0.6735**.

After removing the five least important features, the ROC-AUC decreased slightly to **0.6688**.

This small reduction suggests that the removed variables still contributed some useful predictive information, even though their individual importance scores were low.

Therefore, the removed features cannot be considered completely uninformative.

---

# Production Trade-off

Feature ablation is useful because simpler models require fewer input variables.

A lower-dimensional model generally offers several practical advantages:

- Lower computational cost
- Faster prediction time
- Reduced storage requirements
- Easier feature maintenance
- Improved model interpretability

However, reducing the number of features should only be considered when the decrease in predictive performance is acceptable.

In this project, removing the five least important features resulted in a small decline in ROC-AUC.

Although the performance degradation was minor, the full Random Forest model still produced the best overall performance.

Therefore, retaining all available features is recommended for production deployment because the improvement in predictive accuracy outweighs the relatively small increase in computational complexity.

---

## Summary

- Gradient Boosting achieved a **ROC-AUC score of 0.6843**, outperforming the Random Forest on the test set.
- Feature Ablation showed that removing the five least important features slightly reduced predictive performance.
- The removed features still contained useful information.
- The complete feature set is recommended for deployment because it provides the highest predictive accuracy with only a small increase in model complexity.


---

# Task 5 – Cross-Validated Model Comparison

To obtain a more reliable estimate of model performance, all classification models were evaluated using **5-Fold Stratified Cross Validation**.

Instead of relying on a single train-test split, the training dataset was divided into five equally sized folds.

During each iteration:

- Four folds were used for training.
- One fold was used for validation.
- The process was repeated five times.
- The final performance was obtained by averaging the ROC-AUC scores across all folds.

Stratified sampling ensured that each fold preserved approximately the same class distribution as the original dataset.

---

## Why Cross Validation is Better than a Single Train-Test Split

A single train-test split evaluates the model on only one partition of the data.

As a result, the reported performance may depend heavily on how the dataset was divided.

Cross Validation overcomes this limitation by evaluating the model on multiple train-validation splits.

This provides:

- More reliable performance estimates
- Better measurement of generalization ability
- Lower evaluation variance
- Reduced dependence on a single random split

Therefore, Cross Validation is considered a more robust evaluation technique than a single train-test split.

---

# 5-Fold Cross Validation Results

| Model | Mean ROC-AUC | Standard Deviation |
|--------|-------------:|-------------------:|
| Logistic Regression      | **0.6903** | **0.0152** |
| Gradient Boosting        | **0.6891** | **0.0113** |
| Random Forest            | **0.6783** | **0.0115** |
| Controlled Decision Tree | **0.6718** | **0.0145** |

---

## Interpretation

The **Logistic Regression** model achieved the highest average ROC-AUC (**0.6903**) across the five folds.

Although the improvement over Gradient Boosting (**0.6891**) was small, Logistic Regression consistently produced the strongest average performance.

Gradient Boosting ranked second and showed the **lowest standard deviation (0.0113)** among the ensemble models, indicating stable performance across different folds.

Random Forest achieved competitive results but produced a slightly lower average ROC-AUC than Logistic Regression and Gradient Boosting.

The Controlled Decision Tree achieved the lowest average ROC-AUC among the evaluated models, although it still generalized much better than the unconstrained Decision Tree.

---

# Model Stability

The standard deviation measures how much the model performance changes across different folds.

A lower standard deviation indicates that the model performs consistently on different subsets of the dataset.

From the results:

- **Gradient Boosting** demonstrated the most stable performance.
- **Random Forest** also showed good stability.
- **Logistic Regression** achieved the best overall predictive performance while maintaining acceptable stability.
- **Controlled Decision Tree** showed slightly larger variation than the ensemble models.

Overall, all models produced relatively small standard deviations, indicating that the dataset was reasonably stable and that none of the models suffered from extreme performance fluctuations.

---

## Summary

The Cross Validation experiment demonstrates that evaluating a model using multiple folds provides a more trustworthy estimate of real-world performance than relying on a single train-test split.

Among all evaluated models:

- **Logistic Regression** achieved the highest average ROC-AUC.
- **Gradient Boosting** showed the most consistent performance.
- **Random Forest** remained competitive and offered good generalization.
- **Controlled Decision Tree** improved substantially over the default Decision Tree but remained the weakest performer in terms of average ROC-AUC.

These results provide a reliable basis for selecting the most appropriate model for deployment.


---

# Task 6 – Hyperparameter Tuning using GridSearchCV

To further improve model performance, **GridSearchCV** was applied to the Random Forest classifier.

Instead of manually selecting hyperparameters, Grid Search systematically evaluated multiple parameter combinations and selected the model with the highest cross-validated ROC-AUC score.

To prevent data leakage and ensure reproducibility, the tuning process was performed using a complete Scikit-Learn Pipeline.

---

# Machine Learning Pipeline

The following pipeline was created before running Grid Search:

```python
Pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    RandomForestClassifier(random_state=42)
)
```

The pipeline automatically performs the following operations in sequence:

1. Replace missing values using the median.
2. Standardize the feature values.
3. Train the Random Forest classifier.

Using a Pipeline ensures that every preprocessing step is learned only from the training folds during Cross Validation, preventing information leakage from the validation data.

---

# Hyperparameter Grid

The following parameter combinations were evaluated.

| Hyperparameter | Values |
|---------------|----------------|
| n_estimators | 50, 100, 200 |
| max_depth    | 5, 10, None |
| min_samples_leaf | 1, 5 |

---

# Total Model Configurations

The total number of parameter combinations was:

```
3 × 3 × 2 = 18 combinations
```

Since **5-Fold Cross Validation** was used:

```
18 × 5 = 90 model evaluations
```

Therefore, GridSearchCV trained and evaluated **90 Random Forest models** before selecting the best configuration.

---

# Best Hyperparameters

The best-performing model was obtained using the following hyperparameters:

| Hyperparameter | Best Value |
|---------------|-----------|
| max_depth | **10** |
| min_samples_leaf | **5** |
| n_estimators | **200** |

---

# Best Cross-Validation Score

| Metric | Value |
|--------|------:|
| Best ROC-AUC | **0.6838** |

---

# Interpretation

The Grid Search process identified that increasing the number of trees to **200**, limiting the tree depth to **10**, and requiring at least **5 samples per leaf** produced the highest average ROC-AUC.

Compared with the manually selected Random Forest model, the tuned model is expected to generalize better because the selected hyperparameters reduce unnecessary model complexity while maintaining strong predictive performance.

The use of **min_samples_leaf = 5** also helps reduce overfitting by preventing the creation of extremely small leaf nodes.

---

# Why Hyperparameter Tuning is Important

Machine Learning algorithms contain parameters that control how the model learns from the data.

These parameters are not learned automatically and must be selected by the user.

Choosing inappropriate hyperparameters can lead to:

- Overfitting
- Underfitting
- Poor generalization
- Lower predictive accuracy

GridSearchCV systematically evaluates different combinations to identify the configuration that provides the best validation performance.

---

# Grid Search vs Randomized Search

## Grid Search

Advantages:

- Evaluates every possible parameter combination.
- Guarantees that the best combination within the search grid is found.
- Suitable for small to medium search spaces.

Disadvantages:

- Computationally expensive.
- Becomes slow when many parameters are included.

---

## Randomized Search

Advantages:

- Samples only a subset of parameter combinations.
- Much faster for very large search spaces.
- Suitable when computational resources are limited.

Disadvantages:

- Does not guarantee evaluation of every parameter combination.
- May miss the globally best configuration.

---

# Summary

The GridSearchCV experiment successfully optimized the Random Forest classifier.

The best model used:

- **200 Decision Trees**
- **Maximum Tree Depth = 10**
- **Minimum Samples per Leaf = 5**

A total of **90 model evaluations** were performed using **5-Fold Stratified Cross Validation**.

The tuned model achieved a **Best Cross-Validation ROC-AUC score of 0.6838**, demonstrating that systematic hyperparameter tuning can improve model robustness and generalization performance.


---

# Task 7 – Manual Learning Curve Analysis

To understand how the amount of training data affects model performance, a manual learning curve experiment was conducted using the **best pipeline obtained from GridSearchCV**.

The model was trained using progressively larger portions of the training dataset:

- 20%
- 40%
- 60%
- 80%
- 100%

For each training fraction, the following metrics were computed:

- Training ROC-AUC
- Testing ROC-AUC

The testing dataset remained fixed throughout the experiment so that changes in performance could be attributed only to the amount of training data.

---

# Manual Learning Curve Results

| Training Fraction | Training ROC-AUC | Testing ROC-AUC |
|------------------:|-----------------:|----------------:|
| 20%  | **0.9610**   | **0.6599** |
| 40%  | **0.9314**   | **0.6675** |
| 60%  | **0.9085**   | **0.6697** |
| 80%  | **0.8894**   | **0.6730** |
| 100% | **0.8794**   | **0.6719** |

---

# Interpretation of Results

## Training ROC-AUC Trend

The Training ROC-AUC gradually decreased as the amount of training data increased.

This behaviour is expected for high-capacity models because:

- Small training datasets are easier to memorize.
- As more training data is introduced, the model is forced to learn more general patterns rather than memorizing individual observations.
- Consequently, training performance becomes slightly lower but more realistic.

This reduction in Training ROC-AUC indicates that the model is becoming less prone to overfitting.

---

## Testing ROC-AUC Trend

The Testing ROC-AUC improved steadily as the amount of training data increased.

The performance increased from:

- **0.6599** using only 20% of the training data

to

- approximately **0.6730** when 80% of the training data was used.

At 100% of the training data, the Testing ROC-AUC remained almost unchanged (**0.6719**), indicating that the model had reached a relatively stable level of generalization performance.

---

# Bias-Variance Interpretation

The learning curve demonstrates the classical bias-variance trade-off.

Initially, the model showed signs of high variance because it achieved very high Training ROC-AUC while the Testing ROC-AUC was considerably lower.

As additional training data became available:

- Training performance decreased slightly.
- Testing performance improved.

This behaviour suggests that the additional data helped reduce overfitting and improved the model's ability to generalize.

---

# Is the Model Data-Limited or Capacity-Limited?

One objective of the learning curve is to determine whether collecting more data is likely to improve performance.

In this project:

- Testing ROC-AUC increased consistently until approximately **80%** of the training data.
- Beyond that point, the improvement became very small.
- At **100%** of the available data, the Testing ROC-AUC was almost identical to the value obtained using **80%** of the data.

This indicates that the model performance has largely **plateaued**.

Therefore, the current model appears to be **capacity-limited rather than data-limited**.

Simply collecting more data is unlikely to produce substantial improvements.

Future performance gains are more likely to come from:

- Better feature engineering
- More informative variables
- Alternative machine learning algorithms
- More advanced hyperparameter optimization

---

# Conclusion

The Manual Learning Curve demonstrates that increasing the training dataset improves generalization performance up to a certain point.

The decreasing Training ROC-AUC together with the stabilizing Testing ROC-AUC indicates that the model is learning more robust decision boundaries and is no longer severely overfitting.

Since the Testing ROC-AUC has plateaued near the end of the experiment, further improvements are expected to come from improving the model itself rather than simply increasing the amount of training data.


---

# Task 8 – Model Serialization

After completing model training and hyperparameter tuning, the best-performing machine learning pipeline was saved to disk using the **Joblib** library.

Model serialization allows a trained model to be stored and reused later without repeating the entire training process.

The following code was used to save the model.

```python
joblib.dump(best_pipeline, "best_model.pkl")
```

The saved model was then successfully reloaded.

```python
loaded_model = joblib.load("best_model.pkl")
```

Finally, predictions were generated on two custom test samples to verify that the serialized model was functioning correctly.

The model loaded successfully without errors, confirming that the serialization process was successful.

---

# Summary of All Models

The following table summarizes the performance of all models developed in Parts 2 and 3.

| Model | Test Accuracy | Test ROC-AUC | 5-Fold CV Mean ROC-AUC |
|--------|--------------:|-------------:|-----------------------:|
| Logistic Regression      | -          | **0.9096** | **0.6903** |
| Decision Tree            | **0.5643** | -          | -          |
| Controlled Decision Tree | **0.6214** | -          | **0.6718** |
| Random Forest            | **0.6300** | **0.6735** | **0.6783** |
| Gradient Boosting        | **0.6279** | **0.6843** | **0.6891** |

---

# Overall Model Comparison

### Decision Tree

The unconstrained Decision Tree achieved perfect training accuracy but performed poorly on unseen data due to severe overfitting.

---

### Controlled Decision Tree

Limiting the tree depth and increasing the minimum number of samples required for splitting reduced overfitting and improved testing accuracy.

---

### Random Forest

Random Forest produced more stable predictions than a single Decision Tree.

Feature importance analysis also helped identify the variables that contributed most to prediction performance.

---

### Gradient Boosting

Gradient Boosting produced the highest test-set ROC-AUC among the ensemble models and demonstrated excellent generalization.

---

### Logistic Regression

Logistic Regression achieved the highest average ROC-AUC during 5-Fold Cross Validation.

This indicates that it generalized consistently across different subsets of the dataset.

Although ensemble methods performed competitively, Logistic Regression remained the strongest overall model when evaluated using repeated validation.

---

# Recommended Model

Based on all experiments conducted in Parts 2 and 3, **Logistic Regression** is recommended for deployment.

### Reasons for Recommendation

- Achieved the highest average ROC-AUC during Cross Validation (**0.6903**).
- Demonstrated stable performance across all validation folds.
- Simpler and easier to interpret than ensemble methods.
- Faster to train and make predictions.
- Lower computational cost.
- Less likely to overfit compared with deep Decision Trees.

Although Gradient Boosting produced a slightly higher test-set ROC-AUC than Random Forest, Logistic Regression showed the strongest overall generalization performance across multiple folds and is therefore the preferred production model.

---

# Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-Learn
- Joblib

Machine Learning Algorithms:

- Linear Regression
- Ridge Regression
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting

Techniques Used:

- Standard Scaling
- One-Hot Encoding
- Label Encoding
- Cross Validation
- Hyperparameter Tuning
- GridSearchCV
- Feature Importance Analysis
- Feature Ablation Study
- Manual Learning Curve
- Model Serialization

-

# Conclusion

This project successfully implemented an end-to-end machine learning workflow, beginning with a cleaned dataset and progressing through advanced predictive modelling techniques.

Multiple machine learning algorithms were trained, evaluated, and compared using both hold-out testing and 5-Fold Stratified Cross Validation.

The project also demonstrated practical machine learning techniques including feature importance analysis, feature ablation, hyperparameter tuning using GridSearchCV, learning curve analysis, and model serialization for deployment.

Overall, the experiments showed that controlling model complexity and systematically tuning hyperparameters improved generalization performance.

Among all evaluated models, **Logistic Regression** demonstrated the strongest overall performance in terms of Cross-Validation ROC-AUC, while **Gradient Boosting** achieved the highest ROC-AUC among the ensemble models on the testing dataset.

The project therefore satisfies all required tasks and acceptance criteria while demonstrating a complete and reproducible machine learning pipeline suitable for real-world predictive modelling applications.








