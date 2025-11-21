🍄 Mushroom Classifier: Edible or Poisonous?

This project implements and evaluates several machine learning models to classify mushrooms as edible (e) or poisonous (p) using the UCI Mushroom Dataset.

The data's exceptional quality, particularly the high predictive power of the odor feature (zero class overlap for many odor types), allowed the models—especially the tree-based ones—to achieve near-perfect classification accuracy.

🎯 Project Goal

To build a robust and highly accurate model for identifying poisonous mushrooms, minimizing the risk of false negatives (poisonous mushroom classified as edible), which is critical in this domain.

📊 Dataset Overview

The dataset contains 8124 samples, with 22 categorical features describing various physical characteristics of mushrooms.

    Class Distribution: The data is well-balanced:

        Edible (e): 4208 (51.8%)

        Poisonous (p): 3916 (48.2%)

    Data Integrity: No missing values or duplicates were found.

    Feature Removal: The 'veil-type' feature was removed as it contained only a single unique value and thus had no predictive power.

Key Finding: Odor Dominance

Exploratory Data Analysis revealed that the odor feature is an almost perfect predictor for classification.

    Odor categories unique to Edible: a (almond), l (anise), n (none)

    Odor categories unique to Poisonous: c (creosote), f (foul), m (musty), p (pungent), s (spicy), y (fishy)

This zero-overlap for most odor categories explains the exceptional performance observed across all machine learning models.

🛠️ Methodology

1. Data Preprocessing

    Target Encoding: The target variable 'class' was mapped to numerical values (e: 0, p: 1).

    Feature Encoding: One-Hot Encoding was chosen over Label Encoding. This was crucial because Label Encoding would impose a false ordinal relationship between the categorical features, while One-Hot Encoding correctly treats each category as independent, preserving the true predictive power of features like odor.

2. Modeling & Evaluation

The encoded data was split into training and test sets (80/20 split, stratified on the target class). Four classification algorithms were trained and evaluated:

    Logistic Regression (LGR)

    Decision Tree Classifier (DTC)

    Random Forest Classifier (RFC)

    Gradient Boosting Classifier (GBC)

Model	Accuracy	Precision	Recall	F1 Score
Decision Tree	1.000	1.000	1.000	1.000
Random Forest	1.000	1.000	1.000	1.000
Logistic Regression	0.999	1.000	0.999	0.999
Gradient Boosting	0.999	1.000	0.997	0.999

3. Hyperparameter Tuning

Grid Search with Cross-Validation (CV=5) was performed on all models to find the optimal hyperparameters.
Model	Accuracy	Precision	Recall	F1 Score
Decision Tree	1.000	1.000	1.000	1.000
Random Forest	1.000	1.000	1.000	1.000
Logistic Regression	0.999	1.000	0.999	0.999
Gradient Boosting	0.999	1.000	0.997	0.999

🏆 Conclusion

Both the Decision Tree Classifier and Random Forest Classifier achieved perfect accuracy (100%) on the test set, classifying every mushroom correctly.

While a simple Decision Tree is the most straightforward interpretation of the dataset's clear rules, Random Forest is chosen as the final model due to its inherent advantages:

    Robustness: It is less prone to overfitting than a single Decision Tree.

    Stability: Its ensemble nature provides a more stable and generalizable prediction.

    Performance: It matches the perfect performance of the Decision Tree on this test set.

The project demonstrates that in datasets with highly discriminative features, even relatively simple models can achieve virtually perfect results.

🚀 Future Enhancements

    Feature Importance Analysis: Plotting the feature importances for the Random Forest model to quantify the exact contribution of the 'odor_f' (foul odor) and other features.

    Minimal Feature Model: Retraining a model using only the top 3-5 most important features (e.g., Odor, Gill Size, Bruises) to create the simplest model that still achieves high accuracy.
