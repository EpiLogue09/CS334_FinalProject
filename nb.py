import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import numpy as np

# Load the datasets
x_features = pd.read_csv('normalized_x_features.csv')
y_target = pd.read_csv('y_target_binary.csv')

# Define the number of folds for cross-validation
n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

# List of variance smoothing values to try
var_smoothing_list = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]

# Iterate over the variance smoothing values
for var_smoothing in var_smoothing_list:
    accuracy_scores = []
    auroc_scores = []
    f1_scores = []
    
    # Iterate over each fold
    for train_index, test_index in kf.split(x_features):
        X_train, X_test = x_features.iloc[train_index], x_features.iloc[test_index]
        y_train, y_test = y_target.values.ravel()[train_index], y_target.values.ravel()[test_index]

        # Initialize the Gaussian Naive Bayes model with the current var_smoothing
        nb_classifier = GaussianNB(var_smoothing=var_smoothing)

        # Train the model
        nb_classifier.fit(X_train, y_train)

        # Predictions on the test set
        y_pred_nb = nb_classifier.predict(X_test)
        y_pred_proba_nb = nb_classifier.predict_proba(X_test)[:, 1]  # Probabilities for the positive class

        # Calculate and append accuracies and AUROC
        accuracy_scores.append(accuracy_score(y_test, y_pred_nb))
        auroc_scores.append(roc_auc_score(y_test, y_pred_proba_nb))
        f1_scores.append(f1_score(y_test, y_pred_nb))

    # Calculate average scores across all folds
    avg_accuracy = np.mean(accuracy_scores)
    avg_auroc = np.mean(auroc_scores)
    avg_f1_score = np.mean(f1_scores)

    # Print the results
    print(f"Naive Bayes Classifier with var_smoothing={var_smoothing} ---------------")
    print("Average Test Accuracy:", avg_accuracy)
    print("Average AUROC:", avg_auroc)
    print("Average F1 Score:", avg_f1_score)
