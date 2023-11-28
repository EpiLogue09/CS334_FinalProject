import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

# Load the datasets
x_features = pd.read_csv('normalized_x_features.csv')
y_target = pd.read_csv('y_target_binary.csv')

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(x_features, y_target.values.ravel(), test_size=0.3, random_state=42)

# List of variance smoothing values to try
var_smoothing_list = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]

# Iterate over the variance smoothing values
for var_smoothing in var_smoothing_list:
    # Initialize the Gaussian Naive Bayes model with the current var_smoothing
    nb_classifier = GaussianNB(var_smoothing=var_smoothing)

    # Train the model
    nb_classifier.fit(X_train, y_train)

    # Predictions on the test set
    y_pred_nb = nb_classifier.predict(X_test)
    y_pred_proba_nb = nb_classifier.predict_proba(X_test)[:, 1] # Probabilities for the positive class

    # Calculate accuracies and AUROC
    accuracy_nb = accuracy_score(y_test, y_pred_nb)
    auroc_nb = roc_auc_score(y_test, y_pred_proba_nb)
    f1_score_nb = f1_score(y_test, y_pred_nb)  # Calculate F1 score

    # Print the results
    print(f"Naive Bayes Classifier with var_smoothing={var_smoothing} ---------------")
    print("Test Accuracy:", accuracy_nb)
    print("AUROC:", auroc_nb)
    print("F1 Score:", f1_score_nb)
