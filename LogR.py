import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


def main():
    # Read the datasets
    x_features = pd.read_csv('normalized_x_features.csv')
    y_target = pd.read_csv('y_target_binary.csv')
    y_target = y_target.iloc[:, 0]
    x_features = np.ascontiguousarray(x_features)
    y_target = np.ascontiguousarray(y_target)
    # Run Logistic Regression with 10-fold cross-validation
    kf = KFold(n_splits=10, shuffle=True, random_state=334)
    C_values = [0.001, 0.01, 0.1, 1, 10, 100, 1000]  # List of C values

    # Create logistic regression model with specified C values
    logistic_regression = LogisticRegressionCV(cv=kf, Cs=C_values, random_state=334, max_iter=500)

    # Fit the model
    logistic_regression.fit(x_features, y_target)

    # Extract the best C value
    best_C = logistic_regression.C_
    print("Best C values for each fold:", best_C)
    print("Mean Best C value:", np.mean(best_C))


#best C-value = 1000 (best mean accuracy)
if __name__ == '__main__':
    main()