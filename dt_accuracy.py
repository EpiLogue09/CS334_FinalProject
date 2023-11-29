import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, roc_curve
import matplotlib.pyplot as plt


def main():
    # Read the datasets
    x_features = pd.read_csv('normalized_x_features.csv')
    y_target = pd.read_csv('y_target_binary.csv')

    # 10-fold cross-validation setup
    kf = KFold(n_splits=10, shuffle=True, random_state=334)

    # Lists to store the metrics
    accuracies, f1_scores, auc_rocs = [], [], []

    # Selecting the optimal hyperparameter: alpha=500
    optimal_params = {'criterion': 'gini', 'max_depth': 7, 'min_samples_leaf': 4}

    for train_index, test_index in kf.split(x_features):
        X_train, X_test = x_features.iloc[train_index], x_features.iloc[test_index]
        y_train, y_test = y_target.iloc[train_index], y_target.iloc[test_index]

        # Creating the Ridge model with the optimal parameters and fitting it
        model = DecisionTreeClassifier(**optimal_params)
        model.fit(X_train, y_train)

        # Predicting and scoring
        y_pred_scores = model.predict(X_test)
        threshold = 0.5
        y_pred_binary = (y_pred_scores > threshold).astype(int)

        # Calculating metrics
        accuracies.append(accuracy_score(y_test, y_pred_binary))
        f1_scores.append(f1_score(y_test, y_pred_binary))
        auc_rocs.append(roc_auc_score(y_test, y_pred_scores))

    # Calculating average of the metrics
    avg_accuracy = np.mean(accuracies)
    avg_f1 = np.mean(f1_scores)
    avg_auc_roc = np.mean(auc_rocs)

    print('Average Accuracy:', avg_accuracy)
    print('Average F-1 Score:', avg_f1)
    print('Average AUC-ROC:', avg_auc_roc)

    # Plotting ROC Curve for the last fold
    fpr, tpr, _ = roc_curve(y_test, y_pred_scores)
    plt.figure()
    plt.plot(fpr, tpr, label='ROC')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (Decision Tree)')
    plt.legend(loc="lower right")
    plt.savefig('dt_roc.png')
    plt.show()

if __name__ == "__main__":
    main()