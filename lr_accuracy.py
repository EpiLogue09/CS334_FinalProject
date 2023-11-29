import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve

def main():
    # Read the datasets
    x_features = pd.read_csv('normalized_x_features.csv')
    y_target = pd.read_csv('y_target.csv')

    # Binarize the target variable using the threshold of 75
    y_target['PM_Median'] = (y_target['PM_Median'] > 75).astype(int)
    y_target = y_target.iloc[:, 0]

    # 10-fold cross-validation setup
    kf = KFold(n_splits=10, shuffle=True, random_state=334)

    # Lists to store the metrics
    accuracies, f1_scores, auc_rocs = [], [], []

    # Selecting the optimal hyperparameter: alpha=500
    optimal_alpha = 500

    for train_index, test_index in kf.split(x_features):
        X_train, X_test = x_features.iloc[train_index], x_features.iloc[test_index]
        y_train, y_test = y_target.iloc[train_index], y_target.iloc[test_index]

        # Creating the Ridge model with the optimal alpha and fitting it
        model = Ridge(alpha=optimal_alpha)
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
    plt.title('ROC Curve (LR-Ridge)')
    plt.legend(loc="lower right")
    plt.savefig('lr_roc.png')
    plt.show()
"""
Average Accuracy: 0.7591391484679717
Average F-1 Score: 0.6457939620387042
Average AUC-ROC: 0.815432967680567
"""
if __name__ == "__main__":
    main()
