import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score, roc_auc_score, roc_curve
from lr import lr, lr_ridge
def main():
    # Read the datasets
    x_features = pd.read_csv('normalized_x_features.csv')
    y_target = pd.read_csv('y_target.csv')

    # Binarize the target variable using the threshold of 75
    y_target['PM_Median'] = (y_target['PM_Median'] > 75).astype(int)
    y_target = y_target.iloc[:, 0]

    # Selecting the optimal hyperparameter: alpha=500
    optimal_alpha = 500

    # Creating the Ridge model with the optimal alpha
    model = Ridge(alpha=optimal_alpha)
    model.fit(x_features, y_target)
    y_pred_scores = model.predict(x_features)

    # Threshold for classification
    threshold = 0.5
    y_pred_binary = (y_pred_scores > threshold).astype(int)

    # Calculating accuracy, F-1 score, and AUC-ROC
    accuracy = accuracy_score(y_target, y_pred_binary)
    f1 = f1_score(y_target, y_pred_binary)
    auc_roc = roc_auc_score(y_target, y_pred_scores)

    print('Accuracy:', accuracy)
    print('F-1 Score:', f1)
    print('AUC-ROC:', auc_roc)

    # Plotting ROC Curve
    fpr, tpr, _ = roc_curve(y_target, y_pred_scores)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color='blue', lw=2, label='ROC')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig('ridge_roc.png')
    plt.show()

"""
Accuracy: 0.7595133649178218
F-1 Score: 0.6471940964475718
AUC-ROC: 0.8158435288488035
"""
if __name__ == "__main__":
    main()