import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # Read the datasets
    x_features = pd.read_csv('normalized_x_features.csv')
    y_target = pd.read_csv('y_target_binary.csv')
    y_target = y_target.iloc[:, 0]
    x_features = np.ascontiguousarray(x_features)
    y_target = np.ascontiguousarray(y_target)

    # Create logistic regression model with the optimal C value
    model = LogisticRegression(C=1000, max_iter=500)

    # Setup K-Fold Cross-Validation
    kf = KFold(n_splits=10, shuffle=True, random_state=334)

    # Perform cross-validation for metrics
    accuracy_scores = cross_val_score(model, x_features, y_target, cv=kf, scoring='accuracy')
    f1_scores = cross_val_score(model, x_features, y_target, cv=kf, scoring='f1')
    auroc_scores = cross_val_score(model, x_features, y_target, cv=kf, scoring='roc_auc')

    print(f"Average Accuracy: {np.mean(accuracy_scores)}")
    print(f"Average F-1 Score: {np.mean(f1_scores)}")
    print(f"Average AUROC: {np.mean(auroc_scores)}")

    # Get predictions for ROC and confusion matrix
    predictions = cross_val_predict(model, x_features, y_target, cv=kf)
    probabilities = cross_val_predict(model, x_features, y_target, cv=kf, method='predict_proba')[:, 1]

    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(y_target, probabilities)
    plt.figure()
    plt.plot(fpr, tpr, label='ROC')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (Logistic)')
    plt.legend(loc="lower right")
    plt.savefig('logistic_roc.png')
    plt.show()

    # Plot Confusion Matrix
    predictions_mapped = np.where(predictions == 0, 'Low PM2.5', 'High PM2.5')
    y_target_mapped = np.where(y_target == 0, 'Low PM2.5', 'High PM2.5')

    # Calculate the confusion matrix using string labels
    cm = confusion_matrix(y_target_mapped, predictions_mapped, labels=['Low PM2.5', 'High PM2.5'])

    # Plot Confusion Matrix
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", linewidths=.5, square=True, cmap='Blues', xticklabels=['Low PM2.5', 'High PM2.5'],
                yticklabels=['Low PM2.5', 'High PM2.5'])
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')
    plt.show()

"""
Average Accuracy: 0.7696555880650141
Average F-1 Score: 0.6706168605410563
Average AUROC: 0.8274965217937462
"""
if __name__ == "__main__":
    main()
