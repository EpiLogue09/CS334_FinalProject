import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
#auc
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
    C_values = [0.001, 0.01, 0.1, 1, 10, 100, 1000] #create a list of C values

    # Create logistic regression model
    logistic_regression = LogisticRegressionCV(cv=kf, random_state=334, max_iter=500)
    # Fit the model
    logistic_regression.fit(x_features, y_target)
    # Predict the test data
    y_pred = logistic_regression.predict(x_features)
    # Calculate the accuracy of the model
    accuracy = accuracy_score(y_target, y_pred)
    print('Accuracy: ', accuracy)
    # Calculate the confusion matrix
    cm = confusion_matrix(y_target, y_pred)
    print('Confusion Matrix: \n', cm)
    # Calculate the classification report
    cr = classification_report(y_target, y_pred)
    print('Classification Report: \n', cr)
    # Calculate the ROC curve
    fpr, tpr, thresholds = roc_curve(y_target, y_pred)
    # Calculate the AUC
    auc = roc_auc_score(y_target, y_pred)
    print('AUC: ', auc)
    # Plot the ROC curve
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, marker='o')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.grid(True)
    plt.savefig('logistic_regression_roc.png')
    plt.show()
    # Plot the confusion matrix
    plt.figure(figsize=(10, 6))

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xticks([0, 1], ['Not Heavy PM2.5', 'Heavy PM2.5'])
    plt.yticks([0, 1], ['Not Heavy PM2.5', 'Heavy PM2.5'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig('logistic_regression_cm.png')
    plt.show()

    return accuracy, cm, cr, auc, fpr, tpr, thresholds

if __name__ == '__main__':
    main()