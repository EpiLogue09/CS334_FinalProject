import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt


def knn_plotROC(x_features, y_target):
    kf = KFold(n_splits=10, shuffle=True, random_state=334)
    model = KNeighborsClassifier(n_neighbors=1, metric="euclidean")

    probabilities = cross_val_predict(model, x_features, y_target, cv=kf, method='predict_proba')[:, 1]
    fpr, tpr, _ = roc_curve(y_target, probabilities)

    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve')
    #plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (kNN)')
    plt.legend(loc="lower right")
    plt.savefig('knn_roc.png')
    plt.show()


def main():
    # Read the datasets
    x_features = pd.read_csv('normalized_x_features.csv')
    y_target = pd.read_csv('y_target_binary.csv')
    y_target = y_target.iloc[:, 0]
    x_features = np.ascontiguousarray(x_features)
    y_target = np.ascontiguousarray(y_target)

    # Plot ROC Curve
    knn_plotROC(x_features, y_target)

    # Evaluate KNN model using the best k value (k=1)
    knn = KNeighborsClassifier(n_neighbors=1, metric="euclidean")
    kf = KFold(n_splits=10, shuffle=True, random_state=334)

    accuracy_scores = cross_val_score(knn, x_features, y_target, cv=kf, scoring='accuracy')
    f1_scores = cross_val_score(knn, x_features, y_target, cv=kf, scoring='f1')
    auroc_scores = cross_val_score(knn, x_features, y_target, cv=kf, scoring='roc_auc')

    print(f"Average Accuracy: {np.mean(accuracy_scores)}")
    print(f"Average F-1 Score: {np.mean(f1_scores)}")
    print(f"Average AUROC: {np.mean(auroc_scores)}")

    # Fit the model with the entire dataset
    knn.fit(x_features, y_target)


"""
Average Accuracy: 0.9084401528407382
Average F-1 Score: 0.8749834469426112
Average AUROC: 0.902026336049657
"""
if __name__ == "__main__":
    main()
