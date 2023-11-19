import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
#knn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

def knn(x, y, k):
    # Create KNN classifier
    knn = KNeighborsClassifier(n_neighbors = k)
    # Fit the classifier to the data
    knn.fit(x, y)
    #show first 5 model predictions on the test data
    #print(knn.predict(x_test)[0:5])
    #check accuracy of our model on the test data
    #accuracy = knn.score(x_test, y_test)
    #print(accuracy)
    # Create a new KNN model
    knn_cv = KNeighborsClassifier(n_neighbors=k)
    # Fit model to the data
    knn_cv.fit(x, y)
    # Calculate the accuracy of the model
    accuracy = cross_val_score(knn_cv, x, y, cv=10)
    print(accuracy)
    print("Accuracy: %0.2f (+/- %0.2f)" % (accuracy.mean(), accuracy.std() * 2))

    return accuracy

def main():
    # Read the datasets
    x_features = pd.read_csv('normalized_x_features.csv')
    y_target = pd.read_csv('y_target_binary.csv')
    y_target = y_target.iloc[:, 0]
    x_features = np.ascontiguousarray(x_features)
    y_target = np.ascontiguousarray(y_target)

    # Run KNN with 10-fold cross-validation
    kf = KFold(n_splits=10, shuffle=True, random_state=334)

    #k-values are odd numbers from 1 to 31
    k_values = list(range(1, 26, 2))
    accuracy_scores = []

    for k in k_values:
        print('k = ', k)
        knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
        scores = cross_val_score(knn, x_features, y_target, cv=kf, scoring='accuracy')
        accuracy_scores.append(scores.mean())

    # Plotting the accuracy for different k values
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, accuracy_scores, marker='o')
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Cross-Validated Accuracy')
    plt.title('KNN Performance Over Different k Values')
    plt.xticks(k_values)
    plt.grid(True)
    plt.savefig('knn.png')
    plt.show()

#hyperparameter: k=1
if __name__ == '__main__':
    main()