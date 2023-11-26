import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Load the datasets
x_features = pd.read_csv('normalized_x_features.csv')
y_target = pd.read_csv('y_target_binary.csv')

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(x_features, y_target.values.ravel(), test_size=0.3, random_state=42)

# Initialize the Gaussian Naive Bayes model
nb_classifier = GaussianNB()

# Train the model
nb_classifier.fit(X_train, y_train)

# Predictions on the training and test set
y_pred_train_nb = nb_classifier.predict(X_train)
y_pred_nb = nb_classifier.predict(X_test)

# Calculate accuracies
accuracy_train_nb = accuracy_score(y_train, y_pred_train_nb)
accuracy_nb = accuracy_score(y_test, y_pred_nb)

# Print the results
print("Naive Bayes Classifier ---------------")
print("Training Accuracy:", accuracy_train_nb)
print("Test Accuracy:", accuracy_nb)
