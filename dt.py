import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

# Load the datasets
x_features = pd.read_csv('normalized_x_features.csv')
y_target = pd.read_csv('y_target_binary.csv')

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(x_features, y_target.values.ravel(), test_size=0.3, random_state=42)

# Decision Tree Classifier with GINI criterion
dt_classifier_gini = DecisionTreeClassifier(criterion='gini', random_state=42)
dt_classifier_gini.fit(X_train, y_train)
y_pred_train_gini = dt_classifier_gini.predict(X_train)
y_pred_gini = dt_classifier_gini.predict(X_test)
y_pred_proba_gini = dt_classifier_gini.predict_proba(X_test)[:, 1]  # Probabilities for the positive class

# Calculate accuracies, AUROC, and F1 score for GINI model
accuracy_train_gini = accuracy_score(y_train, y_pred_train_gini)
accuracy_gini = accuracy_score(y_test, y_pred_gini)
auroc_gini = roc_auc_score(y_test, y_pred_proba_gini)
f1_gini = f1_score(y_test, y_pred_gini)

# Decision Tree Classifier with ENTROPY criterion
dt_classifier_entropy = DecisionTreeClassifier(criterion='entropy', random_state=42)
dt_classifier_entropy.fit(X_train, y_train)
y_pred_train_entropy = dt_classifier_entropy.predict(X_train)
y_pred_entropy = dt_classifier_entropy.predict(X_test)
y_pred_proba_entropy = dt_classifier_entropy.predict_proba(X_test)[:, 1]  # Probabilities for the positive class

# Calculate accuracies, AUROC, and F1 score for ENTROPY model
accuracy_train_entropy = accuracy_score(y_train, y_pred_train_entropy)
accuracy_entropy = accuracy_score(y_test, y_pred_entropy)
auroc_entropy = roc_auc_score(y_test, y_pred_proba_entropy)
f1_entropy = f1_score(y_test, y_pred_entropy)

# Print the results
print("GINI Criterion ---------------")
print("Training Accuracy:", accuracy_train_gini)
print("Test Accuracy:", accuracy_gini)
print("AUROC:", auroc_gini)
print("F1 Score:", f1_gini)
print("\nEntropy Criterion ---------------")
print("Training Accuracy:", accuracy_train_entropy)
print("Test Accuracy:", accuracy_entropy)
print("AUROC:", auroc_entropy)
print("F1 Score:", f1_entropy)
