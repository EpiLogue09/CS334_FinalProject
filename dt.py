import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, roc_curve
import matplotlib.pyplot as plt

# Load the datasets
x_features = pd.read_csv('normalized_x_features.csv')
y_target = pd.read_csv('y_target_binary.csv')

# Define the number of folds for cross-validation
n_folds = 10
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

# Store FPR, TPR, and AUC for plotting ROC curves
roc_data = {
    'gini': {'fpr': [], 'tpr': [], 'auc': []},
    'entropy': {'fpr': [], 'tpr': [], 'auc': []}
}

# Iterate over each fold
for train_index, test_index in kf.split(x_features):
    X_train, X_test = x_features.iloc[train_index], x_features.iloc[test_index]
    y_train, y_test = y_target.values.ravel()[train_index], y_target.values.ravel()[test_index]

    # Decision Tree Classifier with GINI criterion
    dt_classifier_gini = DecisionTreeClassifier(criterion='gini', random_state=42)
    dt_classifier_gini.fit(X_train, y_train)
    y_pred_proba_gini = dt_classifier_gini.predict_proba(X_test)[:, 1]  # Probabilities for the positive class
    fpr_gini, tpr_gini, _ = roc_curve(y_test, y_pred_proba_gini)
    auc_gini = roc_auc_score(y_test, y_pred_proba_gini)
    roc_data['gini']['fpr'].append(fpr_gini)
    roc_data['gini']['tpr'].append(tpr_gini)
    roc_data['gini']['auc'].append(auc_gini)

    # Decision Tree Classifier with ENTROPY criterion
    dt_classifier_entropy = DecisionTreeClassifier(criterion='entropy', random_state=42)
    dt_classifier_entropy.fit(X_train, y_train)
    y_pred_proba_entropy = dt_classifier_entropy.predict_proba(X_test)[:, 1]  # Probabilities for the positive class
    fpr_entropy, tpr_entropy, _ = roc_curve(y_test, y_pred_proba_entropy)
    auc_entropy = roc_auc_score(y_test, y_pred_proba_entropy)
    roc_data['entropy']['fpr'].append(fpr_entropy)
    roc_data['entropy']['tpr'].append(tpr_entropy)
    roc_data['entropy']['auc'].append(auc_entropy)

# Plot ROC curves
plt.figure(figsize=(10, 6))
for criterion, data in roc_data.items():
    mean_fpr = np.mean(data['fpr'], axis=0)
    mean_tpr = np.mean(data['tpr'], axis=0)
    mean_auc = np.mean(data['auc'])
    plt.plot(mean_fpr, mean_tpr, label=f'{criterion.capitalize()} (AUC = {mean_auc:.2f})')

plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend(loc='lower right')
plt.show()
