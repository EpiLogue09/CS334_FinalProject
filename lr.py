import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score, roc_auc_score, roc_curve

#function to run linear regression with 10-fold cross-validation
#we don't need to do any regularization here
def lr(x, y):
    kf = KFold(n_splits=10, shuffle=True, random_state=334)

    mse_scores = []
    r2_scores = []

    for train_index, test_index in kf.split(x):

        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model = LinearRegression()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        mse_scores.append(mse)
        r2_scores.append(r2)


    # Diagnostic plots
    plt.scatter(range(len(mse_scores)), mse_scores)
    plt.title('MSE Scores for Each Fold')
    plt.xlabel('Fold')
    plt.ylabel('MSE')
    plt.show()

    plt.scatter(range(len(r2_scores)), r2_scores)
    plt.title('R2 Scores for Each Fold')
    plt.xlabel('Fold')
    plt.ylabel('R2')
    plt.show()

    return np.mean(mse_scores), np.mean(r2_scores)


# Function to run Ridge Regression with 10-fold cross-validation
def lr_ridge(x, y, alphas):
    kf = KFold(n_splits=10, shuffle=True, random_state=334)
    # Store the average MSE and R2 for different alphas
    alpha_mse = {}
    alpha_r2 = {}
    for alpha in alphas:
        cv_mse = []
        cv_r2 = []
        for train_index, test_index in kf.split(x):
            x_train, x_test = x.iloc[train_index], x.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]

            model = Ridge(alpha=alpha)
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)

            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            cv_mse.append(mse)
            cv_r2.append(r2)

        alpha_mse[alpha] = np.mean(cv_mse)
        alpha_r2[alpha] = np.mean(cv_r2)

    # Plotting the MSE for different alphas
    plt.figure(figsize=(7, 5))
    plt.plot(list(alpha_mse.keys()), list(alpha_mse.values()), marker='o')
    plt.xlabel('Alpha')
    plt.ylabel('Average MSE')
    plt.title('Average MSE for Different Alphas')
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig('ridge_mse.png')
    plt.show()

    # Plotting the R2 for different alphas
    plt.figure(figsize=(7, 5))
    plt.plot(list(alpha_r2.keys()), list(alpha_r2.values()), marker='o', color='orange')
    plt.xlabel('Alpha')
    plt.ylabel('Average R2 Score')
    plt.title('Average R2 Score for Different Alphas')
    plt.xscale('log')
    plt.savefig('ridge_r2.png')
    plt.show()

    return alpha_mse, alpha_r2


def main():
    # Read the datasets
    x_features = pd.read_csv('normalized_x_features.csv')
    y_target = pd.read_csv('y_target.csv')

    # Assuming y_target.csv has a single column with the target variable
    y_target = y_target.iloc[:, 0]

    # Define the range of alphas to explore
    alpha_list = [1, 5, 10, 100, 500, 1000, 5000]

    # Run Ridge Regression with 10-fold cross-validation
    alpha_mse, alpha_r2 = lr_ridge(x_features, y_target, alpha_list)

    # Print the alpha with the lowest MSE
    min_mse = min(alpha_mse.values())
    min_mse_alpha = [alpha for alpha, mse in alpha_mse.items() if mse == min_mse][0]
    print('(Ridge) alpha with the lowest MSE: ', min_mse_alpha)

    # Print the alpha with the highest R2
    max_r2 = max(alpha_r2.values())
    max_r2_alpha = [alpha for alpha, r2 in alpha_r2.items() if r2 == max_r2][0]
    print('(Ridge) alpha with the highest R2: ', max_r2_alpha)

    # Run Ridge Regression with the best alpha
    model = Ridge(alpha=min_mse_alpha)
    model.fit(x_features, y_target)
    y_pred = model.predict(x_features)

    # Plot the actual vs. predicted values
    plt.figure(figsize=(7, 5))
    plt.scatter(y_target, y_pred)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Ridge: Actual vs. Predicted Values')
    plt.savefig('lr_actual_vs_predicted_ridge.png')

    # Plot the residuals
    plt.figure(figsize=(7, 5))
    plt.scatter(y_pred, y_target - y_pred)
    plt.xlabel('Predicted')
    plt.ylabel('Residuals')
    plt.title('Ridge: Residual Plot')
    plt.savefig('lr_residuals_ridge.png')
    plt.show()

    

#hyperparameter: alpha=500 (for both MSE and R^2)
if __name__ == "__main__":
    main()