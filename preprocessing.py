import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def cal_corr(df, file_name):
    # calculate the correlation matrix and perform the heatmap
    corrMat = df.corr()

    #perform the heatmap
    plt.figure(figsize=(15,12))
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    sns.heatmap(corrMat, cmap=cmap, center=0, square=True, annot=True,
                linewidths=.5, cbar_kws={"shrink": 0.75},
                annot_kws={"fontsize": 8, "ha": 'center', 'va': 'center'})
    plt.title('Correlation Matrix')
    #plt.savefig('HeatMap_annot.png') #save the plot to a file
    plt.savefig(file_name) #save the plot to a file
    plt.show()
    return corrMat

def correlation_matrix(x, y, file_name, y_col_name):
    #add the target column from yTrain and yTest to the training data
    new_xTrain = x.copy()
    new_xTrain['target'] = y[y_col_name]

    cal_corr(new_xTrain, file_name)

#normalize the data using StandardScaler (z-score normalization)
def normalize_data(df):
    # Define features to normalize
    features_to_normalize = ['year', 'DEWP', 'HUMI', 'PRES', 'TEMP', 'Iws', 'precipitation', 'Iprec']

    # Applying StandardScaler to the selected features
    scaler = StandardScaler()
    df[features_to_normalize] = scaler.fit_transform(df[features_to_normalize])

    # for cyclical features such as hour and day, use sin and cos to transform them
    df['hour_sin'] = np.sin(df.hour * (2. * np.pi / 24))
    df['hour_cos'] = np.cos(df.hour * (2. * np.pi / 24))
    df['day_sin'] = np.sin(df.day * (2. * np.pi / 31))
    df['day_cos'] = np.cos(df.day * (2. * np.pi / 31))
    df['month_sin'] = np.sin(df.month * (2. * np.pi / 12))
    df['month_cos'] = np.cos(df.month * (2. * np.pi / 12))
    df['season_sin'] = np.sin(df.season * (2. * np.pi / 4))
    df['season_cos'] = np.cos(df.season * (2. * np.pi / 4))
    # drop the original columns
    df = df.drop(columns=['hour', 'day', 'month', 'season'])

    return df

def main():
    #read data
    normalized_x = pd.read_csv('normalized_x_features.csv')
    target = pd.read_csv('y_target.csv')
    target_binary = pd.read_csv('y_target_binary.csv')
    #calculate the correlation matrix
    correlation_matrix(normalized_x, target, 'normalized_corrMat.png', 'PM_Median')
    #correlation_matrix(normalized_x, target_binary, 'normalized_corrMat_binary.png', 'PM>75')

    #read data


if __name__ == "__main__":
    main()