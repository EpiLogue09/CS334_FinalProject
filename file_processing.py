#preprocess the data
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer

def file_to_numpy(filename):
    """
    Read an input file and convert it to numpy
    """
    df = pd.read_csv(filename)
    return df.to_numpy()

def preprocess_data():
    #read data
    df_beijing = pd.read_csv('BeijingPM20100101_20151231.csv')
    df_shanghai = pd.read_csv('ShanghaiPM20100101_20151231.csv')

    # For Beijing
    #add a column to indicate the city
    df_beijing['City'] = 'Beijing'
    # drop all rows with 3 NA values in all PM2.5 readings
    df_beijing = df_beijing.dropna(subset=['PM_US Post', 'PM_Dongsi', 'PM_Dongsihuan', 'PM_Nongzhanguan'], how='all')
    # consider the case when all three readings are NA and the case when only one of the readings is NA
    df_beijing['PM_Median'] = df_beijing[['PM_US Post', 'PM_Dongsi', 'PM_Dongsihuan','PM_Nongzhanguan']].median(axis=1)
    df_beijing = df_beijing.dropna(subset=['PM_Median'])
    # drop the original PM2.5 readings
    df_beijing = df_beijing.drop(columns=['PM_US Post', 'PM_Dongsi', 'PM_Dongsihuan','PM_Nongzhanguan'])
    #drop the No column
    df_beijing = df_beijing.drop(columns=['No'])

    #For Shanghai
    #add a column to indicate the city
    df_shanghai['City'] = 'Shanghai'
    # drop all rows with 3 NA values in all PM2.5 readings
    df_shanghai = df_shanghai.dropna(subset=['PM_US Post', 'PM_Xuhui', 'PM_Jingan'], how='all')
    # consider the case when all three readings are NA and the case when only one of the readings is NA
    df_shanghai['PM_Median'] = df_shanghai[['PM_US Post', 'PM_Xuhui', 'PM_Jingan']].median(axis=1)
    df_shanghai = df_shanghai.dropna(subset=['PM_Median'])
    # drop the original PM2.5 readings
    df_shanghai = df_shanghai.drop(columns=['PM_US Post', 'PM_Xuhui', 'PM_Jingan'])
    #drop the No column
    df_shanghai = df_shanghai.drop(columns=['No'])

    #combine the two dataframes
    df = pd.concat([df_beijing, df_shanghai])
    #use one-hot encoding to encode the categorical variable 'cbwd', which indicates the wind direction
    df = pd.get_dummies(df, columns=['cbwd'])
    #use one-hot on the City column
    df = pd.get_dummies(df, columns=['City'])
    #drop the City_Beijing column since there are only two cities
    df = df.drop(columns=['City_Beijing'])
    # Impute missing values using median
    imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    columns_to_impute = ['DEWP', 'HUMI', 'PRES', 'TEMP', 'Iws', 'precipitation', 'Iprec']
    df[columns_to_impute] = imputer.fit_transform(df[columns_to_impute])

    return df

def normalize_data(df):
    # Define features to normalize
    features_to_normalize = ['year', 'DEWP', 'HUMI', 'PRES', 'TEMP', 'Iws', 'precipitation', 'Iprec']

    # Applying StandardScaler to the selected features
    scaler = StandardScaler()
    df[features_to_normalize] = scaler.fit_transform(df[features_to_normalize])

    #for cyclical features such as hour and day, use sin and cos to transform them
    df['hour_sin'] = np.sin(df.hour*(2.*np.pi/24))
    df['hour_cos'] = np.cos(df.hour*(2.*np.pi/24))
    df['day_sin'] = np.sin(df.day*(2.*np.pi/31))
    df['day_cos'] = np.cos(df.day*(2.*np.pi/31))
    df['month_sin'] = np.sin(df.month*(2.*np.pi/12))
    df['month_cos'] = np.cos(df.month*(2.*np.pi/12))
    df['season_sin'] = np.sin(df.season*(2.*np.pi/4))
    df['season_cos'] = np.cos(df.season*(2.*np.pi/4))
    #drop the original columns
    df = df.drop(columns=['hour', 'day', 'month', 'season'])

    return df

def preprocess_data_splitxy(df):
    #split the data into features and target
    x = df.drop(columns=['PM_Median'])
    y = df[['PM_Median']]
    return x, y

#split the data into train and test
def split_data(x, y):
    xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2, random_state=334)
    return xTrain, xTest, yTrain, yTest

def y_to_binary(yTrain, yTest):
    #convert the target variable to binary
    yTrain = np.where(yTrain > 75, 1, 0)
    yTest = np.where(yTest > 75, 1, 0)
    return yTrain, yTest

def select_features(df):
    #select the features that have a correlation coefficient greater than 0.1 with the target
    feature_list = ['year','season','HUMI','TEMP','Iws','cbwd_NE','cbwd_NW','cbwd_SE','cbwd_cv','City_Beijing','City_Shanghai']
    df_w_selected_features = df[feature_list]
    return df_w_selected_features

def main():
    #preprocess the data
    df = preprocess_data()
    #split the data into features and target
    x, y = preprocess_data_splitxy(df)
    #select the features that have a correlation coefficient greater than 0.1 with the target
    #x = select_features(x)
    #save the features and target
    x.to_csv('x_features.csv', index=False)
    y.to_csv('y_target.csv', index=False)

    #change all the target values to binary and save
    y_bin = np.where(y > 75, 1, 0)
    y_bin = pd.DataFrame(y_bin, columns=['PM>75'])
    y_bin.to_csv('y_target_binary.csv', index=False)

    #normalize the data
    normalized_x = normalize_data(x)
    #save the normalized data
    normalized_x.to_csv('normalized_x_features.csv', index=False)



if __name__ == "__main__":
    main()