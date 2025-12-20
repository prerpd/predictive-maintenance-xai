# Author: Prerna
# Date Created: 2024-06-15
# Description: This script preprocesses the predictive maintenance dataset by handling missing values,
#              encoding categorical variables, and scaling numerical features.
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

def dataCleaner(dataPath, testSize=0.2):
    #input: path to the dataset
    #output: cleaned and preprocessed train and test datasets

    # load dataset:
    dataCSV = pd.read_csv(dataPath)

    # IDs are not used for prediction
    # dropping these columns
    dataCSV.drop(columns=['UDI', 'Product ID'], inplace=True)

    # edit column names
    dataCSV.columns = dataCSV.columns.str.replace(r"[\[\]<>\]]", "", regex=True).str.strip()
    print(dataCSV.columns)

    # encoding categorical variables
    labelEncoder = LabelEncoder()
    dataCSV['Type'] = labelEncoder.fit_transform(dataCSV['Type'])

    # saving the encoder
    joblib.dump(labelEncoder, os.path.join('models', 'labelEncoder.pkl'))

    # handling missing values
    # drop target rows
    dropCols = ['Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    X = dataCSV.drop(columns=dropCols, axis=1)
    y = dataCSV['Machine failure']

    #edit column names


    # split training and testing data
    # 'stratify=y' ensures the small % of failures is evenly distributed
    xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=testSize, random_state=42, stratify=y)

    # scaling numerical features
    scaler = StandardScaler()
    xTrainScaled = scaler.fit_transform(xTrain)
    xTestScaled = scaler.transform(xTest)

    # saving the scaler
    joblib.dump(scaler, os.path.join('models', 'scaler.pkl'))

    # converting back to DataFrame
    xTrainDF = pd.DataFrame(xTrainScaled, columns=X.columns)
    xTestDF = pd.DataFrame(xTestScaled, columns=X.columns)

    return xTrainDF, xTestDF, yTrain.reset_index(drop=True), yTest.reset_index(drop=True)

if __name__ == "__main__":
    xTrain, xTest, yTrain, yTest = dataCleaner(dataPath='data/ai4i2020.csv')
    print(xTrain.columns)
    print(f"Training data shape: {xTrain.shape}, Testing data shape: {xTest.shape}")
    print(f"Failure rate in training set:, {yTrain.mean():.2%}")