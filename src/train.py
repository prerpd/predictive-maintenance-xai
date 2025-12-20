# Author: Prerna
# Date Created: 2024-06-15
# Description: This script trains a predictive maintenance model using sensor data.

import xgboost as xgb
import sklearn
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
import time
from preprocess import dataCleaner


def predictiveModel(xTrain, yTrain, xTest, yTest):
    # input: preprocessed training and testing datasets
    # output: trained model saved to disk, evaluation metrics printed

    # handling class imbalance using SMOTE
    smote = SMOTE(random_state=42)
    with tqdm_joblib(tqdm(desc="Applying SMOTE", total=1)):
        xTrainRes, yTrainRes = smote.fit_resample(xTrain, yTrain)

    print(f"Post-SMOTE Training Size: {len(xTrainRes)} samples")

    # initializing the XGBoost classifier
    model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )

    # training the model
    with tqdm_joblib(tqdm(desc="Training XGBoost Model", total=1)):
        model.fit(xTrainRes, yTrainRes)

    # saving the trained model
    if not os.path.exists('models'):
        os.makedirs('models')
    joblib.dump(model, os.path.join('models', 'xgboost_model.pkl'))

    # making predictions on the test set
    yPred = model.predict(xTest)

    # evaluating the model
    print("Confusion Matrix:")
    print(confusion_matrix(yTest, yPred))
    print("\nClassification Report:")
    print(classification_report(yTest, yPred))

    return model, yPred



if __name__ == "__main__":
    # load and process the data
    xTrain, xTest, yTrain, yTest = dataCleaner(dataPath='data/ai4i2020.csv')
    print(f"Training data shape: {xTrain.shape}, Testing data shape: {xTest.shape}")
    print(f"Failure rate in training set:, {yTrain.mean():.2%}")
    predictiveModel(xTrain, yTrain, xTest, yTest)