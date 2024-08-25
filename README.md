# Project-kaggle-competition-House-Prices---Advanced-Regression-Techniques


New update: Increased accuracy and an imporved score of 0.14671 on Kaggle by Hyperparameter Tuning of the xgb_model



# House Prices - Advanced Regression Techniques

This project is part of the Kaggle competition: **House Prices - Advanced Regression Techniques**. The goal of the competition is to predict the sales prices of houses based on various features of the properties. This repository contains the code I used to participate in the competition, where I previosly achieved a leaderboard score of 0.15113 ,  new score is 0.14671 by tuning the xgb_model.

## Project Overview

The project involves using machine learning techniques to predict house prices. Two models were used in this project:
- **Random Forest Regressor**
- **XGBoost Regressor**

The models were trained using features such as `LotArea`, `YearBuilt`, `1stFlrSF`, `GrLivArea`, and several others. The final predictions were made using the model with the best performance, evaluated based on Mean Absolute Error (MAE).

## Files

- **train.csv:** The training dataset containing features and target values.
- **test.csv:** The test dataset used to make final predictions.
- **submission.csv:** The submission file containing the predicted house prices for the test dataset.
- **model.py:** The Python script that includes the data preprocessing, model training, evaluation, and prediction.
