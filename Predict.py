#!/usr/bin/env python

"""
EnergyDataSimulationChallenge by Cambridge Energy Data Lab

"""

import pandas as pd
import numpy as np
import os
from sklearn import linear_model, svm, ensemble, model_selection
from sklearn.cross_validation import train_test_split
from datetime import datetime


def import_csv_data():
    """ Import csv data from the project """
    df_train = pd.read_csv('./data/training_dataset_500.csv')
    df_test = pd.read_csv('./data/test_dataset_500.csv')

    # Add date columns

    df_train['Day'] = int(1)
    df_train['Date'] = df_train.apply(lambda x: datetime(int(x['Year']),
                                      int(x['Month']), int(x['Day'])), axis=1)

    df_test['Day'] = int(1)
    df_test['Date'] = df_test.apply(lambda x: datetime(int(x['Year']),
                                    int(x['Month']), int(x['Day'])), axis=1)

    # Drop excess columns

    df_train = df_train.drop(['Day', 'Year', 'Month', 'Label'], axis=1)
    df_test = df_test.drop(['Day', 'Year', 'Month', 'Label'], axis=1)

    return df_train, df_test


def mape(predictions, target):
    ''' Find the mean absolute percentage error '''
    predictions, target = np.array(predictions), np.array(target)
    return np.mean((np.absolute(predictions-target)/target)*100)


def run_random_forest(train_x, train_y, test_x):
    """Run a random forest regressor model"""

    init_model = ensemble.RandomForestRegressor()
    parameters = {'n_estimators':
                  np.linspace(5, trainX.shape[1], 20).astype(int)}
    gridCV = model_selection.GridSearchCV(init_model, parameters, cv=10)

    train_x_split, test_x_split, train_y_split, test_y_split =
    train_test_split(train_x,  train_y, test_size=500)

    gridCV.fit(train_x_split, train_y_split)

    n_estimators = gridCV.best_params_['n_estimators']
    print n_estimators

    model = ensemble.RandomForestRegressor(n_estimators=n_estimators)

    print "Fitting model..."

    model.fit(train_x, train_y)
    predictions = model.predict(test_x)

    return predictions


def run_svr(train_x, train_y, test_x):
    """Run a support vector regression model"""

    init_model = svm.SVR()

    train_x_split, test_x_split, train_y_split, test_y_split =
    train_test_split(train_x, train_y, test_size=500)

    model = svm.SVR(gamma=0.005)

    print "Fitting model..."

    model.fit(train_x, train_y)
    predictions = model.predict(test_x)

    return predictions


def run_ridge(train_x, train_y, test_x):
    """Run a Ridge model"""

    model = linear_model.RidgeCV(alphas=np.logspace(-1, 3, 100), cv=5)

    print "Fitting model..."

    model.fit(train_x, train_y)
    predictions = model.predict(test_x)

    return predictions


def run_gbr(train_x, train_y, test_x):
    """Run a Gradient Bosted Regressor model"""

    parameters = {'loss': 'ls',
                  'n_estimators': 1000,
                  'learning_rate': 0.0005,
                  'max_depth': 4,
                  'subsample': 0.5}

    model = ensemble.GradientBoostingRegressor(**parameters)

    print "Fitting model..."

    model.fit(train_x, train_y)
    predictions = model.predict(test_x)

    return predictions


def main():

    print "Importing data..."

    train, test = import_csv_data()

    print "Getting trainX and testX data"

    train_x_all = train[['Temperature','Daylight']].values[:,1:]
    test_x_all = test[['Temperature','Daylight']].values[:,1:]

    print "Getting trainY and testY data"

    train_y_all = train['EnergyProduction'].values
    test_y_actual = test['EnergyProduction'].values

    print "Running Gradient Boosted Regressor"

    predictions_GBR = run_gbr(train_x_all, train_y_all, test_x_all)
    print "MAPE, GBR: ", mape(predictions_GBR, testY_actual)

    print "Running random forest"

    predictions_rf = Run_random_forest(train_x_all, train_y_all, test_x_all)
    print "MAPE, RF: ", mape(predictions_rf, test_y_actual)

    print "Running Ridge"

    predictions_ridge = run_ridge(train_x_all, train_y_all, test_x_all)
    print "MAPE, ridge: ", mape(predictions_ridge, test_y_actual)

    print "Running SVR"

    predictions_svr = run_svr(train_x_all, train_y_all, test_x_all)
    print "MAPE, SVR: ", mape(predictions_svr, test_y_actual)


if __name__ == "__main__":
    main()
