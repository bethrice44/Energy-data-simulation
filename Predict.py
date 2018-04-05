#!/usr/bin/env python

"""
EnergyDataSimulationChallenge by Cambridge Energy Data Lab

"""

import pandas as pd
import numpy as np
import os
from sklearn import linear_model, svm, ensemble, grid_search
from sklearn.cross_validation import train_test_split
from datetime import datetime


SEED=48


def import_csv_data():
	
    df_train=pd.read_csv('../../data/training_dataset_500.csv')
    df_test=pd.read_csv('../../data/test_dataset_500.csv')
    
    # Add date columns
    
    df_train['Day']=int(1)
    df_train['Date'] = df_train.apply(lambda x: datetime(int(x['Year']), int(x['Month']), int(x['Day'])),axis=1)
        
    df_test['Day']=int(1)
    df_test['Date'] = df_test.apply(lambda x: datetime(int(x['Year']), int(x['Month']), int(x['Day'])),axis=1)
        
        
    # Drop excess columns
    
    df_train=df_train.drop(['Day','Year','Month','Label'],axis=1)
    df_test=df_test.drop(['Day','Year','Month','Label'],axis=1)


    return df_train,df_test


def MAPE(predictions,target):
	''' Find the mean absolute percentage error '''
	predictions,target=np.array(predictions),np.array(target)
	return np.mean((np.absolute(predictions-target)/target)*100)


def Run_random_forest(trainX,trainY,testX):
    """Run a random forest regressor model"""
    
    init_model=ensemble.RandomForestRegressor()
    parameters={'n_estimators':np.linspace(5, trainX.shape[1], 20).astype(int)}
    gridCV=grid_search.GridSearchCV(init_model,parameters,cv=10)
    
    trainX_split,testX_split,trainY_split,testY_split = train_test_split(trainX,trainY,test_size=500)
    gridCV.fit(trainX_split,trainY_split)

    n_estimators = gridCV.best_params_['n_estimators']
    print n_estimators
    
    model=ensemble.RandomForestRegressor(n_estimators=n_estimators)
    
    print "Fitting model..."
    
    model.fit(trainX,trainY)
    predictions=model.predict(testX)
    
    return predictions


def Run_SVR(trainX,trainY,testX):
    """Run a support vector regression model"""
    
    init_model=svm.SVR()
    parameters={'C':np.logspace(-5,5,10), 'gamma':np.logspace(-5,5,10), 'epsilon':np.logspace(-2,2,10)}
    gridCV=grid_search.GridSearchCV(init_model,parameters,cv=10)
    
    trainX_split,testX_split,trainY_split,testY_split = train_test_split(trainX,trainY,test_size=500)
    gridCV.fit(trainX_split,trainY_split)

    gamma = gridCV.best_params_['gamma']
    C = gridCV.best_params_['C']
    epsilon = gridCV.best_params_['epsilon']
    
    print gamma, C, epsilon
    
    model=svm.SVR(C=C, gamma=gamma, epsilon=epsilon)
    
    print "Fitting model..."
    
    model.fit(trainX,trainY)
    predictions=model.predict(testX)
    
    return predictions


def Run_ridge(trainX,trainY,testX):
    """Run a Ridge model"""
    
    model=linear_model.RidgeCV(alphas=np.logspace(-0,3,100),cv=5)
    
    print "Fitting model..."
    
    model.fit(trainX,trainY)
    predictions=model.predict(testX)
    
    return predictions



def main():

    train,test=import_csv_data()
    
    TrainX_all=train[['Temperature','Daylight']].values[:,1:]
    testX_all=test[['Temperature','Daylight']].values[:,1:]

    TrainY_all=train['EnergyProduction'].values
    testY_actual=test['EnergyProduction'].values

    Predictions_RF=Run_random_forest(TrainX_all,TrainY_all,testX_all)
    print "MAPE, RF: ", MAPE(Predictions_RF,testY_actual)

    Predictions_SVR=Run_SVR(TrainX_all,TrainY_all,testX_all)
    print "MAPE, SVR: ", MAPE(Predictions_SVR,testY_actual)

    Predictions_Ridge=Run_ridge(TrainX_all,TrainY_all,testX_all)
    print "MAPE, ridge: ", MAPE(Predictions_ridge,testY_actual)





if __name__ == "__main__":
	main()

#MAEs = 0
#    for i in range(N):
#trainX, X_CV, trainY, Y_CV = train_test_split(X, Y,test_size=500,random_state = i*SEED)
#model.fit(trainX, trainY)
#predictions = model.predict(X_CV)
#mae = MAPE(Y_CV,predictions)
#MAEs += mae
