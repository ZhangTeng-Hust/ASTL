# -*- coding: utf-8 -*-
"""
Created on 14:11,2021/09/13
@author: ZhangTeng
"""
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


def predict(y_true, y_pred):
    '''
    y_true: 
    y_pred: 
    return: 
    '''

    MAE = mean_absolute_error(y_true, y_pred)
    RMSE = np.sqrt(mean_squared_error(y_true, y_pred))
    print("MAE  {}".format(mean_absolute_error(y_true, y_pred)))
    print("RMSE  {}".format(np.sqrt(mean_squared_error(y_true, y_pred))))
    print('-------------------------------')
    result = np.array([MAE, RMSE])
    return result


def predict_i(y_t, y_p, i):
    """
    y_t: 
    y_p: 
    return: 
    """
    y_true = y_t[:, i]
    y_pred = y_p[:, i]

    MAE = mean_absolute_error(y_true, y_pred)
    RMSE = np.sqrt(mean_squared_error(y_true, y_pred))
    print("MAE  {}".format(mean_absolute_error(y_true, y_pred)))
    print("RMSE  {}".format(np.sqrt(mean_squared_error(y_true, y_pred))))
    print('-------------------------------')
    result = np.array([MAE, RMSE])
    return result
