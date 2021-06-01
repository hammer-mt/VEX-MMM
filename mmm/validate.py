import numpy as np
from sklearn import metrics

def calculate_r2(y_actual, y_pred):
    # r squared score
    return round(metrics.r2_score(y_actual, y_pred),3)

def calculate_mae(y_actual, y_pred):
    # mean absolute error
    return round(metrics.mean_absolute_error(y_actual, y_pred),3)

def calculate_mape(y_actual, y_pred):
    # mean absolute percentage error
    return round(metrics.mean_absolute_error(y_actual, y_pred)/np.sum(y_actual),3)

def calculate_mse(y_actual, y_pred):
    # mean squared error
    return round(metrics.mean_squared_error(y_actual, y_pred),3)

def calculate_rmse(y_actual, y_pred):
    # root mean squared error
    return round(np.sqrt(metrics.mean_squared_error(y_actual, y_pred)),3)

def calculate_nrmse(y_actual, y_pred):
    # normalized root mean square error
    return round(np.sqrt(metrics.mean_squared_error(y_actual, y_pred))/y_actual.mean(),3)