import numpy as np
import pandas as pd
from sklearn import metrics
import statsmodels.stats.api as sms
from statsmodels.compat import lzip
from typing import List, Tuple, Union

def calculate_r2(y_actual: Union[np.array, List[float], pd.Series], 
                 y_pred: Union[np.array, List[float], pd.Series]) -> float:
    # r squared score
    return round(metrics.r2_score(y_actual, y_pred),3)

def calculate_mae(y_actual: Union[np.array, List[float], pd.Series], 
                  y_pred: Union[np.array, List[float], pd.Series]) -> float:
    # mean absolute error
    return round(metrics.mean_absolute_error(y_actual, y_pred),3)

def calculate_mape(y_actual: Union[np.array, List[float], pd.Series], 
                   y_pred: Union[np.array, List[float], pd.Series]) -> float:
    # mean absolute percentage error
    return round(metrics.mean_absolute_error(y_actual, y_pred)/np.sum(y_actual),3)

def calculate_mse(y_actual: Union[np.array, List[float], pd.Series], 
                  y_pred: Union[np.array, List[float], pd.Series]) -> float:
    # mean squared error
    return round(metrics.mean_squared_error(y_actual, y_pred),3)

def calculate_rmse(y_actual: Union[np.array, List[float], pd.Series], 
                   y_pred: Union[np.array, List[float], pd.Series]) -> float:
    # root mean squared error
    return round(np.sqrt(metrics.mean_squared_error(y_actual, y_pred)),3)

def calculate_nrmse(y_actual: Union[np.array, List[float], pd.Series], 
                    y_pred: Union[np.array, List[float], pd.Series]) -> float:
    # normalized root mean square error
    return round(np.sqrt(metrics.mean_squared_error(y_actual, y_pred))/y_actual.mean(),3)

def run_jarque_bera_tests(residuals: Union[np.array, List[float], pd.Series]) -> List[Tuple[str, float]]:
    # Tests for normality of the residuals
    # skewness should be between -2 and 2
    # kurtosis should be between -7 and 7
    name = ['Jarque-Bera', 'Chi^2 prob', 'Skewness', 'Kurtosis']
    test = sms.jarque_bera(residuals)

    return lzip(name, test)

def run_breuschpagan_tests(residuals: Union[np.array, List[float], pd.Series], 
                           exog) -> List[Tuple[str, float]]:
    # tests for heteroskedasticity
    # p-value should be less than 0.05
    name = ['Lagrange', 'p-value','f-value', 'f p-value']
    test = sms.het_breuschpagan(residuals, exog)

    return lzip(name, test)

def run_goldfeldquandt_tests(residuals: Union[np.array, List[float], pd.Series], 
                             exog) -> List[Tuple[str, float]]:
    # tests for heteroskedasticity
    # p-value should be less than 0.05
    name = ['F statistic', 'p-value']
    test = sms.het_breuschpagan(residuals, exog)

    return lzip(name, test)

# # need a way to run without passing results object
# def run_harvey_collier_tests(residuals: Union[np.array, List[float], pd.Series], exog):
#     # p-value should be less than 0.05
#     name = ['t value', 'p value']
#     test = sms.linear_harvey_collier(results)

#     return lzip(name, test)

def run_condition_number_test(exog):
    # tests for multicollinearity
    # condition no should be less than 30
    condition = np.linalg.cond(exog)

def run_ljungbox_tests(residuals: Union[np.array, List[float], pd.Series], 
                       X_labels:List[str]) -> List[Tuple[str, float]]:
    # tests for autocorrelation
    # p-value should be less than 0.05
    name = ['Ljung-Box stat', 'p-value']
    lags = min(len(X_labels)/2-2, 40)
    test = sms.acorr_ljungbox(residuals, lags=[lags])
    return lzip(name, test)

def run_durbin_watson_test(residuals):
    # tests for autocorrelation
    # durbin watson should be between 1.5 and 2.5
    test = sms.durbin_watson(residuals)
    return ('Durbin Watson', test)

# # need a way to run without passing results object
# def run_rainbox_tests(residuals: Union[np.array, List[float], pd.Series], 
#                       X_labels: List[str]):
#     # tests for linearity
#     # p-value should be less than 0.05
#     name = ['rainbow F stat', 'rainbow F stat p-value']
#     test = sms.linear_rainbow(results)
#     return lzip(name, test)