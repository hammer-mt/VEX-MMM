import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from typing import List, Tuple

def run_regression(df:pd.DataFrame, y_label:str, X_labels:List[str]) -> Tuple[np.array, np.array, np.array, np.array]:
    y_actual = df[y_label]
    X_data = df[X_labels]

    results = sm.OLS(y_actual, X_data).fit()

    y_pred = results.predict(X_data)
    coefficients = results.params.values
    p_values = results.pvalues.values

    return y_actual, y_pred, coefficients, p_values

def run_regression_with_test_split(df:pd.DataFrame, y_label:str, 
                                   X_labels:List[str], test_size:float=0.2) -> Tuple[np.array, np.array, np.array, np.array, np.array, np.array]:
    y_actual = df[y_label]
    X_data = df[X_labels]

    X_train, X_test, y_train, y_test = train_test_split(X_data, y_actual,
        test_size=test_size, random_state=0)

    results = sm.OLS(y_train, X_train).fit()

    y_test_pred = results.predict(X_test)
    y_pred = results.predict(X_data)
    coefficients = results.params.values
    p_values = results.pvalues.values
    
    return y_test, y_test_pred, coefficients, p_values, y_actual, y_pred

def run_ml_regression(df:pd.DataFrame, y_label:str, X_labels:List[str]) -> Tuple[np.array, np.array, np.array]:
    y_train = df[y_label]
    X_train = df[X_labels]

    results = Ridge(alpha=0.01, fit_intercept=False).fit(X_train, y_train)

    y_pred = results.predict(X_train)
    coefficients = results.coef_

    return y_train, y_pred, coefficients

def run_ml_regression_with_test_split(df:pd.DataFrame, y_label:str, X_labels:List[str], test_size:float=0.2) -> Tuple[np.array, np.array, np.array]:
    y_data = df[y_label]
    X_data = df[X_labels]

    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data,
        test_size=test_size, random_state=0)

    results = Ridge(alpha=0.01, fit_intercept=False, random_state=0).fit(X_train, y_train)

    y_pred = results.predict(X_test)
    coefficients = results.coef_

    return y_test, y_pred, coefficients

def create_results_df(X_labels:List[str], coefficients:np.array, p_values:List[float]=None) -> pd.DataFrame:
    if p_values is None: # ml algo doesn't give p_values
        results_df = pd.DataFrame({'coefficient': coefficients}, index=X_labels)
    else:
        results_df = pd.DataFrame({'coefficient': coefficients, 'p_value': p_values}, index=X_labels)

    return results_df

def create_pred_df(df:pd.DataFrame, results_df:pd.DataFrame) -> pd.DataFrame:
    # calculate predictions
    X_labels = results_df.index
    pred_df = df[X_labels].copy()
    for x in X_labels:
        pred_df[x] = results_df['coefficient'].loc[x] * df[x]

    return pred_df