import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

def run_regression(df, y_label, X_labels):
    y_train = df[y_label]
    X_train = df[X_labels]

    results = sm.OLS(y_train, X_train).fit()

    y_test = y_train # test model on same dataset
    y_pred = results.predict(X_train)
    coefficients = results.params.values
    p_values = results.pvalues.values

    return y_test, y_pred, coefficients, p_values

def run_regression_with_test_split(df, y_label, X_labels, test_size=0.2):
    y_data = df[y_label]
    X_data = df[X_labels]

    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data,
        test_size=test_size, random_state=0)

    results = sm.OLS(y_train, X_train).fit()

    y_pred = results.predict(X_test)
    coefficients = results.params.values
    p_values = results.pvalues.values

    return y_test, y_pred, coefficients, p_values

def run_ml_regression(df, y_label, X_labels):
    y_train = df[y_label]
    X_train = df[X_labels]

    results = Ridge(alpha=0.01, fit_intercept=False).fit(X_train, y_train)

    y_pred = results.predict(X_train)
    coefficients = results.coef_

    return y_train, y_pred, coefficients

def run_ml_regression_with_test_split(df, y_label, X_labels, test_size=0.2):
    y_data = df[y_label]
    X_data = df[X_labels]

    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data,
        test_size=test_size, random_state=0)

    results = Ridge(alpha=0.01, fit_intercept=False, random_state=0).fit(X_train, y_train)

    y_pred = results.predict(X_test)
    coefficients = results.coef_

    return y_test, y_pred, coefficients

def create_results_df(X_labels, coefficients, p_values=None):
    if p_values is None: # ml algo doesn't give p_values
        results_df = pd.DataFrame({'coefficients': coefficients}, index=X_labels)
    else:
        results_df = pd.DataFrame({'coefficients': coefficients, 'p_values': p_values}, index=X_labels)

    return results_df

def create_pred_df(df, X_labels, coefficients, date_column):
    # calculate predictions
    coef_df = pd.DataFrame(X_labels, coefficients)
    pred_df = df[X_labels].copy()
    pred_df[date_column] = df[date_column]
    for x in X_labels:
        try:
            pred_df[x] = coef_df.loc[x] * df[x]
        except:
            pred_df[x] = 0

    return pred_df