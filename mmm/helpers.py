import pandas as pd
import datetime as dt
import os

def guess_date_column(df):
    guesses = ['date', 'Date', 'day', 'Day', 'week', 'Week', 'Month', 'month']
    for x in guesses:
        if x in df.columns:
            return x

    return None

def add_X_labels(df, X_labels, add_cols):
    for x in add_cols:
        if x not in X_labels:
            X_labels.append(x)

    return X_labels

def del_X_labels(df, X_labels, del_cols):
    for x in del_cols:
        if x in X_labels:
            X_labels.remove(x)

    return X_labels

def get_cols_containing(df, containing):
    return [x for x in list(df.columns) if containing in x]

def rename_column(df, column_label, new_name):
    df.rename(columns={column_label: new_name}, inplace=True)

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

def save_model(y_label, X_labels, error, algo="LinearRegression", file_loc=None):
    if file_loc is None:
        file_loc = "/results/models.csv"

    timestamp = dt.datetime.now()

    data = {
        'timestamp': timestamp,
        'y_label': y_label,
        'X_labels': X_labels,
        'algo': algo,
        'error': error
    }
    df = pd.DataFrame([data])

    if os.path.isfile(file_loc):
        models = pd.read_csv(file_loc)
        df = models.append(df)

    df.to_csv(file_loc)


