import numpy as np
import pandas as pd

def make_column_index(df, column_label):
    df.index = df[column_label]
    df.drop(column_label, axis=1, inplace=True)
    df.index.name = None

def rename_column(df, column_label, new_name):
    df.rename(columns={column_label: new_name}, inplace=True)

def remove_outliers(df, column_label):
    raw_data = df[column_label]
    mean = np.mean(raw_data)
    std_dev = np.std(raw_data)
    outliers_cutoff = std_dev * 3
    lower_limit = mean - outliers_cutoff
    upper_limit = mean + outliers_cutoff

    no_outliers = raw_data.apply(lambda x: mean if x > upper_limit or x < lower_limit else x)

    outlier_column = f'{column_label} (-outliers)'
    df[outlier_column] = no_outliers
    return outlier_column

def unstack_data(df, metric_column, unstack_column):
    pivoted = pd.pivot_table(df, index=['date'], values=[metric_column], columns=[unstack_column], aggfunc=[np.sum])
    pivoted.columns = pivoted.columns.droplevel(0)
    pivoted.columns.name = None
    pivoted = pivoted.reset_index()
    pivoted.columns = [col[1] for col in pivoted.columns]

    metric_columns = list(pivoted.columns[1:])
    metric_columns = [f"{c} | {metric_column}" for c in metric_columns]

    pivoted.columns = ["date"] + metric_columns
    pivoted.fillna(0, inplace=True)

    return pivoted