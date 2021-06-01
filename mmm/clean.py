import numpy as np

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