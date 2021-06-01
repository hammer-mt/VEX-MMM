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