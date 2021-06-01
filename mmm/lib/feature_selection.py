def guess_date_column(df):
    guesses = ['date', 'Date', 'day', 'Day', 'week', 'Week', 'Month', 'month']
    for x in guesses:
        if x in df.columns:
            return x

    return None