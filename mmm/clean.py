import numpy as np
import pandas as pd
import datetime as dt

def make_column_index(df:pd.DataFrame, column_label:str) -> None:
    df.index = df[column_label]
    df.drop(column_label, axis=1, inplace=True)
    df.index.name = None

def rename_column(df:pd.DataFrame, column_label:str, new_name:str) -> None:
    df.rename(columns={column_label: new_name}, inplace=True)

def remove_outliers(df:pd.DataFrame, column_label:str) -> str:
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

def unstack_data(df:pd.DataFrame, metric_column:str, unstack_column:str) -> pd.DataFrame:
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

def transpose_data(df:pd.DataFrame) -> pd.DataFrame:
    date_col = df.columns[0]
    df = df.T
    df.columns = df.iloc[0]
    df.drop(df.index[0], inplace=True)
    df.reset_index(inplace=True)
    df.rename(columns={"index": date_col}, inplace=True)
    df = df.rename_axis(None, axis = 1)
    return df

def interpolate_weekly_data(df, date_col=None, resample_col=None):
    df = df.copy()

    if date_col == None:
        date_col = df.columns[0]
        
    if resample_col == None:
        resample_col = df.columns[1]
        
    df[date_col] = df[date_col].apply(lambda x: dt.datetime.strptime(f"{x}-1", "%Y-%W-%w")) # mondays
    df[date_col] = pd.to_datetime(df[date_col]) # datetime
    df.set_index(date_col, inplace=True)
    df_reindexed = df.reindex(pd.date_range(start=df.index.min(),
                                        end=df.index.max() + dt.timedelta(days=6),
                                        freq='1D'))

    col_to_resample = df_reindexed.columns[0]
    df_reindexed[col_to_resample] = df_reindexed[col_to_resample].fillna(0)
    df_reindexed[col_to_resample] = df_reindexed[col_to_resample].astype(str)
    df_reindexed[col_to_resample] = df_reindexed[col_to_resample].apply(lambda x: x.replace(',',''))
    df_reindexed[col_to_resample] = df_reindexed[col_to_resample].apply(lambda x: x.replace('$',''))
    df_reindexed[col_to_resample] = df_reindexed[col_to_resample].apply(lambda x: x.replace('£',''))
    df_reindexed[col_to_resample] = df_reindexed[col_to_resample].apply(lambda x: x.replace('€',''))
    df_reindexed[col_to_resample] = pd.to_numeric(df_reindexed[col_to_resample])
    df_reindexed[col_to_resample].replace({0:np.nan}, inplace=True)
    df = df_reindexed.interpolate(method='linear')
    df = df / 7
    df.reset_index(inplace=True)
    df.rename({'index': 'date'}, axis=1, inplace=True)
    
    return df

def interpolate_monthly_data(df, date_col=None, resample_col=None):
    df = df.copy()

    if date_col == None:
        date_col = df.columns[0]
        
    if resample_col == None:
        resample_col = df.columns[1]

    df[date_col] = pd.to_datetime(df[date_col], format="%Y-%m")
    df['start_of_month'] = (df[date_col].dt.floor('d') + pd.offsets.MonthEnd(0) - pd.offsets.MonthBegin(1))
    df['end_of_month'] = pd.to_datetime(df['start_of_month']) + pd.offsets.MonthEnd(1)
    df['days_in_month'] = (df['end_of_month'] - df['start_of_month']).dt.days + 1
    df[resample_col] = df[resample_col] / df['days_in_month']
    reindexed = df.set_index("start_of_month")
    reindexed = reindexed.reindex(pd.date_range(start=reindexed.index.min(),
                                        end=reindexed.end_of_month.max(),
                                        freq='1D'))
    
    resampled = reindexed[resample_col]
    resampled.replace({0:np.nan}, inplace=True)
    resampled = resampled.interpolate(method='linear')
    resampled = resampled.reset_index()
    resampled.rename({'index': 'date'}, axis=1, inplace=True)
    resampled.fillna(0, inplace=True)
    return resampled

def group_weekly(df, date_col:str) -> pd.DataFrame:
    weekly = df.copy()
    weekly['week'] = weekly[date_col].dt.isocalendar().week
    weekly['year'] = weekly[date_col].dt.isocalendar().year
    weekly['year_week'] = weekly['year'].astype(str) + "-" + weekly['week'].astype(str)
    weekly = weekly.groupby('year_week').sum()
    weekly.drop(['week', 'year'], axis=1, inplace=True)
    weekly.reset_index(inplace=True)
    return weekly

def group_monthly(df, date_col:str) -> pd.DataFrame:
    monthly = df.copy()
    monthly['month'] = monthly[date_col].dt.month
    monthly['year'] = monthly[date_col].dt.isocalendar().year
    monthly['year_month'] = monthly['year'].astype(str) + "-" + monthly['month'].astype(str)
    monthly = monthly.groupby('year_month').sum()
    monthly.drop(['month', 'year'], axis=1, inplace=True)
    monthly.reset_index(inplace=True)
    return monthly

def handle_search_trends_data(df:pd.DataFrame) -> pd.DataFrame:
    # delete any '<' signs for low volume days
    for c in df.select_dtypes(include=['object']).columns[1:]:
        df[c] = df[c].str.replace('<', '')
        df[c] = pd.to_numeric(df[c])

    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df.set_index(date_col, inplace=True)
    df_reindexed = df.reindex(pd.date_range(start=df.index.min(),
                                            end=df.index.max() + dt.timedelta(days=6), freq='1D'))
    df = df_reindexed.interpolate(method='linear')
    df = df.round(1)
    df.reset_index(inplace=True)
    df.rename({'index': 'date'}, axis=1, inplace=True)
    return df

def handle_covid_data(data:pd.DataFrame, sub_region_1:str=None) -> pd.DataFrame:
    if sub_region_1 is None:
        df = data[data['sub_region_1'].isnull()]
    else:
        df = data[data['sub_region_1'] == sub_region_1]
        df = df[df['sub_region_2'].isnull()]
    
    df.reset_index(inplace=True)
    return df[df.columns[9:]]

def handle_weather_data(df:pd.DataFrame) -> pd.DataFrame:
    year = df['YEAR'].astype(str)
    month = df['MO'].astype(str)
    day = df['DY'].astype(str)
    
    month = month.apply(lambda x: '0'+x if len(x) == 1 else x)
    day = day.apply(lambda x: '0'+x if len(x) == 1 else x)
    
    df['date'] = pd.to_datetime(year + "-" + month + "-" + day)
    df = df[['date', 'T2M_RANGE', 'T2M_MAX', 'T2M_MIN', 'T2M']]
    
    return df

def create_holiday_dummies(df:pd.DataFrame) -> pd.DataFrame:
    dr = pd.date_range(start=df['date'].min(), end=df['date'].max())
    date_df = pd.DataFrame({'date': dr})
    for _, row in df.iterrows():
        date_df[row[1]] = (date_df['date'] == row[0])
        
    date_df.iloc[:, 1:] = date_df.iloc[:, 1:].astype(int)
    return date_df

def create_date_range_dummies(df:pd.DataFrame) -> pd.DataFrame:
    dr = pd.date_range(start=df['start'].min(), end=df['end'].max())
    
    date_df = pd.DataFrame({'date': dr})

    for _, row in df.iterrows():
        date_df[row[2]] = (date_df['date'] >= row[0]) & (date_df['date'] <= row[1])
        
    date_df.iloc[:, 1:] = date_df.iloc[:, 1:].astype(int)
    return date_df

def add_start_of_month(df:pd.DataFrame, date_col:str) -> pd.DataFrame:
    df['start_of_month'] = (df[date_col].dt.floor('d') + pd.offsets.MonthEnd(0) - pd.offsets.MonthBegin(1))
    return df

def add_end_of_month(df:pd.DataFrame, date_col:str) -> pd.DataFrame:
    df['end_of_month'] = pd.to_datetime(df[date_col]) + pd.offsets.MonthEnd(1)
    return df

def num_days_in_month(df:pd.DataFrame, date_col:str) -> pd.DataFrame:
    start_of_month = (df[date_col].dt.floor('d') + pd.offsets.MonthEnd(0) - pd.offsets.MonthBegin(1))
    end_of_month = pd.to_datetime(df[date_col]) + pd.offsets.MonthEnd(1)
    df['days_in_month'] = (end_of_month - start_of_month).dt.days + 1
    return df