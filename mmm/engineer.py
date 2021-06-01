import statsmodels.tsa as tsa
import pandas as pd

from .helpers import get_cols_containing

def add_constant(df):
    df['constant'] = 1
    return df

def add_adstocks(df, column_label, adstock_rates=None):
    if adstock_rates is None:
        adstock_rates = [round(i*0.1,1) for i in range(1,10)]

    added_columns = list()
    for ar in adstock_rates:
        ar_column = f"{column_label} AR={ar}"
        df[ar_column] = tsa.filters.filtertools.recursive_filter(df[column_label], ar)
        added_columns.append(ar_column)
    return added_columns

def add_diminishing_returns(df, column_label, saturation_levels=None):
    if saturation_levels is None:
        saturation_levels = [round(i*0.1,1) for i in range(1,10)]

    added_columns = list()
    for dr in saturation_levels:
        dr_column = f"{column_label} DR={dr}"
        df[dr_column] = df[column_label]**dr
        added_columns.append(dr_column)
    return added_columns

def add_lags(df, column_label, lags=None):
    if lags is None:
        lags = [1, 2, 3, 7, 14, 30, 60, 90, 180, 365]

    added_columns = list()
    for l in lags:
        l_column = f"{column_label} Lag={l}"
        df[l_column] = df[column_label].shift(l)
        df[l_column] = df[l_column].fillna(0)
        added_columns.append(l_column)
    return added_columns

def add_interaction_effect(df, column_label_a, column_label_b):
    interaction_name = f'{column_label_a} x {column_label_b}'

    df[interaction_name] = df[column_label_a] * df[column_label_b]
    return interaction_name

def add_day_of_week_dummies(df, date_label):
    df['day_of_week'] = df[date_label].dt.day_name()
    dummies = pd.get_dummies(df['day_of_week'])
        
    dummies[date_label] = df[date_label]
    
    df = pd.merge(df, dummies, left_on=date_label, right_on=date_label, how='left')
    
    df.drop(['day_of_week'], axis=1, inplace=True)
    dummies.drop([date_label], axis=1, inplace=True)
    
    return list(dummies.columns)

def add_month_of_year_dummies(df, date_label):
    df['month_of_year'] = df[date_label].dt.month_name()
    
    dummies = pd.get_dummies(df['month_of_year'])
        
    dummies[date_label] = df[date_label]
    
    df = pd.merge(df, dummies, left_on=date_label, right_on=date_label, how='left')
    
    df.drop(['month_of_year'], axis=1, inplace=True)
    dummies.drop([date_label], axis=1, inplace=True)
    
    return list(dummies.columns)

def add_payday_dummies(df, date_label):
    payday_column = 'payday'
    df[payday_column] = df[date_label].apply(lambda x:1 if x.strftime('%d') in ('14','15','16','30','31','1','2') else 0)

    return payday_column

def categorize_campaigns(df, containing):
    containing_cols = get_cols_containing(df, containing)

    agg_label = df[f'"{containing}" Agg']
    df[agg_label] = df[containing_cols].sum()

    return agg_label
