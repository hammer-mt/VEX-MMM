import statsmodels.tsa as tsa
import pandas as pd
from typing import List, Tuple

from .select import get_cols_containing

def add_constant(df:pd.DataFrame) -> None:
    df['constant'] = 1

def add_adstocks(df:pd.DataFrame, column_label:str, adstock_rates:List[float]=None) -> List:
    if adstock_rates is None:
        adstock_rates = [round(i*0.1,1) for i in range(1,10)]

    added_columns = list()
    for ar in adstock_rates:
        ar_column = f"{column_label} AR={ar}"
        df[ar_column] = tsa.filters.filtertools.recursive_filter(df[column_label], ar)
        added_columns.append(ar_column)
    return added_columns

def add_diminishing_returns(df:pd.DataFrame, column_label:str, saturation_levels:List[float]=None) -> List:
    if saturation_levels is None:
        saturation_levels = [round(i*0.1,1) for i in range(1,10)]

    added_columns = list()
    for dr in saturation_levels:
        dr_column = f"{column_label} DR={dr}"
        df[dr_column] = df[column_label]**dr
        added_columns.append(dr_column)
    return added_columns

def add_lags(df:pd.DataFrame, column_label:str, lags:List[float]=None) -> List:
    if lags is None:
        lags = [1, 2, 3, 7, 14, 30, 60, 90, 180, 365]

    added_columns = list()
    for l in lags:
        l_column = f"{column_label} Lag={l}"
        df[l_column] = df[column_label].shift(l)
        df[l_column] = df[l_column].fillna(0)
        added_columns.append(l_column)
    return added_columns

def add_interaction_effect(df:pd.DataFrame, column_label_a:str, column_label_b:str) -> str:
    interaction_name = f'{column_label_a} x {column_label_b}'

    df[interaction_name] = df[column_label_a] * df[column_label_b]
    return interaction_name

def add_day_of_week_dummies(df:pd.DataFrame, date_label:str=None) -> Tuple[List[str], pd.DataFrame]:
    if date_label is None:
        dates_index = pd.to_datetime(df.index)
        date_label = '_date'
        df[date_label] = dates_index
        
    else:
        dates_index = pd.to_datetime(df[date_label])

    df['day_of_week'] = dates_index.dt.day_name()
    df['day_of_week'] = df['day_of_week'].str.lower()
    dummies = pd.get_dummies(df['day_of_week'])
        
    dummies[date_label] = dates_index
    
    df = pd.merge(df, dummies, left_on=date_label, right_on=date_label, how='left')
    
    df.drop(['day_of_week'], axis=1, inplace=True)
    
    df.drop(['_date'], axis=1, inplace=True) # in case we added it
    dummies.drop([date_label], axis=1, inplace=True)
    
    return list(dummies.columns), df

def add_month_of_year_dummies(df:pd.DataFrame, date_label:str=None) -> Tuple[List[str], pd.DataFrame]:
    if date_label is None:
        dates_index = pd.to_datetime(df.index)
        date_label = '_date'
        df[date_label] = dates_index
        
    else:
        dates_index = pd.to_datetime(df[date_label])

    df['month_of_year'] = dates_index.dt.month_name()
    df['month_of_year'] = df['month_of_year'].str.lower()
    
    dummies = pd.get_dummies(df['month_of_year'])
        
    dummies[date_label] = df[date_label]
    
    df = pd.merge(df, dummies, left_on=date_label, right_on=date_label, how='left')
    
    df.drop(['month_of_year'], axis=1, inplace=True)

    df.drop(['_date'], axis=1, inplace=True) # in case we added it
    dummies.drop([date_label], axis=1, inplace=True)
    
    return list(dummies.columns), df

def add_payday_dummies(df:pd.DataFrame, date_label:str) -> Tuple[str, pd.DataFrame]:
    payday_column = 'payday'
    df[payday_column] = df[date_label].apply(lambda x:1 if x.strftime('%d') in ('14','15','16','30','31','1','2') else 0)

    return payday_column, df

def categorize_campaigns(df:pd.DataFrame, containing:str) -> Tuple[str, pd.DataFrame]: 
    containing_cols = get_cols_containing(df, containing)

    agg_label = df[f'"{containing}" Agg']
    df[agg_label] = df[containing_cols].sum()

    return agg_label, df
