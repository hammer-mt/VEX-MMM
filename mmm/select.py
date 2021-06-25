# feature selection

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
from sklearn.feature_selection import f_regression
np.seterr(divide='ignore', invalid='ignore') # hide error warning for vif
from sklearn.feature_selection import f_regression, RFE
from sklearn.linear_model import LinearRegression
from typing import List, Tuple, Union

from .build import run_regression
from .validate import calculate_nrmse

def guess_date_column(df:pd.DataFrame) -> None:
    guesses = ['date', 'Date', 'day', 'Day', 'week', 'Week', 'Month', 'month']
    for x in guesses:
        if x in columns:
            return x
    return None

def guess_y_column(df):
    guesses = ['revenue', 'sales', 'conversions', 'purchases']
    columns = [x.lower() for x in df.columns]
    for x in guesses:
        if x in columns:
            return x
    return None

def guess_media_columns(df):
    guesses = ['cost', 'spend', 'impression', 'spent', 'clicks']
    columns = [x.lower() for x in df.columns]
    media_columns = []
    for x in guesses:
        for y in columns:
            if x in y:
                media_columns.append(y)

    return media_columns

def add_X_labels(X_labels, add_cols):
    for x in add_cols:
        if x not in X_labels:
            X_labels.append(x)

    return X_labels

def del_X_labels(X_labels:List[str], del_cols:List[str]) -> List[str]:
    for x in del_cols:
        if x in X_labels:
            X_labels.remove(x)

    return X_labels

def get_all_X_labels(df:pd.DataFrame, y_label:str, date_label:str=None) -> List[str]:
    X_labels = list(df.columns)
    X_labels.remove(y_label)
    if date_label:
        X_labels.remove(date_label)
    
    return X_labels

def get_cols_containing(df:pd.DataFrame, containing:str) -> List[str]:
    return [x for x in list(df.columns) if containing in x]

def y_variable_correlation(df:pd.DataFrame, y_label:str, X_labels:List[str], min_corr:float=0.3) -> Tuple[List, pd.DataFrame]:
    # 1.0 = Perfect
    # 0.7 = Strong
    # 0.5 = Moderate
    # 0.3 = Weak
    # 0 = None
    
    all_variables = X_labels.copy()
    all_variables.extend([y_label])

    corr = df[all_variables].corr()
    corr_df = pd.DataFrame({'corr':abs(corr[y_label].drop(y_label))})
    corr_df['corr_keep'] = corr_df['corr'] > min_corr
    
    corr_keep = list(corr_df[corr_df['corr_keep']==True].index.values)
    
    return corr_keep, corr_df

def variance_inflation_factor(df:pd.DataFrame, X_labels:List[str], max_vif:int=5) -> Tuple[List, pd.DataFrame]:
    # Variance Inflation Factor (VIF)
    # tests for colinearity: A VIF of over 10 for some feature indicates that over 90% 
    # of the variance in that feature is explained by the remaining features. Over 100 
    # indicates over 99%. Best practice is to keep variables with a VIF less than 5.

    X = df[X_labels]
    X_np = np.array(X)

    vif_results = [(X.columns[i], vif(X_np, i)) for i in range(X_np.shape[1])]
    vif_df = pd.DataFrame(vif_results)
    vif_df.columns = ['idx', 'vif']
    vif_df.index = vif_df['idx']
    vif_df.drop(['idx'], axis=1, inplace=True)
    vif_df.index.name = None
    vif_df['vif_keep'] = vif_df['vif'] < max_vif
    
    vif_keep = list(vif_df[vif_df['vif_keep']==True].index.values)

    return vif_keep, vif_df
    
def backwards_feature_elimination(df:pd.DataFrame, y_label:str, 
                                  X_labels:List[str], max_p:float=0.05) -> Tuple[List, pd.DataFrame]:
    # backwards feature elimination
    X_labels_left = X_labels.copy()
    while (len(X_labels_left)>0):
        p_values = run_regression(df, y_label, X_labels_left)[3]
        p = pd.Series(p_values,index=X_labels_left)      
        pmax = max(p)
        feature_with_p_max = p.idxmax()
        if (pmax > max_p):
            X_labels_left.remove(feature_with_p_max)
        else:
            break
    
    bfe = pd.Series(X_labels).apply(lambda x: x in X_labels_left)
    bfe.index = X_labels
    bfe_df = pd.DataFrame({'bfe_keep': bfe})
    bfe_keep = list(bfe_df[bfe_df['bfe_keep']==True].index.values)

    return bfe_keep, bfe_df

def recursive_feature_elimination(df:pd.DataFrame, y_label:str, X_labels:List[str], max_features:List[float]=None) -> Tuple[pd.Series, pd.DataFrame]:
    if max_features is None:
        max_features = max(round(len(X_labels)/5),1)

    rfe = RFE(LinearRegression(), n_features_to_select=max_features).fit(df[X_labels], df[y_label])
    rfe_keep = pd.Series(rfe.support_)
    rfe_keep.index = X_labels
    
    rfe_df = pd.DataFrame({'rfe_keep': rfe_keep})
    rfe_df['rfe_ranking'] = rfe.ranking_
    return rfe_keep, rfe_df

def find_best_feature(df, y_label, X_candidates, X_labels=None):
    models = dict()
    if X_labels is None:
        X_labels = []

    for x in X_candidates:
        y_actual, y_pred, _ = run_regression(df, y_label, [x] + X_labels)

        # test accuracy
        nrmse = calculate_nrmse(y_actual, y_pred)
        
        models[x] = nrmse

    models_df = pd.DataFrame(models.items(), columns=['candidate', 'nrmse'])
    models_df.index = models_df['candidate']
    models_df.drop(columns=['candidate'], inplace=True)
    min_label = models_df[['nrmse']].idxmin()[0]
    return min_label

# TODO: eliminate negative coefficients
