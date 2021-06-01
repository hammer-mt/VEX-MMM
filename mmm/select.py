# feature selection

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
from sklearn.feature_selection import f_regression, RFE
from sklearn.linear_model import LinearRegression

from .build import run_regression


def y_variable_correlation(df, y_label, X_labels, min_corr=0.3):
    # 1.0 = Perfect
    # 0.7 = Strong
    # 0.5 = Moderate
    # 0.3 = Weak
    # 0 = None

    corr = df[X_labels]
    corr_dep = abs(corr[y_label].drop(y_label))
    corr_dep['corr_keep'] = corr_dep['corr'] > min_corr
    return corr_dep

def univariate_feature_selection(df, y_label, X_labels, max_p=0.05):
    # univariate feature selection
    f_select = f_regression(df[X_labels], df[y_label], center=True)
    uni_df = pd.DataFrame(f_select[0], index=X_labels)
    uni_keep = uni_df['uni_p'] < max_p
    uni_df['uni_keep'] = uni_keep
    return uni_df

def variance_inflation_factor(df, X_labels, max_vif=5):
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

    return vif_df
    
def backwards_feature_elimination(df, y_label, X_labels, max_p=0.05):
    # backwards feature elimination
    X_labels_left = X_labels.copy()
    while (len(X_labels_left)>0):
        p_values = run_regression(df, y_label, X_labels_left)[3]
        p = pd.Series(p_values[1:],index=X_labels_left)      
        pmax = max(p)
        feature_with_p_max = p.idxmax()
        if (pmax > max_p):
            X_labels_left.remove(feature_with_p_max)
        else:
            break
    
    bfe_keep = pd.Series(X_labels).apply(lambda x: x in X_labels_left)
    bfe_df = pd.DataFrame(X_labels, bfe_keep)

    return bfe_df

def recursive_feature_elimination(df, y_label, X_labels, max_features=None):
    if max_features is None:
        max_features = round(len(X_labels)/5)

    rfe = RFE(LinearRegression(), max_features).fit(df[X_labels], df[y_label])
    rfe_ranking = rfe.ranking_
    rfe_keep = rfe.support_
    rfe_df = pd.DataFrame(X_labels, rfe_keep)
    rfe_df['rfe_ranking'] = rfe_ranking
    return rfe_df