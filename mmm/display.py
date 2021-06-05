
# show the charts and tables
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from typing import List, Tuple, Union
import os

from pandas.core.arrays import boolean

def display_accuracy_chart(y_actual:Union[np.array, List[float], pd.Series], 
                           y_pred:Union[np.array, List[float], pd.Series], 
                           y_label:str=None, 
                           accuracy:Tuple[str, str]=None) -> None:
    # set up figure and subplots
    fig, ax = plt.subplots(figsize=(14,8), nrows=2, ncols=1, gridspec_kw={'height_ratios': [3, 1]})
    
    # create plot df
    plot_df = pd.DataFrame()
    plot_df['Actual'] = y_actual
    plot_df['Predicted'] = y_pred
    plot_df['Error'] = (y_pred - y_actual) / y_actual * 100
    
    # plot actual vs predicted on grid
    if y_label:
        plot_df[['Actual', 'Predicted']].plot(ax=ax[0], ylabel=y_label)
    else:
        plot_df[['Actual', 'Predicted']].plot(ax=ax[0])

    if accuracy:
        ax[0].annotate(f'{accuracy[0]} = {accuracy[1]}', xy=(0.05, 0.92), xycoords='axes fraction')

    ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=2)
    ax[0].grid(True, which='both')
    
    # plot error on grid
    plot_df[['Error']].plot(ax=ax[1], color='red')
    ax[1].grid(True, which='both')
    ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.35), ncol=2)
    fmt = '%.0f%%' # Format you want the ticks, e.g. '40%'
    yticks = mtick.FormatStrFormatter(fmt)
    ax[1].yaxis.set_major_formatter(yticks)
    
    # show plots
    fig.autofmt_xdate(rotation=45)
    plt.gcf().suptitle("Actual vs Predicted", fontsize=20)
    
    plt.show()

def display_contrib_chart(pred_df:pd.DataFrame) -> None:
    f = plt.figure(figsize=(20,10))
    plt.title('Contribution Chart', color='black', fontsize=20, y=1.03)
    (pred_df.sum() / pred_df.sum().sum()).plot(kind='barh')
    plt.show()

def display_decomp_chart(pred_df:pd.DataFrame) -> None:
    plot_df = pred_df.copy()
    
    # separate positive & negative values
    plot_df = plot_df[plot_df>=0].join(plot_df[plot_df<0], lsuffix="", rsuffix="_$negative")
    plot_df.rename(columns=lambda x: '-'+x.replace('_$negative','') if '_$negative' in x else x, inplace=True)
    plot_df.dropna(axis=1, how='all', inplace=True)
    
    # plot area chart
    f = plt.figure()
    plt.title('Decomposition Chart', color='black', fontsize=20, y=1.03)
    plot_df.plot(kind='area', figsize=(20,10), ax=f.gca())
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))

    plt.show()

def save_model(y_label:str, 
               error:Union[np.array, List[float], pd.Series], 
               X_labels:List[str], 
               coefficients:Union[np.array, List[float], pd.Series], 
               algo:str="LinearRegression", file_loc:str=None) -> None:
    
    if file_loc is None:
        file_loc = "../results/models.csv"

    timestamp = dt.datetime.today().strftime('%Y-%m-%d %H:%M')

    data = {
        'timestamp': timestamp,
        'y_label': y_label,
        'error': error,
        'X_labels': X_labels,
        'coefficients': [round(c, 3) for c in coefficients],
        'algo': algo,
    }
    df = pd.DataFrame([data])

    if os.path.isfile(file_loc):
        models = pd.read_csv(file_loc)
        df = models.append(df)
    
    model_no = df.shape[0]
    df.to_csv(file_loc, index=False)
    print(f"Model {model_no} saved to {file_loc}")

def load_models(file_loc:str=None) -> pd.DataFrame:
    if file_loc is None:
        file_loc = "../results/models.csv"

    if os.path.isfile(file_loc):
        models = pd.read_csv(file_loc)
        return models
    else:
        print(f"No models saved at {file_loc}")
        return pd.DataFrame(columns=['timestamp', 'y_label', 'X_labels', 'algo', 'error'])

def clear_models(file_loc:str=None) -> None:
    if file_loc is None:
        file_loc = "../results/models.csv"

    if os.path.isfile(file_loc):
        os.remove(file_loc)

        print(f"Models cleared from {file_loc}")

    else:
        print(f"No models saved at {file_loc}")