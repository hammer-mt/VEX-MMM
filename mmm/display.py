# show the charts and tables

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from IPython.display import display
import pandas as pd
pd.set_option('display.float_format', lambda x: '%.3f' % x) # suppress scientific notation

def display_accuracy_charts(y_actual, y_pred, dates_index):
    # set up figure and subplots
    fig, ax = plt.subplots(figsize=(14,8), nrows=2, ncols=1, gridspec_kw={'height_ratios': [3, 1]})
    
    # create plot df
    plot_df = pd.DataFrame()
    plot_df['Actual'] = y_actual
    plot_df['Predicted'] = y_pred
    plot_df['Error'] = (y_pred - y_actual) / y_actual * 100
    plot_df.index = dates_index
    
    # plot actual vs predicted on grid
    plot_df[['Actual', 'Predicted']].plot(ax=ax[0])
    ax[0].grid(True, which='both')
    
    # plot error on grid
    plot_df[['Error']].plot(ax=ax[1], color='red')
    ax[1].grid(True, which='both')
    fmt = '%.0f%%' # Format you want the ticks, e.g. '40%'
    yticks = mtick.FormatStrFormatter(fmt)
    ax[1].yaxis.set_major_formatter(yticks)
    
    # show plots
    fig.autofmt_xdate(rotation=45)
    plt.show()

def display_contrib_chart(pred_df):
    f = plt.figure(figsize=(20,10))
    plt.title('Contribution Chart', color='black')
    (pred_df.sum() / pred_df.sum().sum()).plot(kind='barh')
    plt.show()

def display_decomp_chart(pred_df):
    plot_df = pred_df.copy()
    
    # separate positive & negative values
    plot_df = plot_df[plot_df>=0].join(plot_df[plot_df<0], lsuffix="", rsuffix="_$negative")
    plot_df.rename(columns=lambda x: '-'+x.replace('_$negative','') if '_$negative' in x else x, inplace=True)
    plot_df.dropna(axis=1, how='all', inplace=True)
    
    # plot area chart
    f = plt.figure()
    plt.title('Decomposition Chart', color='black')
    plot_df.plot(kind='area', figsize=(20,10), ax=f.gca())
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))

    plt.show()