# self driving econometric modeling

import pandas as pd
import datetime as dt
import statsmodels.api as sm
import os

from .validate import calculate_nrmse

class Model():
    def __init__(self):
        self.data = None
        self.conv_column = None
        self.date_column = None
        self.base_columns = None
        self.media_columns = None
        self.last_result = None
        self.results = None
        self.project_name = None
        self.X_columns = None
        self.error = None

    def set_project_name(self, project_name):
        self.project = project_name
    
    def load_data(self, file_name):
        df = pd.read_csv(f"data/{self.project_name}/{file_name}")
        self.data = df

    def set_conv_column(self, conv_column):
        self.conv_column = conv_column

    def set_date_column(self, date_column=None):
        if date_column:
            self.date_column = date_column
        else:
            self._guess_date_column()
    
    def _guess_date_column(self):
        guesses = ['date', 'Date', 'day', 'Day']
        for x in guesses:
            if x in self.data.columns:
                self.date_column = x

    def set_base_columns(self, base_columns):
        self.base_columns = base_columns

    def set_media_columns(self, base_columns):
        self.media_columns = base_columns

    def run_regression(self, add_constant=True):
        y = self.data[self.conv_column]

        if add_constant:
            if 'constant' not in self.base_columns:
                self.data['constant'] = 1
                self.base_columns.append('constant')

        X_columns = self.base_columns + self.media_columns
        self.X_columns = X_columns
        X = self.data[X_columns]

        results = sm.OLS(y, X).fit()

        self.last_result = results

        self._test_error()

        self._save_results()

    def _test_error(self):
        y_actual = self.data[self.conv_column]
        X = self.data[self.X_columns]
        y_pred = self.last_result.predict(X)
        error = nrmse(y_actual, y_pred)
        self.error = error

    def _save_results(self):
        timestamp = dt.datetime.now()

        model_data = {
            'timestamp': timestamp,
            'company': self.company_name,
            'project': self.project_name,
            'conv_column': self.conv_column,
            'base_columns': self.base_columns,
            'media_columns': self.media_columns,
            'error': self.error,
        }
        df = pd.DataFrame([model_data])

        if self.results:
            self.results.append(df)
        else:
            self.results = df

    def export_models(self):
        models_file_name = f"results/{self.project_name}/models.csv"

        if os.path.isfile(models_file_name):
            db = pd.read_csv(models_file_name)
            df = db.append(self.results)
            df.to_csv(models_file_name)

    

