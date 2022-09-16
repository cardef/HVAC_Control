import pandas as pd
import numpy as np

class Preprocessor():

    def __init__(self, imputer):

         self.col_const = []
         self.mean = 0
         self.std = 1
         self.imputer = imputer

    def fit(self, df):
        self.col_const = df.columns[df.nunique() <= 1]
        self.mean = df.drop(self.col_const, axis = 1).mean(axis = 0)
        self.std = df.drop(self.col_const, axis = 1).std(axis = 0)

        for col in list(df.drop(self.col_const, axis = 1).columns):
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            self.lower_lim = Q1 - 1.5*IQR
            self.upper_lim = Q3 + 1.5*IQR

        self.imputer = self.imputer.fit(df.drop(self.col_const, axis = 1))
        return self

    def transform(self, df):
        df = df.drop(self.col_const, axis = 1)
        
        for col in list(df.columns):
            outliers = (df[col] < self.lower_lim) | (df[col] > self.upper_lim)
            df[col] = df[col].where(~outliers, np.nan)

        df = pd.DataFrame(self.imputer.transform(df), columns = df.columns)
        df = (df-self.mean)/self.std

        return df