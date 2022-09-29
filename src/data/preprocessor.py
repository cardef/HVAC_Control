import pandas as pd
import numpy as np

class Preprocessor():

    def __init__(self, col_to_ignore = [], scaling = True, outliers = True, remove_col_const = True):

        self.col_const = []
        self.lower_lim = []
        self.upper_lim = []
        self.mean = 0
        self.std = 1
        self.scaling = scaling
        self.outliers = outliers
        self.remove_col_const = remove_col_const
        self.col_to_ignore = col_to_ignore

    def fit(self, df):
        if self.remove_col_const:
            self.col_const = list(df.columns[df.nunique() <= 1])
        df_cleaned = df.drop(self.col_const+self.col_to_ignore, axis = 1)

        if self.scaling:
            self.mean = df_cleaned.mean(axis = 0)
            self.std = df_cleaned.std(axis = 0)
            self.std.where(self.std == 0, 1, inplace = True)
            df_cleaned = (df_cleaned-self.mean)/self.std
        
        if self.outliers:
            for col in list(df_cleaned.columns):
                Q1 = df_cleaned[col].quantile(0.25)
                Q3 = df_cleaned[col].quantile(0.75)
                IQR = Q3 - Q1
                self.lower_lim.append(Q1 - 1.5*IQR)
                self.upper_lim.append(Q3 + 1.5*IQR)

        
        return self

    def transform(self, df):
        df_cleaned = df.drop(self.col_const + self.col_to_ignore, axis = 1)
        df_cleaned = (df_cleaned-self.mean)/self.std
        
        if self.outliers:
            for i, col in enumerate(list(df_cleaned.columns)):
                outliers = (df_cleaned[col] < self.lower_lim[i]) | (df_cleaned[col] > self.upper_lim[i])
                df_cleaned[col] = df_cleaned[col].where(~outliers, np.nan)
        
        
        
        df_cleaned[self.col_to_ignore] = df[self.col_to_ignore].copy()
        return df_cleaned