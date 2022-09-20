import pandas as pd
import numpy as np

class Preprocessor():

    def __init__(self, imputer, col_to_ignore):

        self.col_const = []
        self.lower_lim = []
        self.upper_lim = []
        self.mean = 0
        self.std = 1
        self.imputer = imputer
        self.col_to_ignore = col_to_ignore

    def fit(self, df):
        self.col_const = list(df.columns[df.nunique() <= 1])
        df_cleaned = df.drop(self.col_const+self.col_to_ignore, axis = 1)
        self.mean = df_cleaned.mean(axis = 0)
        self.std = df_cleaned.std(axis = 0)

        for col in list(df_cleaned.columns):
            Q1 = df_cleaned[col].quantile(0.25)
            Q3 = df_cleaned[col].quantile(0.75)
            IQR = Q3 - Q1
            self.lower_lim.append(Q1 - 1.5*IQR)
            self.upper_lim.append(Q3 + 1.5*IQR)

        self.imputer = self.imputer.fit(df_cleaned)
        return self

    def transform(self, df):
        df_cleaned = df.drop(self.col_const + self.col_to_ignore, axis = 1)
        
        for i, col in enumerate(list(df_cleaned.columns)):
            outliers = (df_cleaned[col] < self.lower_lim[i]) | (df_cleaned[col] > self.upper_lim[i])
            df_cleaned[col] = df_cleaned[col].where(~outliers, np.nan)

        df_cleaned = pd.DataFrame(self.imputer.transform(df_cleaned), columns = df_cleaned.columns)
        df_cleaned = (df_cleaned-self.mean)/self.std
        df_cleaned[self.col_to_ignore] = df[self.col_to_ignore]
        return df_cleaned