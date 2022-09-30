from operator import truediv
import pandas as pd
from data.imputer import Imputer
from data.preprocesser import Preprocesser
import json
import torch

from utils import MAIN_DIR


class Merger():
    def __init__(self, path, csv_list, raw_csv_json, cleaned_csv_json, key):
        self.path = path
        self.csv_list = []
        self.preprocesser = Preprocesser(['date'], scaling=False, outliers=True)
        self.key = key
        self.cleaned_csv_json = cleaned_csv_json

        with open(raw_csv_json, 'rb') as f:
            self.raw_csv_json = json.load(f)

        for csv in csv_list:
            self.csv_list.extend(self.raw_csv_json[csv])

    def merge(self):
        cleaned_csv = {}
        features = []
        col_const = []
        df = pd.read_csv(self.path/self.csv_list[0])
        df = df.drop_duplicates(subset=['date'], keep='first', ignore_index=True)
        df['date'] = pd.to_datetime(df['date'])
        
        df['date'] = pd.Series(pd.date_range(
            min(df['date']), max(df['date']), freq='15min'))

        self.preprocesser.fit(df)
        df = self.preprocesser.transform(df)
        print(self.preprocesser.col_const)
        col_const.extend(self.preprocesser.col_const)
        imputer = Imputer(df, ['date'], 100, 50, 1e-1, 1e-8)
        df = imputer.impute()
        features.extend(list(df.drop('date', axis = 1).columns))

        for dataframe_name in self.csv_list[1:]:
            print(dataframe_name)
            dataframe = pd.read_csv(self.path/dataframe_name)
            dataframe = dataframe.drop_duplicates(
                subset=['date'], keep='first', ignore_index = True)
            dataframe['date'] = pd.to_datetime(dataframe['date'], errors='raise')
            self.preprocesser.fit(dataframe)
            dataframe = self.preprocesser.transform(dataframe)

            col_const.extend(self.preprocesser.col_const)

         
            imputer = Imputer(dataframe, ['date'], 100, 50, 1e-1, 1e-8)
            dataframe = imputer.impute()
           
            max_date = max(dataframe['date'])
            min_date = min(dataframe['date'])
            df = df[(df['date'] <= max_date) & (df['date'] >= min_date)]
            
            df = pd.merge_asof(df, dataframe, on='date',
                               tolerance=pd.Timedelta('15min'))
            df.drop(df.filter(regex="Unnamed"), axis=1, inplace=True)
            features.extend(list(dataframe.drop('date', axis = 1).columns))
            torch.cuda.empty_cache()
        
        with open(self.cleaned_csv_json, 'r') as f:
            cleaned_csv = json.load(f)


        cleaned_csv[self.key] = {}
        cleaned_csv[self.key]['features'] = features
        cleaned_csv[self.key]['col_const'] = col_const

        with open(self.cleaned_csv_json, 'w') as f:
            json.dump(cleaned_csv, f, indent=2)

        torch.cuda.empty_cache()

        return df
