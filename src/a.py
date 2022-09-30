from utils import MAIN_DIR, merge, split
from src.data.preprocesser import Preprocessor
from data.dataset import Dataset, collate_fn
from sklearn.impute import KNNImputer, SimpleImputer
from pickle import dump
from pathlib import Path
from torch.utils.data import DataLoader
import json
import torch
import pandas as pd
import numpy as np

'''
df= pd.read_csv(Path(__file__).parent.parent/'data'/'raw'/'doi_10.7941_D1N33Q__v6'/'Building_59'/'Bldg59_clean data'/'ele.csv')
df5= pd.read_csv(Path(__file__).parent.parent/'data'/'raw'/'doi_10.7941_D1N33Q__v6'/'Building_59'/'Bldg59_clean data'/'site_weather.csv')
df6 = merge(Path(__file__).parent.parent/'data'/'raw'/'doi_10.7941_D1N33Q__v6'/'Building_59'/'Bldg59_clean data', ['ele.csv', 'site_weather.csv'])
df2 = pd.read_csv(Path(__file__).parent.parent/'data'/'cleaned'/'energy'/'test_set_imp.csv')
energy_preprocessor = Preprocessor(SimpleImputer(), ['date'])
energy_preprocessor.fit(df)
df3 = energy_preprocessor.transform(df)
#print(np.array(df3['date'], dtype = 'datetime64[ns]'))
energy_train_loader = DataLoader(Dataset(df3, 96, 4, ['hvac_S']), batch_size = 64, collate_fn = collate_fn)
for a,b,c,d in energy_train_loader:
    print(pd.to_datetime(d))
    break

#df6 = merge(Path(__file__).parent.parent/'data'/'raw'/'doi_10.7941_D1N33Q__v6'/'Building_59'/'Bldg59_clean data', ['ele.csv', 'site_weather.csv'])
#print(df6['date'])
temp_train_set=pd.read_csv(Path(__file__).parent.parent/'data'/'cleaned'/'energy'/'valid_set_imp.csv')
#energy_train_loader = DataLoader(Dataset(temp_train_set, 96, 4, ['hvac']), batch_size = 64, collate_fn = collate_fn)
energy_test_loader = torch.load(Path(__file__).parent.parent/'data'/'cleaned'/'energy'/'train_loader.pt')
for a,b,c,d in energy_test_loader:
    print(pd.to_datetime(d))
    break
print(temp_train_set['date'])

df6 = merge(Path(__file__).parent.parent/'data'/'raw'/'doi_10.7941_D1N33Q__v6'/'Building_59'/'Bldg59_clean data', ['ele.csv', 'site_weather.csv'])
energy_train_set, energy_test_set = split(df6)
#print(energy_test_set['date'])
energy_train_set, energy_valid_set = split(df6, train_size = 0.9)
energy_imputer = SimpleImputer()
energy_preprocessor = Preprocessor(energy_imputer, ['date'])
energy_preprocessor.fit(energy_train_set)
#energy_train_set = energy_preprocessor.transform(energy_train_set)
#energy_valid_set = energy_preprocessor.transform(energy_valid_set)
energy_test_set = energy_preprocessor.transform(energy_test_set)
#print(energy_train_set['date'])
'''
df = pd.read_csv(MAIN_DIR/'results'/'outputs'/'temp_prediction.csv', header = [0,1], index_col=[0])
print(df)
print(df['cerc_templogger_1'])
df_melted= df['cerc_templogger_1'].reset_index().melt(id_vars = ['date'], value_vars = ['true', 'pred'])
print(df_melted)