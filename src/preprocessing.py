from email.errors import MissingHeaderBodySeparatorDefect
from utils import MAIN_DIR, merge, split
from data.preprocessor import Preprocessor
from data.dataset import Dataset, collate_fn
from sklearn.impute import KNNImputer, SimpleImputer, IterativeImputer
from pickle import dump
from pathlib import Path
from torch.utils.data import DataLoader
import json
import torch
#from fancyimpute import MatrixFactorization
from sklearn.experimental import enable_iterative_imputer
import sys
import sklearn.neighbors._base
from data.imputer import Imputer
import missingno as msno
import pandas as pd
import numpy as np
from statistics import mean

with open('config.json') as f:
    config = json.load(f)
time_window = config['PARAMETER']['DATA']['time_window']
len_forecast = config['PARAMETER']['DATA']['len_forecast']
len_forecast = config['PARAMETER']['DATA']['len_forecast'] 
col_temp_in = config['PARAMETER']['DATA']['col_temp_in']  
col_temp_ext = config['PARAMETER']['DATA']['col_temp_ext']

en = ['ele.csv']
outdoor = ['site_weather.csv']
hvac_op = ['hp_hws_temp.csv', 'rtu_sa_t_sp.csv', 'rtu_sa_t.csv', 'rtu_ra_t.csv', 'rtu_ma_t.csv','rtu_oa_t.csv', 'rtu_sa_fr.csv', 'rtu_oa_damper.csv', 'rtu_econ_sp.csv', 'rtu_sa_p_sp.csv', 'rtu_plenum_p.csv', 'rtu_fan_spd.csv', 'uft_fan_spd.csv', 'uft_hw_valve.csv']
indoor = ['zone_temp_sp_c.csv', 'zone_temp_sp_h.csv', 'zone_temp_interior.csv', 'zone_temp_exterior.csv']
main_dir = Path(__file__).parent.parent
path_raw = r'D:\My Drive\Uni\Vicomtech\Tesi\HVAC_Control\dataset\doi_10.7941_D1N33Q__v6\Building_59\Bldg59_clean data'
path_cleaned = r'D:\My Drive\Uni\Vicomtech\Tesi\HVAC_Control\dataset'

try:
    energy_df = pd.read_csv(main_dir/'data'/'lbnlbldg59'/'lbnlbldg59.processed'/'LBNLBLDG59'/'clean_Bldg59_2018to2020'/'clean data'/'energy.csv') 
except:
    energy_df = merge(main_dir/'data'/'lbnlbldg59'/'lbnlbldg59.processed'/'LBNLBLDG59'/'clean_Bldg59_2018to2020'/'clean data', en+outdoor+hvac_op)
    energy_df['hvac'] = energy_df['hvac_S']+energy_df['hvac_N']
    energy_df = energy_df.drop(['hvac_S', 'hvac_N', 'mels_S', 'lig_S', 'mels_N'], axis = 1)
    energy_df.to_csv(main_dir/'data'/'lbnlbldg59'/'lbnlbldg59.processed'/'LBNLBLDG59'/'clean_Bldg59_2018to2020'/'clean data'/'energy.csv', index=False)

energy_preprocessor = Preprocessor(['date'], scaling = False)
energy_preprocessor.fit(energy_df)
energy_df = energy_preprocessor.transform(energy_df)
imputer = Imputer(energy_df, 'date', 500, 100)

energy_df = imputer.impute()
print(energy_df.isnull().sum().sum())



energy_full_train_set, energy_test_set = split(energy_df)
energy_train_set, energy_valid_set = split(energy_full_train_set, train_size = 0.9)
energy_preprocessor = Preprocessor(['date'], outliers=False)
energy_preprocessor.fit(energy_train_set)
energy_train_set = energy_preprocessor.transform(energy_train_set)
energy_valid_set = energy_preprocessor.transform(energy_valid_set)


energy_preprocessor.fit(energy_full_train_set)
energy_full_train_set=energy_preprocessor.transform(energy_full_train_set)
energy_test_set = energy_preprocessor.transform(energy_test_set)
print('Prep completed')

energy_train_set.to_csv(main_dir/'data'/'cleaned'/'energy'/'train_set_imp.csv', index = False)
energy_valid_set.to_csv(main_dir/'data'/'cleaned'/'energy'/'valid_set_imp.csv', index = False)
energy_test_set.to_csv(main_dir/'data'/'cleaned'/'energy'/'test_set_imp.csv', index = False)
energy_full_train_set.to_csv(main_dir/'data'/'cleaned'/'energy'/'full_train_set_imp.csv', index = False)
dump(energy_preprocessor, open(main_dir/'data'/'cleaned'/'preprocessor'/'energy_preprocessor.pkl', 'wb'))


try:
    temp_df = pd.read_csv(main_dir/'data'/'lbnlbldg59'/'lbnlbldg59.processed'/'LBNLBLDG59'/'clean_Bldg59_2018to2020'/'clean data'/'temp.csv') 

except:
    temp_df = merge(main_dir/'data'/'lbnlbldg59'/'lbnlbldg59.processed'/'LBNLBLDG59'/'clean_Bldg59_2018to2020'/'clean data', outdoor+hvac_op+indoor)
    temp_df.to_csv(main_dir/'data'/'lbnlbldg59'/'lbnlbldg59.processed'/'LBNLBLDG59'/'clean_Bldg59_2018to2020'/'clean data'/'temp.csv', index=False)


temp_preprocessor = Preprocessor(['date'], scaling = False)
temp_preprocessor.fit(temp_df)
temp_df = temp_preprocessor.transform(temp_df)
imputer = Imputer(temp_df, 'date', 500, 100)

temp_df = imputer.impute()
print(temp_df.isnull().sum().sum())

temp_full_train_set, temp_test_set = split(temp_df)
temp_train_set, temp_valid_set = split(temp_full_train_set, train_size = 0.9)
temp_preprocessor = Preprocessor(['date'], outliers=False)
temp_preprocessor.fit(temp_train_set)
temp_train_set = temp_preprocessor.transform(temp_train_set)
temp_valid_set = temp_preprocessor.transform(temp_valid_set)

temp_preprocessor.fit(temp_full_train_set)
temp_full_train_set=temp_preprocessor.transform(temp_full_train_set)
temp_test_set = temp_preprocessor.transform(temp_test_set)

temp_train_set.to_csv(main_dir/'data'/'cleaned'/'temp'/'train_set_imp.csv', index = False)
temp_valid_set.to_csv(main_dir/'data'/'cleaned'/'temp'/'valid_set_imp.csv', index = False)
temp_test_set.to_csv(main_dir/'data'/'cleaned'/'temp'/'test_set_imp.csv', index = False)
temp_full_train_set.to_csv(main_dir/'data'/'cleaned'/'temp'/'full_train_set_imp.csv', index = False)
dump(temp_preprocessor, open(main_dir/'data'/'cleaned'/'preprocessor'/'temp_preprocessor.pkl', 'wb'))