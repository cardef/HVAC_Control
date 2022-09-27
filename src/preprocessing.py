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
from fancyimpute import MatrixFactorization
from data.missforest import MissForest
from sklearn.experimental import enable_iterative_imputer
import sys
import sklearn.neighbors._base
import missingno as msno
import pandas as pd

sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base

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
    energy_df = pd.read_csv(main_dir/'data'/'raw'/'doi_10.7941_D1N33Q__v6'/'Building_59'/'Bldg59_clean data'/'energy.csv') 
except:
    energy_df = merge(main_dir/'data'/'raw'/'doi_10.7941_D1N33Q__v6'/'Building_59'/'Bldg59_clean data', en+outdoor+hvac_op)
    print('Merge completed')
    energy_df['hvac'] = energy_df['hvac_S']+energy_df['hvac_N']
    energy_df = energy_df.drop(['hvac_S', 'hvac_N', 'mels_S', 'lig_S', 'mels_N'], axis = 1)
    energy_df.to_csv(main_dir/'data'/'raw'/'doi_10.7941_D1N33Q__v6'/'Building_59'/'Bldg59_clean data'/'energy.csv', index=False)

#energy_df.loc[:, energy_df.columns != 'date'] = energy_df.drop('date', axis = 1).interpolate(method = 'polynomial', order = 5, axis = 0, limit=4)
#energy_df.loc[:, energy_df.columns != 'date'] = MatrixFactorization().fit_transform(energy_df.loc[:, energy_df.columns != 'date'])
print(energy_df.isnull().sum().sum())

energy_full_train_set, energy_test_set = split(energy_df[energy_df['date'] >= '2019-04-01'])
energy_train_set, energy_valid_set = split(energy_full_train_set, train_size = 0.9)
energy_imputer = IterativeImputer(n_nearest_features=100, skip_complete=True, verbose = 2, max_iter = 50)
energy_preprocessor = Preprocessor(energy_imputer, ['date'])
energy_preprocessor.fit(energy_train_set)
energy_train_set = energy_preprocessor.transform(energy_train_set)
energy_valid_set = energy_preprocessor.transform(energy_valid_set)
energy_test_set = energy_preprocessor.transform(energy_test_set)

energy_preprocessor.fit(energy_full_train_set)
energy_full_train_set=energy_preprocessor.transform(energy_full_train_set)

print('Prep completed')

energy_train_set.to_csv(main_dir/'data'/'cleaned'/'energy'/'train_set_imp.csv', index = False)
energy_valid_set.to_csv(main_dir/'data'/'cleaned'/'energy'/'valid_set_imp.csv', index = False)
energy_test_set.to_csv(main_dir/'data'/'cleaned'/'energy'/'test_set_imp.csv', index = False)
energy_full_train_set.to_csv(main_dir/'data'/'cleaned'/'energy'/'full_train_set_imp.csv', index = False)
dump(energy_preprocessor, open(main_dir/'data'/'cleaned'/'preprocessor'/'energy_preprocessor.pkl', 'wb'))


try:
    temp_df = pd.read_csv(main_dir/'data'/'raw'/'doi_10.7941_D1N33Q__v6'/'Building_59'/'Bldg59_clean data'/'temp.csv') 

except:
    temp_df = merge(main_dir/'data'/'raw'/'doi_10.7941_D1N33Q__v6'/'Building_59'/'Bldg59_clean data', outdoor+hvac_op+indoor)
    temp_df.to_csv(main_dir/'data'/'raw'/'doi_10.7941_D1N33Q__v6'/'Building_59'/'Bldg59_clean data'/'temp.csv', index=False)

print('Merge completed')

#temp_df.loc[:, temp_df.columns != 'date'] = temp_df.drop('date', axis = 1).interpolate(method = 'polynomial', order = 5, axis = 0, limit=4)
print('Imputing completed')
#temp_df.loc[:, temp_df.columns != 'date'] = MatrixFactorization().fit_transform(temp_df.loc[:, temp_df.columns != 'date'])

temp_full_train_set, temp_test_set = split(temp_df[temp_df['date'] >= '2019-04-01'])
temp_train_set, temp_valid_set = split(temp_full_train_set, train_size = 0.9)
temp_imputer = IterativeImputer(n_nearest_features=100, skip_complete=True, verbose = 2, max_iter = 50)
temp_preprocessor = Preprocessor(temp_imputer, ['date'])
temp_preprocessor.fit(temp_train_set)
temp_train_set = temp_preprocessor.transform(temp_train_set)
temp_valid_set = temp_preprocessor.transform(temp_valid_set)
temp_test_set = temp_preprocessor.transform(temp_test_set)
temp_preprocessor.fit(temp_full_train_set)
temp_full_train_set=temp_preprocessor.transform(temp_full_train_set)
print('Prep completed')


temp_train_set.to_csv(main_dir/'data'/'cleaned'/'temp'/'train_set_imp.csv', index = False)
temp_valid_set.to_csv(main_dir/'data'/'cleaned'/'temp'/'valid_set_imp.csv', index = False)
temp_test_set.to_csv(main_dir/'data'/'cleaned'/'temp'/'test_set_imp.csv', index = False)
temp_full_train_set.to_csv(main_dir/'data'/'cleaned'/'temp'/'full_train_set_imp.csv', index = False)
dump(temp_preprocessor, open(main_dir/'data'/'cleaned'/'preprocessor'/'temp_preprocessor.pkl', 'wb'))