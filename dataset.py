import torch
import seaborn as sns
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader

#!pip install fancyimpute
import csv
import numpy as np
import pandas as pd
from pandas import Series
import datetime
import time
import os
import gc
import re
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer , KNNImputer, SimpleImputer
from utils import col_out_to_index


class Dataset(Dataset):
    
    def __init__(self, dataset):
        self.dataset = dataset
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]

time_window = 96
len_forecast = 4
batch_size = 64

en = ['ele.csv']
outdoor = ['site_weather.csv']
hvac_op = ['hp_hws_temp.csv', 'rtu_sa_t_sp.csv', 'rtu_sa_t.csv', 'rtu_ra_t.csv', 'rtu_ma_t.csv','rtu_oa_t.csv', 'rtu_sa_fr.csv', 'rtu_oa_damper.csv', 'rtu_econ_sp.csv', 'rtu_sa_p_sp.csv', 'rtu_plenum_p.csv', 'rtu_fan_spd.csv', 'uft_fan_spd.csv', 'uft_hw_valve.csv']
indoor = ['zone_temp_sp_c.csv', 'zone_temp_sp_h.csv', 'zone_temp_interior.csv', 'zone_temp_exterior.csv']
path_raw = r'C:\Users\cdellefemine\Documents\GitHub\HVAC-Control\Dataset\doi_10.7941_D1N33Q__v6\Building_59\Bldg59_clean data\\'
path_cleaned = r'C:\Users\cdellefemine\Documents\GitHub\HVAC-Control\Dataset\\'
energy_df = merge(path_raw, en+outdoor+hvac_op)
energy_df['hvac'] = energy_df['hvac_S']+energy_df['hvac_N']
energy_df = remove_outliers(energy_df)
energy_train_set, energy_valid_set, energy_test_set = split(energy_df)

energy_train_set.to_csv(path_cleaned+'Energy\train_set.csv', index = False)
energy_test_set.to_csv(path_cleaned+'Energy\test_set.csv', index = False)
energy_valid_set.to_csv(path_cleaned+'Energy\valid_set.csv', index = False)

costant_col = energy_train_set.columns[energy_train_set.nunique() <= 1]
train_set = energy_train_set.drop(costant_col, axis = 1)
test_set = energy_test_set.drop(costant_col, axis = 1)
valid_set = energy_valid_set.drop(costant_col, axis = 1)

train_set = energy_train_set.drop('date', axis = 1)
test_set = energy_test_set.drop('date', axis = 1)
valid_set = energy_valid_set.drop('date', axis = 1)

X_mean = energy_train_set.mean(axis=0)
X_std = energy_train_set.mean(axis = 0)

imputer = SimpleImputer()
energy_train_set_imp = imputation(energy_train_set, imputer, True)
energy_valid_set_imp = imputation(energy_valid_set, imputer)
energy_test_set_imp = imputation(energy_test_set, imputer)

energy_train_set_imp.to_csv(path_cleaned+'Energy\train_set_imp.csv', index = False)
energy_test_set_imp.to_csv(path_cleaned+'Energy\test_set_imp.csv', index = False)
energy_valid_set_imp.to_csv(path_cleaned+'Energy\valid_set_imp.csv', index = False)

col_out = ['hvac']
col_out_in = col_out_to_index(energy_df, col_out)
col_out_in_x = [col[0] for col in col_out_in.values()]


train_seq = split_seq(energy_train_set_imp, time_window, len_forecast, col_out_in_x)
test_seq = split_seq(energy_test_set_imp, time_window, len_forecast, col_out_in_x)
valid_seq = split_seq(energy_valid_set_imp, time_window, len_forecast, col_out_in_x)

train_loader = DataLoader(Dataset(train_seq), batch_size)
valid_loader = DataLoader(Dataset(valid_seq), batch_size)

torch.save(train_loader, path_cleaned+'Energy\train_loader.pt')
torch.save(valid_loader, path_cleaned+'Energy\valid_loader.pt')
torch.save(train_seq, path_cleaned+'Energy\train_seq.pt')
torch.save(valid_seq, path_cleaned+'Energy\valid_seq.pt')
torch.save(test_seq, path_cleaned+'Energy\test_seq.pt')
torch.save(valid_loader, path_cleaned+'Energy\valid_loader.pt')



temp_df = merge(path_raw, outdoor+hvac_op+indoor)
temp_df = remove_outliers(temp_df)
temp_train_set, temp_valid_set, temp_test_set = split(temp_df)

temp_train_set.to_csv(path_cleaned+'Temp\train_set.csv', index = False)
temp_test_set.to_csv(path_cleaned+'Temp\test_set.csv', index = False)
temp_valid_set.to_csv(path_cleaned+'Temp\valid_set.csv', index = False)

costant_col = temp_train_set.columns[temp_train_set.nunique() <= 1]
train_set = temp_train_set.drop(costant_col, axis = 1)
test_set = temp_test_set.drop(costant_col, axis = 1)
valid_set = temp_valid_set.drop(costant_col, axis = 1)

train_set = temp_train_set.drop('date', axis = 1)
test_set = temp_test_set.drop('date', axis = 1)
valid_set = temp_valid_set.drop('date', axis = 1)

X_mean = temp_train_set.mean(axis=0)
X_std = temp_train_set.mean(axis = 0)

imputer = SimpleImputer()
temp_train_set_imp = imputation(temp_train_set, imputer, True)
temp_valid_set_imp = imputation(temp_valid_set, imputer)
temp_test_set_imp = imputation(temp_test_set, imputer)

temp_train_set_imp.to_csv(path_cleaned+'Temp\train_set_imp.csv', index = False)
temp_test_set_imp.to_csv(path_cleaned+'Temp\test_set_imp.csv', index = False)
temp_valid_set_imp.to_csv(path_cleaned+'Temp\valid_set_imp.csv', index = False)

col_out = [col for col in list(temp_df.columns) if re.search('zone_\d\d\d_temp', col) or re.search('cerc_templogger_', col)]
col_out_in = col_out_to_index(temp_df, col_out)
col_out_in_x = [col[0] for col in col_out_in.values()]


train_seq = split_seq(temp_train_set_imp, time_window, len_forecast, col_out_in_x)
test_seq = split_seq(temp_test_set_imp, time_window, len_forecast, col_out_in_x)
valid_seq = split_seq(temp_valid_set_imp, time_window, len_forecast, col_out_in_x)

train_loader = DataLoader(Dataset(train_seq), batch_size)
valid_loader = DataLoader(Dataset(valid_seq), batch_size)

torch.save(train_loader, path_cleaned+'Temp\train_loader.pt')
torch.save(valid_loader, path_cleaned+'Temp\valid_loader.pt')
torch.save(train_seq, path_cleaned+'Temp\train_seq.pt')
torch.save(valid_seq, path_cleaned+'Temp\valid_seq.pt')
torch.save(test_seq, path_cleaned+'Temp\test_seq.pt')
torch.save(valid_loader, path_cleaned+'Temp\valid_loader.pt')