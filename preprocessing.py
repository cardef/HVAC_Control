from utils import merge, split
from data_loader.preprocessor import Preprocessor
from data_loader.dataset import Dataset
from sklearn.impute import KNNImputer, SimpleImputer
from pickle import dump
from pathlib import Path
from torch.utils.data import DataLoader
import json

config = json.load("config.json")
time_window = config['PARAMETER']['DATA']['time_window']
len_forecast = config['PARAMETER']['DATA']['len_forecast'] 


en = ['ele.csv']
outdoor = ['site_weather.csv']
hvac_op = ['hp_hws_temp.csv', 'rtu_sa_t_sp.csv', 'rtu_sa_t.csv', 'rtu_ra_t.csv', 'rtu_ma_t.csv','rtu_oa_t.csv', 'rtu_sa_fr.csv', 'rtu_oa_damper.csv', 'rtu_econ_sp.csv', 'rtu_sa_p_sp.csv', 'rtu_plenum_p.csv', 'rtu_fan_spd.csv', 'uft_fan_spd.csv', 'uft_hw_valve.csv']
indoor = ['zone_temp_sp_c.csv', 'zone_temp_sp_h.csv', 'zone_temp_interior.csv', 'zone_temp_exterior.csv']
main_dir = Path(__file__).parent
path_raw = r'D:\My Drive\Uni\Vicomtech\Tesi\HVAC_Control\dataset\doi_10.7941_D1N33Q__v6\Building_59\Bldg59_clean data'
path_cleaned = r'D:\My Drive\Uni\Vicomtech\Tesi\HVAC_Control\dataset'


energy_df = merge(path_raw, en+outdoor+hvac_op)
energy_df['hvac'] = energy_df['hvac_S']+energy_df['hvac_N']
energy_df = energy_df.drop(['hvac_S', 'hvac_N'], axis = 1)

energy_train_set, energy_test_set = split(energy_df)
energy_train_set, energy_valid_set = split(energy_df, float = 0.7)
energy_imputer = SimpleImputer()
energy_preprocessor = Preprocessor(energy_imputer, ['date'])
energy_preprocessor.fit(energy_train_set)
energy_train_set = energy_preprocessor.transform(energy_train_set)
energy_valid_set = energy_preprocessor.transform(energy_valid_set)
energy_test_set = energy_preprocessor.transform(energy_test_set)

energy_train_set.to_csv(main_dir/'dataset'/'cleaned'/'energy'/'train_set_imp.csv', index = False)
energy_valid_set.to_csv(main_dir/'dataset'/'cleaned'/'energy'/'valid_set_imp.csv', index = False)
energy_test_set.to_csv(main_dir/'dataset'/'cleaned'/'energy'/'test_set_imp.csv', index = False)
dump(energy_preprocessor, open(main_dir/'dataset'/'cleaned'/'preprocessor'/'energy_preprocessor.pkl', 'wb'))

energy_train_loader = DataLoader(Dataset(energy_train_set, time_window, len_forecast, ['hvac']), batch_size = 64)
energy_valid_loader = DataLoader(Dataset(energy_valid_set, time_window, len_forecast, ['hvac']), batch_size = 64)
energy_tes_loader = DataLoader(Dataset(energy_test_set, time_window, len_forecast, ['hvac']), batch_size = 64)





temp_df = merge(path_raw, outdoor+hvac_op+indoor)

temp_train_set, temp_test_set = split(temp_df)
temp_train_set, temp_valid_set = split(temp_df, float = 0.7)
temp_imputer = SimpleImputer()
temp_preprocessor = Preprocessor(temp_imputer, ['date'])
temp_preprocessor.fit(temp_train_set)
temp_train_set = temp_preprocessor.transform(temp_train_set)
temp_valid_set = temp_preprocessor.transform(temp_valid_set)
temp_test_set = temp_preprocessor.transform(temp_test_set)

temp_train_set.to_csv(main_dir/'dataset'/'cleaned'/'temp'/'train_set_imp.csv', index = False)
temp_valid_set.to_csv(main_dir/'dataset'/'cleaned'/'temp'/'valid_set_imp.csv', index = False)
temp_test_set.to_csv(main_dir/'dataset'/'cleaned'/'temp'/'test_set_imp.csv', index = False)
dump(temp_preprocessor, open(main_dir/'dataset'/'cleaned'/'preprocessor'/'temp_preprocessor.pkl', 'wb'))

temp_train_loader = DataLoader(Dataset(temp_train_set, time_window, len_forecast, ['hvac']), batch_size = 64)
temp_valid_loader = DataLoader(Dataset(temp_valid_set, time_window, len_forecast, ['hvac']), batch_size = 64)
temp_tes_loader = DataLoader(Dataset(temp_test_set, time_window, len_forecast, ['hvac']), batch_size = 64)