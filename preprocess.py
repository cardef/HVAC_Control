from ..utils import col_out_to_index, merge, remove_outliers, split, imputation, split_seq
from sklearn.impute import IterativeImputer , KNNImputer, SimpleImputer
from dataloader import Dataset
import os

en = ['ele.csv']
outdoor = ['site_weather.csv']
hvac_op = ['hp_hws_temp.csv', 'rtu_sa_t_sp.csv', 'rtu_sa_t.csv', 'rtu_ra_t.csv', 'rtu_ma_t.csv','rtu_oa_t.csv', 'rtu_sa_fr.csv', 'rtu_oa_damper.csv', 'rtu_econ_sp.csv', 'rtu_sa_p_sp.csv', 'rtu_plenum_p.csv', 'rtu_fan_spd.csv', 'uft_fan_spd.csv', 'uft_hw_valve.csv']
indoor = ['zone_temp_sp_c.csv', 'zone_temp_sp_h.csv', 'zone_temp_interior.csv', 'zone_temp_exterior.csv']
path_raw = r'C:\Users\cdellefemine\Documents\GitHub\HVAC-Control\dataset\doi_10.7941_D1N33Q__v6\Building_59\Bldg59_clean data'
path_cleaned = r'C:\Users\cdellefemine\Documents\GitHub\HVAC-Control\dataset'
energy_df = merge(path_raw, en+outdoor+hvac_op)
energy_df['hvac'] = energy_df['hvac_S']+energy_df['hvac_N']
energy_df = remove_outliers(energy_df)
energy_train_set, energy_test_set = split(energy_df)

energy_train_set.to_csv(os.path.join(path_cleaned, 'energy\train_set.csv'), index = False)
energy_test_set.to_csv(os.path.join(path_cleaned, 'energy\test_set.csv'), index = False)

costant_col = energy_train_set.columns[energy_train_set.nunique() <= 1]
train_set = energy_train_set.drop(costant_col, axis = 1)
test_set = energy_test_set.drop(costant_col, axis = 1)

train_set = energy_train_set.drop('date', axis = 1)
test_set = energy_test_set.drop('date', axis = 1)

X_mean = energy_train_set.mean(axis=0)
X_std = energy_train_set.mean(axis = 0)

imputer = SimpleImputer()
energy_train_set_imp = imputation(energy_train_set, imputer, True)
energy_test_set_imp = imputation(energy_test_set, imputer)

energy_train_set_imp.to_csv(os.path.join(path_cleaned,'energy\train_set_imp.csv'), index = False)
energy_test_set_imp.to_csv(os.path.join(path_cleaned,'energy\test_set_imp.csv'), index = False)

col_out = ['hvac']
col_out_in = col_out_to_index(energy_df, col_out)
col_out_in_x = [col[0] for col in col_out_in.values()]