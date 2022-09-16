from utils import merge, split
from dataset.preprocessor import Preprocessor
from sklearn.impute import KNNImputer, SimpleImputer
import os

en = ['ele.csv']
outdoor = ['site_weather.csv']
hvac_op = ['hp_hws_temp.csv', 'rtu_sa_t_sp.csv', 'rtu_sa_t.csv', 'rtu_ra_t.csv', 'rtu_ma_t.csv','rtu_oa_t.csv', 'rtu_sa_fr.csv', 'rtu_oa_damper.csv', 'rtu_econ_sp.csv', 'rtu_sa_p_sp.csv', 'rtu_plenum_p.csv', 'rtu_fan_spd.csv', 'uft_fan_spd.csv', 'uft_hw_valve.csv']
indoor = ['zone_temp_sp_c.csv', 'zone_temp_sp_h.csv', 'zone_temp_interior.csv', 'zone_temp_exterior.csv']
path_raw = r'D:\My Drive\Uni\Vicomtech\Tesi\HVAC_Control\dataset\doi_10.7941_D1N33Q__v6\Building_59\Bldg59_clean data'
path_cleaned = r'D:\My Drive\Uni\Vicomtech\Tesi\HVAC_Control\dataset'


energy_df = merge(path_raw, en+outdoor+hvac_op)
energy_df['hvac'] = energy_df['hvac_S']+energy_df['hvac_N']
energy_df = energy_df.drop(['hvac_S', 'hvac_N'], axis = 1)
energy_train_set, energy_test_set = split(energy_df)
enerrgy_test_set_timestamp = energy_test_set['date']
energy_train_set=energy_train_set.drop('date', axis=1)
energy_test_set=energy_test_set.drop('date', axis=1)
energy_imputer = SimpleImputer()
energy_preprocessor = Preprocessor(energy_imputer)
energy_preprocessor.fit(energy_train_set)
energy_train_set = energy_preprocessor.transform(energy_train_set)
energy_test_set = energy_preprocessor.transform(energy_test_set)

energy_train_set.to_csv(os.path.join(path_cleaned,'energy', 'train_set_imp.csv'), index = False)
energy_test_set.to_csv(os.path.join(path_cleaned,'energy', 'test_set_imp.csv'), index = False)