from email.errors import MissingHeaderBodySeparatorDefect
from utils import MAIN_DIR, split
from data.preprocesser import Preprocesser
from data.dataset import Dataset, collate_fn

from pickle import dump
from pathlib import Path
from torch.utils.data import DataLoader
import json
import torch
import sys
from data.imputer import Imputer
import missingno as msno
import pandas as pd
import numpy as np
import re
from statistics import mean
from data.merger import Merger

with open('config.json') as f:
    config = json.load(f)

time_window = config['time_window']
len_forecast = config['len_forecast']


main_dir = Path(__file__).parent.parent
path = main_dir/'data'/'lbnlbldg59'/'lbnlbldg59.processed' / 'LBNLBLDG59'/'clean_Bldg59_2018to2020'/'clean data'

try:
    energy_df = pd.read_csv(path/'energy.csv')
except:
    merger = Merger(path, ['en', 'outdoor', 'hvac_op'], main_dir /
                    'raw_csv.json', main_dir/'cleaned_csv.json', 'energy')
    energy_df = merger.merge()
    energy_df['hvac'] = energy_df['hvac_S']+energy_df['hvac_N']
    energy_df = energy_df.drop(
        ['hvac_S', 'hvac_N', 'mels_S', 'lig_S', 'mels_N'], axis=1)

    with open(main_dir/'cleaned_csv.json', 'r') as f:
        cleaned_csv = json.load(f)
    cleaned_csv['energy']['features'].remove('hvac_S')
    cleaned_csv['energy']['features'].remove('hvac_N')
    cleaned_csv['energy']['features'].remove('mels_S')
    cleaned_csv['energy']['features'].remove('lig_S')
    cleaned_csv['energy']['features'].remove('mels_N')
    cleaned_csv['energy']['col_out'] = ['hvac']
    with open(main_dir/'cleaned_csv.json', 'w') as f:
        json.dump(cleaned_csv, f, indent=2)

    energy_df.to_csv(path / 'energy.csv', index=False)


energy_full_train_set, energy_test_set = split(energy_df)


energy_preprocessor = Preprocesser(
    ['date'], outliers=False, remove_col_const=False)
energy_preprocessor.fit(energy_full_train_set)
energy_full_train_set = energy_preprocessor.transform(energy_full_train_set)


energy_test_set = energy_preprocessor.transform(energy_test_set)
dump(energy_preprocessor, open(main_dir/'data'/'cleaned' /
     'preprocessor'/'energy_full_train_preprocessor.pkl', 'wb'))


energy_train_set, energy_valid_set = split(
    energy_full_train_set, train_size=0.9)
energy_preprocessor = Preprocesser(
    ['date'], outliers=False, remove_col_const=False)
energy_preprocessor.fit(energy_train_set)
energy_train_set = energy_preprocessor.transform(energy_train_set)

energy_valid_set = energy_preprocessor.transform(energy_valid_set)

dump(energy_preprocessor, open(main_dir/'data'/'cleaned' /
     'preprocessor'/'energy_train_preprocessor.pkl', 'wb'))


print('Prep completed')

energy_train_set.to_csv(main_dir/'data'/'cleaned' /
                        'energy'/'train_set_imp.csv', index=False)
energy_valid_set.to_csv(main_dir/'data'/'cleaned' /
                        'energy'/'valid_set_imp.csv', index=False)
energy_test_set.to_csv(main_dir/'data'/'cleaned' /
                       'energy'/'test_set_imp.csv', index=False)
energy_full_train_set.to_csv(
    main_dir/'data'/'cleaned'/'energy'/'full_train_set_imp.csv', index=False)

try:
    temp_df = pd.read_csv(path/'temp.csv')

except:
    merger = Merger(path, ['outdoor', 'hvac_op', 'indoor'],
                    main_dir/'raw_csv.json', main_dir/'cleaned_csv.json', 'temp')
    temp_df = merger.merge()
    temp_df.to_csv(path/'temp.csv', index=False)

    with open(main_dir/'cleaned_csv.json', 'r') as f:
        cleaned_csv = json.load(f)
    temp_col_out = [x for x in cleaned_csv['temp']['features'] if bool(
    re.search("zone_\d\d\d_temp", x)) or x.startswith("cerc_templogger")]
    cleaned_csv['temp']['col_out'] = temp_col_out
    with open(main_dir/'cleaned_csv.json', 'w') as f:
        json.dump(cleaned_csv, f, indent=2)


temp_full_train_set, temp_test_set = split(temp_df)

temp_preprocessor = Preprocesser(
    ['date'], outliers=False, remove_col_const=False)
temp_preprocessor.fit(temp_full_train_set)
temp_full_train_set = temp_preprocessor.transform(temp_full_train_set)
temp_test_set = temp_preprocessor.transform(temp_test_set)
dump(temp_preprocessor, open(main_dir/'data'/'cleaned' /
     'preprocessor'/'temp_full_train_preprocessor.pkl', 'wb'))


temp_train_set, temp_valid_set = split(temp_full_train_set, train_size=0.9)
temp_preprocessor = Preprocesser(
    ['date'], outliers=False, remove_col_const=False)
temp_preprocessor.fit(temp_train_set)
temp_train_set = temp_preprocessor.transform(temp_train_set)
temp_valid_set = temp_preprocessor.transform(temp_valid_set)
dump(temp_preprocessor, open(main_dir/'data'/'cleaned' /
     'preprocessor'/'temp_train_preprocessor.pkl', 'wb'))

temp_train_set.to_csv(main_dir/'data'/'cleaned'/'temp' /
                      'train_set_imp.csv', index=False)
temp_valid_set.to_csv(main_dir/'data'/'cleaned'/'temp' /
                      'valid_set_imp.csv', index=False)
temp_test_set.to_csv(main_dir/'data'/'cleaned'/'temp' /
                     'test_set_imp.csv', index=False)
temp_full_train_set.to_csv(
    main_dir/'data'/'cleaned'/'temp'/'full_train_set_imp.csv', index=False)
