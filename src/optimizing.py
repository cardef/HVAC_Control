from copyreg import pickle
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from evaluator.evaluator import Evaluator
from model.cnn_enc_dec_attn import CNNEncDecAttn
from utils import MAIN_DIR, col_out_to_index
import json
from torch.utils.data import Dataset,DataLoader
from data.dataset import collate_fn
from pythermalcomfort.models import pmv_ppd, clo_tout
from pythermalcomfort.utilities import v_relative, clo_dynamic
from pythermalcomfort.utilities import met_typical_tasks
import pickle

with open('config.json') as f:
    config = json.load(f)
    
with open(MAIN_DIR/'data'/'cleaned'/'preprocessor'/'energy_full_train_preprocessor.pkl', 'rb') as f:
    energy_preprocessor = pickle.load(f)

with open(MAIN_DIR/'data'/'cleaned'/'preprocessor'/'temp_full_train_preprocessor.pkl', 'rb') as f:
    temp_preprocessor = pickle.load(f)

print(energy_preprocessor.col_const)

col_out = config['PARAMETER']['DATA']['col_temp_in'] + config['PARAMETER']['DATA']['col_temp_ext']
len_forecast = config['PARAMETER']['DATA']['len_forecast']
time_window = config['PARAMETER']['DATA']['time_window']
hvac_op_col = config['PARAMETER']['DATA']['hvac_op_col']
outdoor_col = config['PARAMETER']['DATA']['outdoor_col']

forecaster_energy = CNNEncDecAttn.load_from_checkpoint(MAIN_DIR/'results'/'models'/'forecaster_energy.ckpt')


forecaster_temp = CNNEncDecAttn.load_from_checkpoint(MAIN_DIR/'results'/'models'/'forecaster_temp.ckpt')


energy_test_set = pd.read_csv(MAIN_DIR/'data'/'cleaned'/'energy'/'test_set_imp.csv')[:time_window]
temp_test_set = pd.read_csv(MAIN_DIR/'data'/'cleaned'/'temp'/'test_set_imp.csv')[:time_window]
outdoor_index,_ = zip(*col_out_to_index(energy_test_set, outdoor_col).values())
hvac_op_index,_ = zip(*col_out_to_index(energy_test_set, hvac_op_col).values())
print(outdoor_index, hvac_op_index, energy_preprocessor.col_out, temp_preprocessor.col_out)
'''
temp_test = Dataset(temp_test_set, time_window, len_forecast, col_out)
energy_test = Dataset(energy_test_set, time_window, len_forecast, col_out)

for n in range(100):
    energy_pred = forecaster_energy(energy_test)
    temp_pred = forecaster_temp(temp_test)
    
    input = torch.normal(torch.zeros(len(hvac_op)) ,1)
    outdoor = energy_test_set.iloc[]
'''