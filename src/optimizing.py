import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from evaluator.evaluator import Evaluator
from model.cnn_enc_dec_attn import CNNEncDecAttn
from utils import MAIN_DIR
import json
from torch.utils.data import Dataset,DataLoader

with open('config.json') as f:
    config = json.load(f)
    
col_out = config['PARAMETER']['DATA']['col_temp_in'] + config['PARAMETER']['DATA']['col_temp_ext']
len_forecast = config['PARAMETER']['DATA']['len_forecast']
time_window = config['PARAMETER']['DATA']['time_window']

forecaster_energy = CNNEncDecAttn(len_forecast)
forecaster_energy.load_state_dict(torch.load(MAIN_DIR/'results'/'models'/'forecaster_energy.pt'))

forecaster_temp = CNNEncDecAttn(len_forecast, len(col_out))
forecaster_temp.load_state_dict(torch.load(MAIN_DIR/'results'/'models'/'forecaster_temp.pt'))

energy_test_set = pd.read_csv(MAIN_DIR/'data'/'cleaned'/'energy'/'test_set_imp.csv')
temp_test_set = pd.read_csv(MAIN_DIR/'data'/'cleaned'/'temp'/'test_set_imp.csv')

temp_test_loader = DataLoader(Dataset(temp_test_set, time_window, len_forecast, col_out), batch_size = 64, collate_fn = collate_fn, num_workers = 14)
energy_test_loader = DataLoader(Dataset(energy_test_set, time_window, len_forecast, col_out), batch_size = 64, collate_fn = collate_fn, num_workers = 14)