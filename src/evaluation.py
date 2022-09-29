import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from evaluator.evaluator import Evaluator
from model.cnn_enc_dec_attn import CNNEncDecAttn
from utils import MAIN_DIR
import json
from data.dataset import collate_fn
from torch.utils.data import DataLoader
from data.dataset import Dataset

with open('config.json') as f:
    config = json.load(f)
    
col_out = config['PARAMETER']['DATA']['col_temp_in'] + config['PARAMETER']['DATA']['col_temp_ext']
len_forecast = config['PARAMETER']['DATA']['len_forecast']
time_window = config['PARAMETER']['DATA']['time_window']

energy_test_set =pd.read_csv(MAIN_DIR/'data'/'cleaned'/'energy'/'test_set_imp.csv')
energy_test_loader = DataLoader(Dataset(energy_test_set, time_window, len_forecast, ['hvac']), batch_size = 64, collate_fn = collate_fn, num_workers = 14)
forecaster_energy = CNNEncDecAttn.load_from_checkpoint(MAIN_DIR/'results'/'models'/'forecaster_energy.ckpt')
energy_evaluator = Evaluator(energy_test_loader, forecaster_energy, ['hvac'])
energy_res = energy_evaluator.evaluation() 

temp_test_set =pd.read_csv(MAIN_DIR/'data'/'cleaned'/'temp'/'test_set_imp.csv')
temp_test_loader = DataLoader(Dataset(temp_test_set, time_window, len_forecast, col_out), batch_size = 64, collate_fn = collate_fn, num_workers = 14)
forecaster_temp = CNNEncDecAttn.load_from_checkpoint(MAIN_DIR/'results'/'models'/'forecaster_temp.ckpt')
temp_evaluator = Evaluator(temp_test_loader, forecaster_temp, col_out)
temp_res = temp_evaluator.evaluation()

energy_res.to_csv(MAIN_DIR/'results'/'outputs'/'energy_prediction.csv')
temp_res.to_csv(MAIN_DIR/'results'/'outputs'/'temp_prediction.csv')