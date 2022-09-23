import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from evaluator.evaluator import Evaluator
from model.cnn_enc_dec_attn import CNNEncDecAttn
from utils import MAIN_DIR
import json

with open('config.json') as f:
    config = json.load(f)
    
col_out = config['PARAMETER']['DATA']['col_temp_in'] + config['PARAMETER']['DATA']['col_temp_ext']
len_forecast = config['PARAMETER']['DATA']['len_forecast']

energy_test_loader = torch.load(MAIN_DIR/'data'/'cleaned'/'energy'/'test_loader.pt')
forecaster_energy = CNNEncDecAttn(len_forecast)
forecaster_energy.load_state_dict(torch.load(MAIN_DIR/'results'/'models'/'forecaster_energy.pt'))
energy_evaluator = Evaluator(energy_test_loader, forecaster_energy, ['hvac'])
energy_res,_ = energy_evaluator.evaluation() 

temp_test_loader = torch.load(MAIN_DIR/'data'/'cleaned'/'temp'/'test_loader.pt')
forecaster_temp = CNNEncDecAttn(len_forecast, len(col_out))
forecaster_temp.load_state_dict(torch.load(MAIN_DIR/'results'/'models'/'forecaster_temp.pt'))
temp_evaluator = Evaluator(temp_test_loader, forecaster_temp, col_out)
temp_res,_ = temp_evaluator.evaluation()

energy_res.to_csv(MAIN_DIR/'results'/'outputs'/'energy_prediction.csv')
temp_res.to_csv(MAIN_DIR/'results'/'outputs'/'temp_prediction.csv')