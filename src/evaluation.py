import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from evaluator.evaluator import Evaluator
from model.forecaster_energy import ForecasterEnergy
from model.forecaster_temp import ForecasterTemp
from utils import MAIN_DIR
import json
from pickle import load

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('config.json') as f:
    config = json.load(f)
    
col_out = config['PARAMETER']['DATA']['col_temp_in'] + config['PARAMETER']['DATA']['col_temp_ext']
len_forecast = config['PARAMETER']['DATA']['len_forecast']

loss_fn = nn.MSELoss()

with open(MAIN_DIR/'data'/'cleaned'/'preprocessor'/'energy_preprocessor.pkl', 'rb') as f:
    preprocessor_energy = load(f)

energy_test_loader = torch.load(MAIN_DIR/'data'/'cleaned'/'energy'/'test_loader.pt')
forecaster_energy = ForecasterEnergy(len_forecast)
forecaster_energy.load_state_dict(torch.load(MAIN_DIR/'results'/'models'/'forecaster_energy.pt', map_location=torch.device(device)))
energy_evaluator = Evaluator(energy_test_loader, preprocessor_energy, forecaster_energy, loss_fn, ['hvac'])
energy_res,_ = energy_evaluator.evaluation() 

with open(MAIN_DIR/'data'/'cleaned'/'preprocessor'/'temp_preprocessor.pkl', 'rb') as f:
    preprocessor_temp = load(f)

temp_test_loader = torch.load(MAIN_DIR/'data'/'cleaned'/'temp'/'test_loader.pt')
forecaster_temp = ForecasterTemp(len_forecast, len(col_out))
forecaster_temp.load_state_dict(torch.load(MAIN_DIR/'results'/'models'/'forecaster_temp.pt', map_location=torch.device(device)))
temp_evaluator = Evaluator(temp_test_loader, preprocessor_temp, forecaster_temp, loss_fn, col_out)
temp_res,_ = temp_evaluator.evaluation()

energy_res.to_csv(MAIN_DIR/'results'/'outputs'/'energy_prediction.csv')
temp_res.to_csv(MAIN_DIR/'results'/'outputs'/'temp_prediction.csv')