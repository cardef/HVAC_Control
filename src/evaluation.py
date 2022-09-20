import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from evaluator.evaluator import Evaluator
from model.forecaster_energy import ForecasterEnergy
from model.forecaster_temp import ForecasterTemp
from src.utils import MAIN_DIR
import json

config = json.load("config.json")
col_out = config['PARAMETER']['DATA']['col_temp_in'] + config['PARAMETER']['DATA']['col_temp_ext']
len_forecast = config['PARAMETER']['DATA']['len_forecast']

loss_fn = nn.MSELoss()
energy_test_loader = torch.load(MAIN_DIR/'data'/'cleaned'/'energy'/'energy_test_loader.pt')
forecaster_energy = ForecasterEnergy(len_forecast, 1)
forecaster_energy.load_state_dict(torch.load(MAIN_DIR/'results'/'models'/'forecaster_energy.pt'))
energy_evaluator = Evaluator(energy_test_loader, forecaster_energy, loss_fn, ['hvac'])
energy_res,_ = energy_evaluator.evaluation() 

temp_test_loader = torch.load(MAIN_DIR/'data'/'cleaned'/'temp'/'temp_test_loader.pt')
forecaster_temp = ForecasterTemp(len_forecast, len(col_out))
forecaster_temp.load_state_dict(torch.load(MAIN_DIR/'results'/'models'/'forecaster_temp.pt'))
temp_evaluator = Evaluator(temp_test_loader, forecaster_temp, loss_fn, col_out)
temp_res,_ = temp_evaluator.evaluation()

energy_res.to_csv(MAIN_DIR/'results'/'outputs'/'energy_prediction.csv')
temp_res.to_csv(MAIN_DIR/'results'/'outputs'/'temp_prediction.csv')