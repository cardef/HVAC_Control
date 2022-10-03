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

def loss_fn(energy, temp):
    '''
    activity = "Seated, quiet"
    met = met_typical_tasks[activity]  #metabolic rate associated to the activity

    vr = v_relative(v=0.1, met=met) #relative air velocity considering the activity
    clo = clo_tout(15) #clothing level given outdoor temp

    results = pmv_ppd(tdb=temp, tr=temp, vr=vr, rh=50, met=met, clo=clo, standard="ISO")
    print(results)

    ppd = results['ppd']
    '''
    loss = torch.sum(energy) + 100*(temp-23) + 100*(temp+19)
    return loss


with open('config.json') as f:
    config = json.load(f)

with open("cleaned_csv.json") as f:
    cleaned_csv = json.load(f)
len_forecast = config['len_forecast']
time_window = config['time_window']
col_out_temp = cleaned_csv['temp']['col_out']
col_out_energy = cleaned_csv['energy']['col_out']
features_temp = cleaned_csv['temp']['features']
features_energy = cleaned_csv['energy']['features']

forecaster_energy = CNNEncDecAttn.load_from_checkpoint(MAIN_DIR/'results'/'models'/'forecaster_energy.ckpt')


forecaster_temp = CNNEncDecAttn.load_from_checkpoint(MAIN_DIR/'results'/'models'/'forecaster_temp.ckpt')


energy_test_set = pd.read_csv(MAIN_DIR/'data'/'cleaned'/'energy'/'test_set_imp.csv')[:time_window]
temp_test_set = pd.read_csv(MAIN_DIR/'data'/'cleaned'/'temp'/'test_set_imp.csv')[:time_window]
energy_test_set.set_index('date', inplace=True)
temp_test_set.set_index('date', inplace=True)
temp_test = torch.Tensor(temp_test_set.values).unsqueeze(0)
energy_test = torch.Tensor(energy_test_set.values).unsqueeze(0)

for i in range(720):
    energy_pred = forecaster_energy(energy_test)
    temp_pred = forecaster_temp(temp_test)
    
    hvac_op = torch.normal(torch.zeros(len(features_temp)-len(col_out_temp)-5 ,1))
    outdoor = torch.Tensor(energy_test_set.iloc[:,:4].values)
    print(outdoor.shape)
    optimizer = torch.optim.Adam([hvac_op])
    for epoch in range(100):

        optimizer.zero_grad()

        energy_input = torch.cat((outdoor, hvac_op, energy_pred), dim = 0)
        energy_input = torch.cat((input, energy_test[-(time_window-len_forecast)]), dim = 1)
        energy_pred = forecaster_energy(energy_input)

        temp_input = torch.cat(outdoor, hvac_op, temp_pred, dim = 0)
        temp_input = torch.cat(input, temp_test[-(time_window-len_forecast)], dim = 1)
        temp_pred = forecaster_temp(temp_input)

        loss = loss_fn(energy_pred, temp_pred)

        loss.backward()

        optimizer.step()