from copyreg import pickle
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from evaluator.evaluator import Evaluator
from model.cnn_enc_dec_attn import CNNEncDecAttn
from utils import MAIN_DIR, col_out_to_index
import json
from torch.utils.data import Dataset, DataLoader
from data.dataset import collate_fn
from pythermalcomfort.models import pmv_ppd, clo_tout
from pythermalcomfort.utilities import v_relative, clo_dynamic
from pythermalcomfort.utilities import met_typical_tasks
from statistics import mean
from tqdm import tqdm
from torchviz import make_dot, make_dot_from_trace


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
    loss = torch.sum(energy) + 100*torch.sum(temp-0.3) + \
        100*torch.sum(temp+0.3)
    return loss


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
print(len(features_energy), len(features_temp), len(col_out_temp))
hvac_not_in_en = [a for a in features_temp if (
    a not in features_energy) & (a not in col_out_temp)]

forecaster_energy = CNNEncDecAttn.load_from_checkpoint(
    MAIN_DIR/'results'/'models'/'forecaster_energy.ckpt').to(device)


forecaster_temp = CNNEncDecAttn.load_from_checkpoint(
    MAIN_DIR/'results'/'models'/'forecaster_temp.ckpt').to(device)


energy_test_set = pd.read_csv(
    MAIN_DIR/'data'/'cleaned'/'energy'/'test_set_imp.csv')
temp_test_set = pd.read_csv(
    MAIN_DIR/'data'/'cleaned'/'temp'/'test_set_imp.csv')
energy_test_set.set_index('date', inplace=True)
temp_test_set.set_index('date', inplace=True)
outdoor = energy_test_set.iloc[time_window:, :5]

temp_test = torch.Tensor(temp_test_set[:time_window].values).to(device)
energy_test = torch.Tensor(energy_test_set[:time_window].values).to(device)
temp_test.requires_grad_(False)
energy_test.requires_grad_(False)
energy_first_pred = forecaster_energy(energy_test.to(
    device).unsqueeze(0)).squeeze(0).transpose(0, 1)
temp_first_pred = forecaster_temp(temp_test.to(
    device).unsqueeze(0)).squeeze(0).transpose(0, 1)
temp_first_pred.detach()
energy_first_pred.detach()
for i in range(720):

    hvac_op = torch.normal(torch.zeros(
        len(features_temp)-len(col_out_temp)-5, len_forecast)).to(device)
    hvac_op.requires_grad_(True)

    optimizer = torch.optim.Adam([hvac_op])
    loss_epochs = []
    for epoch in tqdm(range(100)):

        optimizer.zero_grad()
        '''
        outdoor_pred = torch.Tensor(outdoor[i*len_forecast:(i+1)*len_forecast].values).to(device)
        outdoor_pred.requires_grad_(False)
        energy_input = torch.cat((outdoor_pred.transpose(0,1), hvac_op[:len(hvac_op)-len(hvac_not_in_en)], energy_first_pred), dim = 0)
        energy_opt_test = torch.cat((energy_input, energy_test.transpose(0,1)), dim = 1)
        outdoor_pred.requires_grad_(False)
        

        temp_input = torch.cat((outdoor_pred.transpose(0,1), hvac_op, temp_first_pred), dim = 0)
        temp_opt_test = torch.cat((temp_input, temp_test.transpose(0,1)), dim = 1)
        
        
        '''
        energy_pred = forecaster_energy(
            torch.cat((torch.cat((torch.Tensor(outdoor[i*len_forecast:(i+1)*len_forecast].values).to(device), hvac_op[:len(hvac_op)-len(hvac_not_in_en)], forecaster_energy(
                energy_test.to(device)).squeeze(0)), dim=0), energy_test), dim=1)[:, i*len_forecast:time_window+i*len_forecast].to(device)).squeeze(0)
        temp_pred = forecaster_temp(torch.cat((torch.cat((torch.Tensor(outdoor[i*len_forecast:(i+1)*len_forecast].values).to(device), hvac_op, forecaster_temp(temp_test.to(device)).squeeze(0)), dim=0), temp_test), dim=1)[:, i*len_forecast:time_window+i*len_forecast].to(device)).squeeze(0)
        loss = loss_fn(energy_pred, temp_pred)

        temp_test.requires_grad_(False)
        energy_test.requires_grad_(False)
        loss.backward(retain_graph=False)

        loss_epochs.append(loss.item())
        optimizer.step()
    print(mean(loss_epochs))
