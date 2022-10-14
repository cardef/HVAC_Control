from copyreg import pickle
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from evaluator.evaluator import Evaluator
from model.cnn_enc_dec_attn import CNNEncDecAttn
from optimizer.optimizer import Optimizer
from utils import MAIN_DIR, col_out_to_index
import json
from torch.utils.data import Dataset, DataLoader
from data.dataset import collate_fn
from statistics import mean
from tqdm import tqdm
from pytorch_lightning.callbacks import LearningRateMonitor
import math
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from itertools import product
import pytorch_lightning as pl


def res_df(true_df, opt_df, col_out):
    opt_res = opt_df.loc[:, col_out].to_numpy()
    true = true_df.loc[opt_df.index, col_out].to_numpy()
    columns = product(col_out, ['true', 'opt'])
    res_matrix = np.zeros((opt_res.shape[0], opt_res.shape[1]*2))
    for i in range(opt_res.shape[1]):
        res_matrix[:, i*2] = true[:, i]
        res_matrix[:, i*2 + 1] = opt_res[:, i]
    res_df = pd.DataFrame(res_matrix, columns=columns)
    res_df['date'] = opt_df.index
    res_df = res_df.set_index('date')
    res_df.columns = pd.MultiIndex.from_tuples(
        res_df.columns, names=['Variable', 'Type'])

    return res_df


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
hvac_index = cleaned_csv['energy']['features'].index('hvac')
col_out_temp_index = [cleaned_csv['temp']
                      ['features'].index(col) for col in col_out_temp]
hvac_op_index = [cleaned_csv['temp']
                 ['features'].index(col) for col in col_out_temp]
hvac_op_std = cleaned_csv['energy']['normalisation'][1][6:]
temp_norm = [[cleaned_csv['temp']['normalisation'][0][col] for col in col_out_temp_index], [
    cleaned_csv['temp']['normalisation'][1][col] for col in col_out_temp_index]]
horizon = 3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

forecaster_energy = CNNEncDecAttn.load_from_checkpoint(
    MAIN_DIR/'results'/'models'/'forecaster_energy.ckpt').to(device)


forecaster_temp = CNNEncDecAttn.load_from_checkpoint(
    MAIN_DIR/'results'/'models'/'forecaster_temp.ckpt').to(device)


energy_test_set = pd.read_csv(
    MAIN_DIR/'data'/'cleaned'/'energy'/'test_set_imp.csv')
temp_test_set = pd.read_csv(
    MAIN_DIR/'data'/'cleaned'/'temp'/'test_set_imp.csv')
energy_test_set['date'] = pd.to_datetime(energy_test_set['date'])
temp_test_set['date'] = pd.to_datetime(temp_test_set['date'])
energy_test_set.set_index('date', inplace=True)
temp_test_set.set_index('date', inplace=True)
energy_opt_df = energy_test_set[energy_test_set.index.month == 11]
temp_opt_df = temp_test_set[temp_test_set.index.month == 11]
#energy_opt_df = energy_test_set.head(-(len(energy_opt_df)-time_window)%len_forecast)
#temp_opt_df = temp_test_set.head(-(len(temp_opt_df)-time_window)%len_forecast)
#energy_opt_df = energy_opt_df.iloc[:96*30]
#temp_opt_df = temp_opt_df.iloc[:96*2]
outdoor = energy_opt_df.iloc[time_window:, :5]



for i in tqdm(range(int((len(energy_opt_df)-time_window)/len_forecast))):
    energy_opt = torch.Tensor(
        energy_opt_df[i*len_forecast:time_window+i*len_forecast].values)
    
    temp_opt = torch.Tensor(
        temp_opt_df[i*len_forecast:time_window+i*len_forecast].values)

    outdoor_pred = torch.Tensor(
        outdoor[i*len_forecast:(i+1)*len_forecast].values).detach()
    
    hvac_op = energy_opt[-len_forecast:,5:-1]
    print(hvac_op.shape)
    optimizer = Optimizer(forecaster_energy, forecaster_temp, lr=1e-4, epochs=100, hvac_op=hvac_op,
                          time_window=time_window, len_forecast=len_forecast, dim_hvac=len(features_temp)-len(col_out_temp)-5, temp_norm=temp_norm)
    
    
    
    trainer = pl.Trainer(max_epochs=100, accelerator='auto', callbacks = [LearningRateMonitor(logging_interval='epoch'), EarlyStopping(monitor="loss", min_delta=0.00, patience=5, verbose=False, mode="min")])
    lr_finder = trainer.tuner.lr_find(optimizer, train_dataloaders = (energy_opt.unsqueeze(0), temp_opt.unsqueeze(0), outdoor_pred.unsqueeze(0)))
    optimizer.hparams.lr = lr_finder.suggestion()
    print(lr_finder.suggestion())
    
    trainer.fit(optimizer, train_dataloaders = (energy_opt.unsqueeze(0), temp_opt.unsqueeze(0), outdoor_pred.unsqueeze(0)))
    energy_first_pred = forecaster_energy(energy_opt[-time_window:]).squeeze(0)
    temp_first_pred = forecaster_temp(temp_opt[-time_window:]).squeeze(0)
    energy_opt = torch.cat((outdoor_pred, optimizer.hvac_op, energy_first_pred), dim=1)
    temp_opt = torch.cat((outdoor_pred, optimizer.hvac_op, temp_first_pred), dim=1)
    
    energy_opt_df.iloc[time_window+i*len_forecast:time_window +
                       (i+1)*len_forecast, 5:] = energy_opt[-len_forecast:, 5:].detach().numpy()
    temp_opt_df.iloc[time_window+i*len_forecast:time_window +
                     (i+1)*len_forecast, 5:] = temp_opt[-len_forecast:, 5:].detach().numpy()


energy_opt_df.to_csv(MAIN_DIR/'results'/'opt' /
                     'optimal_input'/'energy_opt_df.csv')
temp_opt_df.to_csv(MAIN_DIR/'results'/'opt'/'optimal_input'/'temp_opt_df.csv')

energy_res_df = res_df(energy_test_set, energy_opt_df, col_out_energy)
temp_res_df = res_df(temp_test_set, temp_opt_df, col_out_temp)

energy_res_df.to_csv(MAIN_DIR/'results'/'opt'/'comparison'/'energy_res_df.csv')
temp_res_df.to_csv(MAIN_DIR/'results'/'opt'/'comparison'/'temp_res_df.csv')
