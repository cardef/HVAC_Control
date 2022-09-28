from copyreg import pickle
from pickle import load
from utils import MAIN_DIR
from model.cnn_enc_dec_attn import CNNEncDecAttn
from pytorch_lightning.trainer import Trainer
import pandas as pd
from torch.utils.data import DataLoader
from data.dataset import Dataset, collate_fn
import torch
import json

with open('config.json') as f:
    config = json.load(f)
    
col_out = config['PARAMETER']['DATA']['col_temp_in'] + config['PARAMETER']['DATA']['col_temp_ext']
len_forecast = config['PARAMETER']['DATA']['len_forecast']
time_window = config['PARAMETER']['DATA']['time_window']

with open(MAIN_DIR/'tuning'/'temp'/'cnn_lstm'/'best_results.pkl', 'rb') as f:
    temp_results = load(f)

with open(MAIN_DIR/'tuning'/'energy'/'cnn_lstm'/'best_results.pkl', 'rb') as f:
    energy_results = load(f)


temp_full_train_set = pd.read_csv(MAIN_DIR/'data'/'cleaned'/'temp'/'full_train_set_imp.csv')
temp_test_set = pd.read_csv(MAIN_DIR/'data'/'cleaned'/'temp'/'test_set_imp.csv')
temp_full_train_loader = DataLoader(Dataset(temp_full_train_set, time_window, len_forecast, col_out), batch_size = 64, collate_fn = collate_fn, num_workers = 14)
temp_test_loader = DataLoader(Dataset(temp_test_set, time_window, len_forecast, col_out), batch_size = 64, collate_fn = collate_fn, num_workers = 14)


energy_full_train_set = pd.read_csv(MAIN_DIR/'data'/'cleaned'/'energy'/'full_train_set_imp.csv')
energy_test_set = pd.read_csv(MAIN_DIR/'data'/'cleaned'/'energy'/'test_set_imp.csv')
energy_full_train_loader = DataLoader(Dataset(energy_full_train_set, time_window, len_forecast, ['hvac']), batch_size = 64, collate_fn = collate_fn, num_workers = 14)
energy_test_loader = DataLoader(Dataset(energy_test_set, time_window, len_forecast, ['hvac']), batch_size = 64, collate_fn = collate_fn, num_workers = 14)

energy_results.config['col_out'] = 1




forecaster_temp = CNNEncDecAttn(temp_results.config)
trainer = Trainer(accelerator='auto',  auto_lr_find=False, max_epochs=5)
trainer.fit(forecaster_temp, temp_full_train_loader, temp_test_loader)
trainer.save_checkpoint(MAIN_DIR/'results'/'models'/'forecaster_temp.ckpt')



forecaster_energy = CNNEncDecAttn(energy_results.config)
trainer = Trainer(accelerator='auto',  auto_lr_find=False, max_epochs=5)
trainer.fit(forecaster_energy, energy_full_train_loader, energy_test_loader)
trainer.save_checkpoint(MAIN_DIR/'results'/'models'/'forecaster_energy.ckpt')
