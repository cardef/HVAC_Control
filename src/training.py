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

with open("cleaned_csv.json") as f:
    cleaned_csv = json.load(f)
len_forecast = config['len_forecast']
time_window = config['time_window']
col_out_temp = cleaned_csv['temp']['col_out']

with open(MAIN_DIR/'tuning'/'cnn_lstm'/'temp'/'best_results.pkl', 'rb') as f:
    temp_results = load(f)

with open(MAIN_DIR/'tuning'/'cnn_lstm'/'energy'/'best_results.pkl', 'rb') as f:
    energy_results = load(f)


temp_full_train_set = pd.read_csv(MAIN_DIR/'data'/'cleaned'/'temp'/'full_train_set_imp.csv')
temp_test_set = pd.read_csv(MAIN_DIR/'data'/'cleaned'/'temp'/'test_set_imp.csv')
temp_full_train_loader = DataLoader(Dataset(temp_full_train_set, time_window, len_forecast, "temp"), batch_size = 64, collate_fn = collate_fn, num_workers = 12)
temp_test_loader = DataLoader(Dataset(temp_test_set, time_window, len_forecast, "temp"), batch_size = 64, collate_fn = collate_fn, num_workers = 12)


energy_full_train_set = pd.read_csv(MAIN_DIR/'data'/'cleaned'/'energy'/'full_train_set_imp.csv')
energy_test_set = pd.read_csv(MAIN_DIR/'data'/'cleaned'/'energy'/'test_set_imp.csv')
energy_full_train_loader = DataLoader(Dataset(energy_full_train_set, time_window, len_forecast, "energy"), batch_size = 64, collate_fn = collate_fn, num_workers = 24)
energy_test_loader = DataLoader(Dataset(energy_test_set, time_window, len_forecast, "energy"), batch_size = 64, collate_fn = collate_fn, num_workers = 24)






forecaster_temp = CNNEncDecAttn(len_forecast, len(col_out_temp), temp_results.config)
trainer = Trainer(accelerator='auto',  auto_lr_find=False, max_epochs=50)
trainer.fit(forecaster_temp, temp_full_train_loader, temp_test_loader)
trainer.save_checkpoint(MAIN_DIR/'results'/'models'/'forecaster_temp.ckpt')



forecaster_energy = CNNEncDecAttn(len_forecast, 1, energy_results.config)
trainer = Trainer(accelerator='auto',  auto_lr_find=False, max_epochs=50)
trainer.fit(forecaster_energy, energy_full_train_loader, energy_test_loader)
trainer.save_checkpoint(MAIN_DIR/'results'/'models'/'forecaster_energy.ckpt')
