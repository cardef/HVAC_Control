
from utils import MAIN_DIR
import torch
from model.cnn_enc_dec_attn import CNNEncDecAttn
from torch.utils.data import Dataset, DataLoader
from data.dataset import Dataset
from pathlib import Path
import json
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

checkpoint_callback = ModelCheckpoint(dirpath="my/path/", save_top_k=2, monitor="val_loss")

main_dir = Path(__file__).parent.parent
with open('config.json') as f:
    config = json.load(f)
    col_out = config['PARAMETER']['DATA']['col_temp_in'] + config['PARAMETER']['DATA']['col_temp_ext']
    len_forecast = config['PARAMETER']['DATA']['len_forecast']

energy_train_loader = torch.load(main_dir/'data'/'cleaned'/'energy'/'train_loader.pt')
energy_valid_loader = torch.load(main_dir/'data'/'cleaned'/'energy'/'valid_loader.pt')

temp_train_loader = torch.load(main_dir/'data'/'cleaned'/'temp'/'train_loader.pt')
temp_valid_loader = torch.load(main_dir/'data'/'cleaned'/'temp'/'valid_loader.pt')

forecaster_energy = CNNEncDecAttn(len_forecast,len(col_out), 
                                lr = 3e-4, 
                                conv_layers = [(512, 3, 1, 1)],
                                linear_layers=[250, 100, 50, 10],
                                hidden_size_enc=246,
                                scheduler_patience=5,
                                p_dropout=0.5).to(dtype = torch.float)

early_stopper = EarlyStopping(monitor = 'val_loss', mode = 'min', patience = 10, verbose = True)
checkpoint_callback = ModelCheckpoint(dirpath=main_dir/'results'/'energy'/'checkpoint', save_top_k=1, monitor="val_loss")

trainer = Trainer(accelerator='auto', default_root_dir=main_dir/'checkpoint'/'energy', auto_lr_find=False, callbacks=[early_stopper, checkpoint_callback], max_epochs=50)
trainer.tune(forecaster_energy, energy_train_loader, energy_valid_loader)
trainer.fit(forecaster_energy, energy_train_loader, energy_valid_loader)

forecaster_temp = CNNEncDecAttn(len_forecast,len(col_out), 
                                lr = 3e-4, 
                                conv_layers = [(512, 3, 1, 1)],
                                linear_layers=[250, 100, 50, 10],
                                hidden_size_enc=246,
                                scheduler_patience=5,
                                p_dropout=0.5).to(dtype = torch.float)

checkpoint_callback = ModelCheckpoint(dirpath=main_dir/'results'/'temp'/'checkpoint', save_top_k=1, monitor="val_loss")
trainer = Trainer(accelerator='auto', default_root_dir=main_dir/'checkpoint'/'temp', auto_lr_find=False, callbacks=[early_stopper, checkpoint_callback], max_epochs=50)
trainer.tune(forecaster_energy, temp_train_loader, temp_valid_loader)
trainer.fit(temp_train_loader, temp_valid_loader, 2)

torch.save(forecaster_energy.state_dict(),main_dir/'results'/'models'/'forecaster_energy.pt')
torch.save(forecaster_temp.state_dict(),main_dir/'results'/'models'/'forecaster_temp.pt')