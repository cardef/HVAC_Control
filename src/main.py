
from utils import MAIN_DIR
import trainer.training as training
import torch
from model.forecaster_energy import ForecasterEnergy
from model.forecaster_temp import ForecasterTemp
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from data.dataset import Dataset
from pathlib import Path
import json

main_dir = Path(__file__).parent.parent
with open('config.json') as f:
    config = json.load(f)
    col_out = config['PARAMETER']['DATA']['col_temp_in'] + config['PARAMETER']['DATA']['col_temp_ext']
    len_forecast = config['PARAMETER']['DATA']['len_forecast']

energy_train_loader = torch.load(main_dir/'data'/'cleaned'/'energy'/'train_loader.pt')
energy_valid_loader = torch.load(main_dir/'data'/'cleaned'/'energy'/'valid_loader.pt')

forecaster_energy = ForecasterEnergy(len_forecast).to(dtype = torch.float)

loss_fn = nn.MSELoss()
optimizer = Adam(forecaster_energy.parameters(), lr=0.0003)
scheduler = ReduceLROnPlateau(optimizer, patience = 5, factor=0.5, verbose = True)
#trainer = training.Trainer(forecaster_energy, loss_fn, optimizer, scheduler, logger_kwargs = None)

#trainer.fit(energy_train_loader, energy_valid_loader, 50)


temp_train_loader = torch.load(main_dir/'data'/'cleaned'/'temp'/'train_loader.pt')
temp_valid_loader = torch.load(main_dir/'data'/'cleaned'/'temp'/'valid_loader.pt')

forecaster_temp = ForecasterTemp(len_forecast,len(col_out)).to(dtype = torch.float)

loss_fn = nn.MSELoss()
optimizer = Adam(forecaster_temp.parameters(), lr = 3e-4)
scheduler = ReduceLROnPlateau(optimizer, patience = 5, factor=0.5, verbose = True)
trainer = training.Trainer(forecaster_temp, loss_fn, optimizer, scheduler, logger_kwargs = None)

trainer.fit(temp_train_loader, temp_valid_loader, 10)

#torch.save(forecaster_energy.state_dict(),main_dir/'results'/'models'/'forecaster_energy.pt')
torch.save(forecaster_temp.state_dict(),main_dir/'results'/'models'/'forecaster_temp.pt')