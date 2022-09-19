
import trainer.training as training
import torch
from model.forecaster_energy import ForecasterEnergy
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from data_loader.dataset import Dataset
from pathlib import Path

main_dir = Path(__file__).parent

energy_train_loader = torch.load(main_dir/'dataset'/'energy'/'train_loader.pt')
energy_valid_loader = torch.load(main_dir/'dataset'/'energy'/'valid_loader.pt')

forecaster_energy = ForecasterEnergy(4,1).to(dtype = torch.float)

loss_fn = nn.MSELoss()
optimizer = Adam(forecaster_energy.parameters())
scheduler = ReduceLROnPlateau(optimizer, patience = 7)
trainer = training.Trainer(forecaster_energy, loss_fn, optimizer, scheduler, logger_kwargs = None)

trainer.fit(energy_train_loader, energy_valid_loader, 10)