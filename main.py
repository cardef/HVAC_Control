
import training
import torch
from model import model
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader


class DatasetSeq(Dataset):
    
    def __init__(self, dataset):
        self.dataset = dataset
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]

energy_train_loader = torch.load(r'C:\Users\cdellefemine\Documents\GitHub\HVAC-Control\dataset\energy\train_loader.pt')
energy_valid_loader = torch.load(r'C:\Users\cdellefemine\Documents\GitHub\HVAC-Control\dataset\energy\valid_loader.pt')

forecaster_energy = model.TimeSeriesForecastingModel(4,1).to(dtype = torch.float)

loss_fn = nn.MSELoss()
optimizer = Adam(forecaster_energy.parameters())
scheduler = ReduceLROnPlateau(optimizer, patience = 7)
trainer = training.Trainer(forecaster_energy, loss_fn, optimizer, scheduler, logger_kwargs = None)

trainer.fit(energy_train_loader, energy_valid_loader, 10)