import torch
from torch.utils.data import Dataset
import numpy as np

class Dataset(Dataset):
    
    def __init__(self, dataset, time_window, len_forecast, col_out):
        self.dataset = dataset
        self.time_window = time_window
        self.len_forecast = len_forecast
        self.col_out = col_out
        

        self.tensor = torch.tensor(self.dataset.values).to(dtype = torch.float)

    def __len__(self):
            return self.tensor.size(0)-self.time_window-self.len_forecast+1
    
    def __getitem__(self, idx):
            seq_x, seq_y = self.tensor[idx:idx+self.time_window], self.tensor[idx+self.time_window: idx+self.time_window+self.len_forecast, self.col_out]
            return seq_x, seq_y