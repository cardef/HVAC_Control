import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class Dataset(Dataset):
    
    def __init__(self, dataset, time_window, len_forecast, freq = '15min', col_out, imputer, train_size = 0.8, test = False):
        self.dataset = dataset
        self.time_window = time_window
        self.len_forecast = len_forecast
        self.col_out = col_out
        
        self.train_indices = np.ceil(len(dataset)*train_size).astype(int)
        self.normalization_parameters = (dataset[:self.train_indices].mean(axis = 0), 
                                         dataset[:self.train_indices].std(axis = 0))
        self.dataset_imp = imputer
        imputer.fit(dataset[self.train_indices:])

        if test:
            self.split_df = imputer.transform(dataset[self.train_indices:])
            self.split = torch.tensor(self.split_df.values).to(dtype = torch.float)
        
        else:
            self.split_df = imputer.transform(dataset[:self.train_indices])
            self.split = torch.tensor(self.split_df.values).to(dtype = torch.float)

    def __len__(self):
            return self.split-self.time_window-self.len_forecast+1
    
    def __getitem__(self, idx):
            seq_x, seq_y = self.split[idx:idx+self.time_window], self.split[idx+self.time_window: idx+self.time_window+self.len_forecast, self.col_out]
            return seq_x, seq_y