import torch
from torch.utils.data import Dataset
import numpy as np

class Dataset(Dataset):
    
    def __init__(self, dataframe, time_window, len_forecast, col_out, col_date = 'date'):
        self.dataframe = dataframe
        self.time_window = time_window
        self.len_forecast = len_forecast
        self.col_out = col_out

        self.tensor = torch.tensor(self.dataframe.drop(col_date, axis = 1).values).to(dtype = torch.float)
        self.col_date = np.array(dataframe[col_out], dtype='datetime64')
    def __len__(self):
            return (self.tensor.size(0)-self.time_window-self.len_forecast)%self.len_forecast+1
    
    def __getitem__(self, idx):
            seq_x, seq_y = self.tensor[self.len_forecast*idx:self.len_forecast*idx+self.time_window], self.tensor[self.len_forecast*idx+self.time_window: self.len_forecast*idx+self.time_window+self.len_forecast, self.col_out]
            timestamp_x, timestamp_y = self.col_date[self.len_forecast*idx:self.len_forecast*idx+self.time_window], self.col_date[self.len_forecast*idx+self.time_window: self.len_forecast*idx+self.time_window+self.len_forecast, self.col_out]
            return seq_x, seq_y, timestamp_x, timestamp_y