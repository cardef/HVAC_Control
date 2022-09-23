import torch
from torch.utils.data import Dataset
import numpy as np
from utils import col_out_to_index

class Dataset(Dataset):
    
    def __init__(self, dataframe, time_window, len_forecast, col_out, col_date = 'date'):
        self.dataframe = dataframe
        self.time_window = time_window
        self.len_forecast = len_forecast
        self.col_out = col_out
        self.tensor = torch.tensor(self.dataframe.drop(col_date, axis = 1).values).to(dtype = torch.float)
        self.col_date = list(dataframe[col_date])
        col_ind_dict = col_out_to_index(dataframe, col_out)
        self.col_out_ind,_=zip(*col_ind_dict.values())
    def __len__(self):
            return int((self.tensor.size(0)-self.time_window-self.len_forecast)/self.len_forecast)+1
    
    def __getitem__(self, idx):
            seq_x, seq_y = self.tensor[self.len_forecast*idx:self.len_forecast*idx+self.time_window], self.tensor[self.len_forecast*idx+self.time_window: self.len_forecast*idx+self.time_window+self.len_forecast, self.col_out_ind]
            timestamp_x, timestamp_y = self.col_date[self.len_forecast*idx:self.len_forecast*idx+self.time_window], self.col_date[self.len_forecast*idx+self.time_window: self.len_forecast*idx+self.time_window+self.len_forecast]
            return seq_x, seq_y, timestamp_x, timestamp_y
        
def collate_fn(batch):
    seq_x, seq_y, timestamp_x, timestamp_y = zip(*batch)
    seq_x = torch.stack(seq_x, dim = 0)
    seq_y = torch.stack(seq_y, dim = 0)
        
    return seq_x, seq_y, sum(timestamp_x, []), sum(timestamp_y, [])