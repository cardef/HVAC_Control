from torch.utils.data import Dataset, DataLoader

class Dataset(Dataset):
    
    def __init__(self, dataset, time_window, len_forecast, col_out):
        self.dataset = dataset
        self.time_window = time_window
        self.len_forecast = len_forecast
        self.col_out = col_out
        self.last_timestep = dataset.size(0)
    def __len__(self):
        return self.last_timestep-self.time_window-self.len_forecast+1
    
    def __getitem__(self, idx):
        seq_x, seq_y = self.dataset[idx:idx+self.time_window], self.dataset[idx+self.time_window: idx+self.time_window+self.len_forecast, self.col_out]
        return seq_x, seq_y