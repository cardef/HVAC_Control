import torch
import pandas as pd
import numpy as np
import tqdm
from statistics import mean
from itertools import product

class Evaluator():
    
    def __init__(self, test_loader, model, criterion, col_out):
        self.test_loader = test_loader
        self.model = model
        self.criterion = criterion
        self.col_out = col_out
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        
    def evaluation(self, original_df = None):
        self.prediction = []
        self.ground_truth = []
        self.timestamp = []
        self.loss_ = []
        self.model.eval()
        with torch.no_grad():
            with tqdm(self.test_loader, unit_size = 'batch', desc = 'Evaluation') as pbar:
                for X, Y, timestamp_y in pbar:
                    pred = self.model(X.to(self.device)).unsqueeze()
                    loss = self.criterion(pred, Y).item()
                    self.loss_.append(loss)
                    pbar.set_postfix(loss = loss)
                    self.prediction.append(pred.view(-1, pred.size(2)).detach.cpu().numpy())
                    self.ground_truth.append(Y.view(-1, Y.size(2)).detach.cpu().numpy())
                    self.timestamp.append(timestamp_y)
        self.prediction = np.cat(self.prediction, axis = 0)
        self.ground_truth = np.cat(self.ground_truth, axis = 0)
        res_matrix = np.zeros(self.ground_truth.size(0), self.ground_truth.size(0)*2)
        for i in range(self.prediction.size(1)):
            res_matrix[:,i*2] = self.ground_truth[:,i]
            res_matrix[:,i*2 + 1] = self.prediction[:,i]
        print("Loss:", self.loss_.mean())
        columns = product(self.col_out, ['true', 'pred'])
        res_df = pd.DataFrame(self.prediction, columns=columns)    
        res_df['date'] = self.timestamp
        res_df.set_index('date')
        res_df.columns = pd.MultiIndex.from_tuples(res_df.columns, names=['Variable','Type'])
        
        return res_df, self.loss_.mean()        