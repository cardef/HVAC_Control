import torch
import pandas as pd
import numpy as np
import tqdm
from statistics import mean
from itertools import product

class Evaluator():
    
    def __init__(self, test_loader, preprocessor, model, criterion, col_out):
        self.test_loader = test_loader
        self.model = model
        self.criterion = criterion
        self.col_out = col_out
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        
        self.col_out_ind = test_loader.dataset.col_out_ind
        self.mean = np.take(np.array(preprocessor.mean), self.col_out_ind)
        self.std = np.take(np.array(preprocessor.std), self.col_out_ind)
        
    def evaluation(self, original_df = None):
        self.prediction = []
        self.ground_truth = []
        self.timestamp = []
        self.loss_ = []
        self.model.eval()
        with torch.no_grad():
            with tqdm.tqdm(self.test_loader, unit = 'batch', desc = 'Evaluation') as pbar:
                for X, Y,_ ,timestamp_y in pbar:
                    pred = self.model(X.to(self.device))
                    loss = self.criterion(pred, Y.to(self.device)).item()
                    self.loss_.append(loss)
                    pbar.set_postfix(loss = loss)
                    self.prediction.append(pred.view(-1, pred.size(2)).detach().cpu().numpy())
                    self.ground_truth.append(Y.view(-1, Y.size(2)).detach().cpu().numpy())
                    self.timestamp.extend(timestamp_y)
        self.prediction = np.concatenate(self.prediction, axis = 0)*self.std + self.mean
        self.ground_truth = np.concatenate(self.ground_truth, axis = 0)*self.std + self.mean
        res_matrix = np.zeros((self.ground_truth.shape[0], self.ground_truth.shape[1]*2))
        for i in range(self.prediction.shape[1]):
            res_matrix[:,i*2] = self.ground_truth[:,i]
            res_matrix[:,i*2 + 1] = self.prediction[:,i]
        print("Loss:", mean(self.loss_))
        columns = product(self.col_out, ['true', 'pred'])
        res_df = pd.DataFrame(res_matrix, columns=columns)
        res_df['date'] = pd.to_datetime(self.timestamp)
        res_df = res_df.set_index('date')
        res_df.columns = pd.MultiIndex.from_tuples(res_df.columns, names=['Variable','Type'])
        
        return res_df, mean(self.loss_)        