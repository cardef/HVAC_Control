import torch
import pandas as pd
import numpy as np
from statistics import mean
from itertools import product
from pytorch_lightning import Trainer

class Evaluator():
    
    def __init__(self, test_loader, model, col_out):
        self.test_loader = test_loader
        self.model = model
        self.col_out = col_out
        self.trainer = Trainer(accelerator='auto')
        
    def evaluation(self, original_df = None):
        results = self.trainer.predict(self.model, self.test_loader)
        prediction, truth, timestamp = zip(*results)
        prediction = np.concatenate(prediction, axis = 0)
        truth = np.concatenate(truth, axis = 0)
        timestamp = np.concatenate(timestamp, axis = 0)
        res_matrix = np.zeros((truth.shape[0], truth.shape[1]*2))
        for i in range(prediction.shape[1]):
            res_matrix[:,i*2] = truth[:,i]
            res_matrix[:,i*2 + 1] = prediction[:,i]
        columns = product(self.col_out, ['true', 'pred'])
        res_df = pd.DataFrame(res_matrix, columns=columns)   
        res_df['date'] = pd.to_datetime(timestamp)
        res_df = res_df.set_index('date')
        res_df.columns = pd.MultiIndex.from_tuples(res_df.columns, names=['Variable','Type'])
        return res_df      