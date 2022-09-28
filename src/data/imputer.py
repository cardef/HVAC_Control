from data.matrix_factorization import MatrixFactorization
import torch
import numpy as np
from pytorch_lightning.trainer import Trainer
import pandas as pd

class Imputer():
    def __init__(self, df, col_to_ignore, n_features, n_epochs):
        self.features_name = df.drop(col_to_ignore, axis = 1).columns
        self.df = df
        self.mean =  df.drop(col_to_ignore, axis = 1).mean(axis = 0)
        self.std =  df.drop(col_to_ignore, axis = 1).std(axis = 0)
        self.df_scaled = (df.drop(col_to_ignore, axis = 1)-self.mean)/self.std
        self.matrix =  torch.Tensor(np.array(self.df_scaled))
        self.col_to_ignore = col_to_ignore
        self.model = MatrixFactorization(self.matrix, n_features)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.optimizer = torch.optim.SparseAdam(self.model.parameters(), lr=0.1)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer, patience= 100, factor=0.5)
        self.n_epochs = n_epochs
    def mse_loss_with_nans(self, input, target):

        # Missing data are nan's
        mask = torch.isnan(target)

        # Missing data are 0's
        #mask = target == 0

        out = (input[~mask]-target[~mask])**2
        loss = out.mean()

        return loss

    def impute(self):

        for i in range(self.n_epochs):
        
            # Set gradients to zero
            self.optimizer.zero_grad()
            
           
            # Predict and calculate loss
            prediction = self.model().to(self.device)
            loss = self.mse_loss_with_nans(prediction, self.matrix.to(self.device))
            print(loss.item())

            # Backpropagate
            loss.backward()

            # Update the parameters
            self.optimizer.step()
            self.scheduler.step(loss.item())

        matrix_pred = self.model()
        matrix_imputed = torch.where(torch.isnan(self.matrix), matrix_pred.to("cpu"), self.matrix)
        matrix_imputed = pd.DataFrame(matrix_imputed.detach().cpu().numpy(), columns=self.features_name)
        matrix_imputed = matrix_imputed*self.std + self.mean
        matrix_imputed[self.col_to_ignore] = self.df[self.col_to_ignore]
        return matrix_imputed