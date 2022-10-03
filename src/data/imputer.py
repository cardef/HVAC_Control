from data.matrix_factorization import MatrixFactorization
import torch
import numpy as np
from pytorch_lightning.trainer import Trainer
import pandas as pd


class Imputer():
    def __init__(self, df, n_features, n_epochs, lr, penalty):
        self.features_name = df.columns
        self.df = df
        self.min = df.min(axis=0)
        self.max = df.max(axis=0)
        norm = self.max-self.min
        norm.where(norm !=0, 1, inplace=True)
        self.df_scaled = (df-self.min)/norm
        self.matrix = np.array(self.df_scaled)
        self.train_matrix = self.matrix
        self.test_matrix = np.empty(
            (self.matrix.shape[0], self.matrix.shape[1]))
        self.test_matrix[:] = np.nan
        random_ind = (np.random.choice(self.matrix.shape[0], int(
            self.matrix.size*0.3), replace=True), np.random.choice(self.matrix.shape[1], int(self.matrix.size*0.3), replace=True))
        self.test_matrix[random_ind] = self.matrix[random_ind]
        self.train_matrix[random_ind] = np.nan
        self.matrix = torch.Tensor(self.matrix)
        self.train_matrix = torch.Tensor(self.train_matrix)
        self.test_matrix = torch.Tensor(self.test_matrix)
        self.model = MatrixFactorization(self.train_matrix, n_features)
        self.lr = lr
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=self.optimizer, patience=20, factor=0.5)
        self.n_epochs = n_epochs
        
        self.penalty = penalty

    def mse_loss_with_nans(self, input, target):

        # Missing data are nan's
        mask = torch.isnan(target)

        # Missing data are 0's
        #mask = target == 0
        reg = 0
        for param in self.model.parameters():
            reg += 0.5 * (param ** 2).sum()
        loss = ((input[~mask]-target[~mask]) **
                2).mean().sqrt() + self.penalty*reg
        return loss

    def impute(self):

        for i in range(self.n_epochs):

            # Set gradients to zero
            self.optimizer.zero_grad()

            # Predict and calculate loss
            prediction = self.model(self.train_matrix).to(self.device)
            train_loss = self.mse_loss_with_nans(
                prediction, self.train_matrix.to(self.device))

            # Backpropagate
            train_loss.backward()

            # Update the parameters
            self.optimizer.step()

            with torch.no_grad():
                prediction = self.model(self.test_matrix).to(self.device)
                test_loss = self.mse_loss_with_nans(
                    prediction, self.test_matrix.to(self.device))

                
            self.scheduler.step(test_loss.item())
        print(train_loss.item(), test_loss.item())
        matrix_pred = self.model(self.matrix)
        matrix_imputed = torch.where(torch.isnan(
            self.matrix), matrix_pred.to("cpu"), self.matrix)
        matrix_imputed = pd.DataFrame(
            matrix_imputed.detach().cpu().numpy(), columns=self.features_name, index=self.df.index)
        matrix_imputed = matrix_imputed*(self.max-self.min) + self.min
    

        del self.matrix
        del self.train_matrix
        del self.test_matrix

        torch.cuda.empty_cache()

        return matrix_imputed
