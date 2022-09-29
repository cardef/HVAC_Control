import torch
import pytorch_lightning as pl
import numpy as np

class MatrixFactorization(torch.nn.Module):
    def __init__(self, matrix, n_factors=20):
        super().__init__()
        self.matrix = matrix
        self.n_rows = self.matrix.size(0)
        self.n_cols = self.matrix.size(1)
        self.n_factors = n_factors
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.rows_factors = torch.nn.Embedding(self.n_rows, self.n_factors, sparse=True).to(self.device)
        self.cols_factors = torch.nn.Embedding(self.n_cols, self.n_factors, sparse=True).to(self.device)
        self.user_biases = torch.nn.Embedding(self.n_rows, 1, sparse=True).to(self.device)
        self.item_biases = torch.nn.Embedding(self.n_cols, 1, sparse=True).to(self.device)
    def forward(self, matrix):
        matrix = matrix.to(self.device)
        rows = torch.arange(matrix.size(0)).to(self.device)
        cols = torch.arange(matrix.size(1)).to(self.device)
        
        pred = torch.matmul(self.rows_factors(rows) + self.user_biases(rows), (self.cols_factors(cols) + self.item_biases(cols)).transpose(1,0) )

        return pred