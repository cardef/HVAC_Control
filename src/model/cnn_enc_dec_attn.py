from torch import nn
import torch
from model.layers import attndecoder, conv1d, encoder, fcc
import pytorch_lightning as pl

class CNNEncDecAttn(pl.LightningModule):
    
    def __init__(self, len_forecast, col_out, lr, p_dropout, conv_layers, linear_layers, hidden_size_enc, scheduler_patience):
        super(CNNEncDecAttn, self).__init__()
        self.conv1d = conv1d.Conv1d(conv_layers)
        self.dropout = nn.Dropout(p_dropout)
        self.encoder = encoder.Encoder(hidden_size_enc, hidden_size_enc, 1)
        self.decoder = attndecoder.AttnDecoder(hidden_size_enc, hidden_size_enc)
        self.fcc = fcc.FCC(linear_layers, p_dropout)
        self.output = nn.LazyLinear(col_out)
        self.len_forecast = len_forecast
        self.lr = lr
        self.scheduler_patience = scheduler_patience
        self.save_hyperparameters()

    def forward(self, x):
        x = self.conv1d(x)
        x = self.dropout(x)
        x, h = self.encoder(x.transpose(1,2))

        out = [None] * self.len_forecast
        out[0] = self.decoder(x[:,-1,:].squeeze(1), x)
        for i in range(1, self.len_forecast):
            out[i] = self.decoder(out[i-1], x)
        out = torch.stack(out, dim = 1)
        out = self.fcc(out)
        out = self.output(out)
        return out
    
    def training_step(self, batch, batch_idx):
        X, Y, _, _ = batch
        pred = self.forward(X)
        train_loss = nn.MSELoss()(pred, Y)
        self.log("train_loss", train_loss, prog_bar=True, on_epoch=True)
        return train_loss
    
    def validation_step(self, batch, batch_idx):
        X, Y, _, _ = batch
        pred = self.forward(X)
        val_loss = nn.MSELoss()(pred, Y)
        self.log("val_loss", val_loss, prog_bar=True, on_epoch=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        X, Y, _, _ = batch
        pred = self.forward(X)
        test_loss = nn.MSELoss()(pred, Y)
        self.log("test_loss", test_loss, prog_bar=True, on_epoch=True)
        return test_loss
    
    def predict_step(self, batch, batch_idx):
        X, Y, timestamp_x, timestamp_y = batch
        pred = self(X)
        pred = pred.view(-1, pred.size(2)).numpy()
        truth = Y.view(-1, Y.size(2)).numpy()
        
        return pred, truth, timestamp_y
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience= self.scheduler_patience, factor=0.5)
        return {'optimizer' : optimizer, 'lr_scheduler' : {'scheduler': scheduler, 'monitor':'val_loss'}}