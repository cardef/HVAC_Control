from itertools import dropwhile
from torch import nn
import torch
from model.layers import attndecoder, conv1d, encoder, fcc
import pytorch_lightning as pl

class CNNEncDecAttn(pl.LightningModule):
    
    def __init__(self, config, scheduler_patience = 5,conv_layers = [(32, 3, 1, 1)], linear_layers = [500,10]):
        super(CNNEncDecAttn, self).__init__()
        self.hidden_size_enc = int(config['hidden_size_enc'])
        self.len_forecast = config['len_forecast']
        self.col_out = config['col_out']
        self.lr = config['lr']
        self.p_dropout = config['p_dropout']
        self.conv1d = conv1d.Conv1d([(int(config['conv_features']), config['kernel_size'], 1, 1)])
        self.dropout = nn.Dropout(self.p_dropout)
        self.encoder = encoder.Encoder(int(config['conv_features']), self.hidden_size_enc, 1)
        self.decoder = attndecoder.AttnDecoder(self.hidden_size_enc, self.hidden_size_enc)
        self.fcc = fcc.FCC([config['linear_layer1'], config['linear_layer2'], config['linear_layer3'], config['linear_layer4']], self.p_dropout)
        self.output = nn.LazyLinear(self.col_out)
        self.len_forecast = self.len_forecast
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