from model.cnn_enc_dec_attn import CNNEncDecAttn
import pytorch_lightning as pl
import torch

class Optimizer(pl.LightningModule):
    
    def __init__(self, model_energy, model_temp, lr, epochs, hvac_op, time_window, len_forecast, dim_hvac,temp_norm):
        super(Optimizer, self).__init__()
        self.lr = lr
        self.epochs = epochs
        self.model_energy = model_energy
        self.model_temp = model_temp
        self.time_window = time_window
        self.len_forecast = len_forecast
        self.temp_norm = temp_norm
        self.energy_first_pred = 0
        self.temp_first_pred = 0
        #self.hvac_op = hvac_op
        self.hvac_op  = torch.normal(torch.zeros(len_forecast, dim_hvac), 2.5).to(self.device)
        
        
        
    def get_predictions(self, energy_historical, temp_historical, outdoor_pred):
        self.energy_first_pred = self.model_energy(energy_historical[-self.time_window:]).squeeze(0).to(self.device)
        self.temp_first_pred = self.model_temp(temp_historical[-self.time_window:]).squeeze(0).to(self.device)
    
        energy_input = torch.cat((outdoor_pred, self.hvac_op, self.energy_first_pred), dim=1).to(self.device)
        
        energy_opt = torch.cat(
            (energy_historical[:-self.len_forecast], energy_input), dim=0).to(self.device)
        

        temp_input = torch.cat((outdoor_pred, self.hvac_op, self.temp_first_pred), dim=1).to(self.device)
        temp_opt = torch.cat((temp_historical, temp_input), dim=0).to(self.device)
        
        
        
        energy_pred = self.model_energy(energy_opt[-self.time_window:]).squeeze().to(self.device)
        temp_pred = self.model_temp(temp_opt[-self.time_window:]).squeeze().to(self.device)
        '''
        energy_pred.requires_grad_(True)
        energy_pred.retain_grad()
        energy_opt.requires_grad_(True)
        energy_opt.retain_grad()
        temp_pred.requires_grad_(True)
        '''
        return energy_pred, temp_pred
        
    def pmv_ppd_optimized(self, tdb, tr, vr, rh, met, clo, wme):

        pa = rh * 10 * torch.exp(16.6536 - 4030.183 / (tdb + 235))

        icl = 0.155 * clo  # thermal insulation of the clothing in M2K/W
        m = met * 58.15  # metabolic rate in W/M2
        w = wme * 58.15  # external work in W/M2
        mw = m - w  # internal heat production in the human body
        # calculation of the clothing area factor
        if icl <= 0.078:
            f_cl = 1 + (1.29 * icl)  # ratio of surface clothed body over nude body
        else:
            f_cl = 1.05 + (0.645 * icl)

        # heat transfer coefficient by forced convection
        hcf = 12.1 * torch.sqrt(vr)
        hc = hcf  # initialize variable
        taa = tdb + 273
        tra = tr + 273
        t_cla = taa + (35.5 - tdb) / (3.5 * icl + 0.1)

        p1 = icl * f_cl
        p2 = p1 * 3.96
        p3 = p1 * 100
        p4 = p1 * taa
        p5 = (308.7 - 0.028 * mw) + (p2 * (tra / 100.0) ** 4)
        xn = t_cla / 100
        xf = t_cla / 50
        eps = 0.00015

        n = 0
        
        mask = (torch.abs(xn-xf) > eps).clone().detach()
        #print(mask.shape, xf.shape, xn.shape, eps.shape)
        while mask.sum() > 0:
            xf = torch.where(mask,(xf + xn) / 2, xf)
            hcn = 2.38 * abs(100.0 * xf - taa) ** 0.25
            hc = torch.where(hcf > hcn, hcf, hcn)
            xn = torch.where(mask, (p5 + p4 * hc - p2 * torch.pow(xf,4)) / (100 + p3 * hc), xn)
            mask = (torch.abs(xn-xf) > eps).clone().detach()
            n += 1
            if n > 150:
                break
           
        tcl = 100 * xn - 273

        # heat loss diff. through skin
        hl1 = 3.05 * 0.001 * (5733 - (6.99 * mw) - pa)
        # heat loss by sweating
        if mw > 58.15:
            hl2 = 0.42 * (mw - 58.15)
        else:
            hl2 = 0
        # latent respiration heat loss
        hl3 = 1.7 * 0.00001 * m * (5867 - pa)
        # dry respiration heat loss
        hl4 = 0.0014 * m * (34 - tdb)
        # heat loss by radiation
        hl5 = 3.96 * f_cl * (xn ** 4 - (tra / 100.0) ** 4)
        # heat loss by convection
        hl6 = f_cl * hc * (tcl - tdb)

        ts = 0.303 * torch.exp(-0.036 * m) + 0.028
        _pmv = ts * (mw - hl1 - hl2 - hl3 - hl4 - hl5 - hl6)

        return _pmv
    
    
    def loss_fn(self, energy, temp, vr=torch.Tensor([0.1]), rh=torch.Tensor([50]), met=torch.Tensor([1]), clo=torch.Tensor([0.7]), wme=torch.Tensor([0])):
    
        temp = torch.where(temp < 0, temp * 9/5 + 32, (temp - 32) * 5/9)
        
        

        # if v_r is higher than 0.1 follow methodology ASHRAE Appendix H, H3
        ce = 0.0
        
        temp

        temp = temp.clone() - ce
        #vr = np.where(ce > 0, 0.1, vr)

        pmv_array = self.pmv_ppd_optimized(temp, temp, vr, rh, met, clo, wme)

        ppd_array = 100.0 - 95.0 * torch.exp(
            -0.03353 * torch.pow(pmv_array, 4.0) - 0.2179 * torch.pow(pmv_array, 2.0)
        )

        
        loss = torch.mean(energy) + 0*torch.mean(torch.abs(ppd_array)-10)
        
        return loss
    
    
    def training_step(self, batch, batch_idx):
        (energy_historical, temp_historical, outdoor_pred) = batch
        energy_historical=energy_historical.to(self.device)
        temp_historical = temp_historical.to(self.device)
        outdoor_pred = outdoor_pred.to(self.device)
        energy_pred, temp_pred = self.get_predictions(energy_historical, temp_historical, outdoor_pred)
        loss = self.loss_fn(energy_pred, temp_pred*(torch.Tensor(self.temp_norm[1]))+torch.Tensor(self.temp_norm[0]))
        cur_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("loss", loss, prog_bar=True, on_epoch=True)
        self.log("lr", cur_lr, prog_bar=True, on_epoch=True)
        
        return loss
    
    def configure_optimizers(self):
        optim = torch.optim.SGD([self.hvac_op], lr = self.lr, momentum = 0.9)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optim, patience= 4, factor=0.1)
        return {'optimizer' : optim, 'lr_scheduler' : {'scheduler': scheduler, 'monitor':'loss', "interval": "epoch", "frequency":1}}
    
    def on_before_zero_grad(self, *args, **kwargs):
        self.hvac_op = torch.clamp(self.hvac_op, -2,+2)