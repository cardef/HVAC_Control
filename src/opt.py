from copyreg import pickle
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from evaluator.evaluator import Evaluator
from model.cnn_enc_dec_attn import CNNEncDecAttn
from utils import MAIN_DIR, col_out_to_index
import json
from torch.utils.data import Dataset, DataLoader
from data.dataset import collate_fn
#from pythermalcomfort.models import pmv_ppd, clo_tout
#from pythermalcomfort.utilities import v_relative, clo_dynamic
#from pythermalcomfort.utilities import met_typical_tasks
from statistics import mean
from tqdm import tqdm
#from pythermalcomfort.optimized_functions import pmv_ppd_optimized
#from torchviz import make_dot, make_dot_from_trace
import math
#from torchviz import make_dot, make_dot_from_trace

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def pmv_ppd_optimized(tdb, tr, vr, rh, met, clo, wme):

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
        '''
        while abs(xn - xf) > eps:
            xf = (xf + xn) / 2
            hcn = 2.38 * abs(100.0 * xf - taa) ** 0.25
            if hcf > hcn:
                hc = hcf
            else:
                hc = hcn
            xn = (p5 + p4 * hc - p2 * xf ** 4) / (100 + p3 * hc)
            n += 1
            if n > 150:
                raise StopIteration("Max iterations exceeded")
        '''
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

    ts = 0.303 * math.exp(-0.036 * m) + 0.028
    _pmv = ts * (mw - hl1 - hl2 - hl3 - hl4 - hl5 - hl6)

    return _pmv

def loss_fn(energy, temp, vr=torch.Tensor([0.1]), rh=torch.Tensor([50]), met=torch.Tensor([1]), clo=torch.Tensor([0.6]), wme=torch.Tensor([0])):
   
    vr = vr.to(device)
    rh = rh.to(device)
    met = met.to(device)
    clo = clo.to(device)
    wme = wme.to(device)
    

    # if v_r is higher than 0.1 follow methodology ASHRAE Appendix H, H3
    ce = 0.0
    
    temp

    temp = temp.clone() - ce
    #vr = np.where(ce > 0, 0.1, vr)

    pmv_array = pmv_ppd_optimized(temp, temp, vr, rh, met, clo, wme)

    ppd_array = 100.0 - 95.0 * torch.exp(
        -0.03353 * torch.pow(pmv_array, 4.0) - 0.2179 * torch.pow(pmv_array, 2.0)
    )


    

    
    loss = torch.sum(energy) + torch.sum(torch.pow(ppd_array,2)-100)
    loss2 = torch.sum(energy)
    print(pmv_array)
    print(ppd_array)
    return loss



with open('config.json') as f:
    config = json.load(f)

with open("cleaned_csv.json") as f:
    cleaned_csv = json.load(f)
len_forecast = config['len_forecast']
time_window = config['time_window']
col_out_temp = cleaned_csv['temp']['col_out']
col_out_energy = cleaned_csv['energy']['col_out']
features_temp = cleaned_csv['temp']['features']
features_energy = cleaned_csv['energy']['features']


forecaster_energy = CNNEncDecAttn.load_from_checkpoint(
    MAIN_DIR/'results'/'models'/'forecaster_energy.ckpt').to(device)


forecaster_temp = CNNEncDecAttn.load_from_checkpoint(
    MAIN_DIR/'results'/'models'/'forecaster_temp.ckpt').to(device)


energy_test_set = pd.read_csv(
    MAIN_DIR/'data'/'cleaned'/'energy'/'test_set_imp.csv')
temp_test_set = pd.read_csv(
    MAIN_DIR/'data'/'cleaned'/'temp'/'test_set_imp.csv')
energy_test_set.set_index('date', inplace=True)
temp_test_set.set_index('date', inplace=True)
outdoor = energy_test_set.iloc[time_window:, :5]
outdoor_pred = torch.Tensor(outdoor[0*len_forecast:(0+1)*len_forecast].values).to(device).detach()
energy_opt = torch.Tensor(energy_test_set[-time_window:].values)
energy_opt.requires_grad_(True)
energy_opt.retain_grad()
temp_opt = torch.Tensor(temp_test_set[-time_window:].values)

hvac_op = torch.normal(torch.zeros(len_forecast,
    len(features_temp)-len(col_out_temp)-5), 1).to(device)
hvac_op.requires_grad_(True)
print(hvac_op.shape)
optimizer = torch.optim.Adam([hvac_op])
energy_first_pred = forecaster_energy(energy_opt[-time_window:].to(
    device)).squeeze(0)

energy_input = torch.cat((outdoor_pred, hvac_op, energy_first_pred), dim=1)
  
energy_opt2 = torch.cat(
        (energy_opt, energy_input), dim=0)

outdoor_pred.requires_grad_(True)
energy_opt2.requires_grad_(True)
energy_opt2.retain_grad()

energy_pred = forecaster_energy(energy_opt2[-time_window:].to(device))
torch.sum(energy_pred).backward()
print(energy_opt2.grad[99])
print(hvac_op.grad)