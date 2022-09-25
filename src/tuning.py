
from utils import MAIN_DIR
import torch
from model.cnn_enc_dec_attn import CNNEncDecAttn
from torch.utils.data import Dataset, DataLoader
from data.dataset import Dataset
from pathlib import Path
from pytorch_lightning.loggers import TensorBoardLogger
import json
import os
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from ray import air, tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune.schedulers import ASHAScheduler
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
from ray.tune.search.bohb import TuneBOHB
from data.dataset import collate_fn
import pandas as pd
checkpoint_callback = ModelCheckpoint(dirpath="my/path/", save_top_k=2, monitor="val_loss")

main_dir = Path(__file__).parent.parent
with open('config.json') as f:
    config = json.load(f)
    col_out = config['PARAMETER']['DATA']['col_temp_in'] + config['PARAMETER']['DATA']['col_temp_ext']
    len_forecast = config['PARAMETER']['DATA']['len_forecast']
    time_window = config['PARAMETER']['DATA']['time_window']

energy_train_set = pd.read_csv(main_dir/'data'/'cleaned'/'energy'/'train_set_imp.csv')
energy_valid_set =pd.read_csv(main_dir/'data'/'cleaned'/'energy'/'valid_set_imp.csv')
energy_train_loader = DataLoader(Dataset(energy_train_set[:100], time_window, len_forecast, ['hvac']), batch_size = 64, collate_fn = collate_fn, num_workers = 2)
energy_valid_loader = DataLoader(Dataset(energy_valid_set[:100], time_window, len_forecast, ['hvac']), batch_size = 64, collate_fn = collate_fn, num_workers = 2)

sum = 0
for a,b,c,d in energy_train_loader:
    sum += 1
print(sum)

def trainer_tuning(config, train_loader, valid_loader, num_epochs, num_gpus):
    model = CNNEncDecAttn(config)
    trainer = Trainer(
        max_epochs=num_epochs,
        gpus=num_gpus,
        logger=TensorBoardLogger(
            save_dir=os.getcwd(), name="", version="."),
        enable_progress_bar=True,
        callbacks=[
            TuneReportCallback(
                {
                    "loss": "val_loss"
                },
                on="validation_end")
        ])
    trainer.fit(model,train_loader, valid_loader)

def tuner(train_loader, valid_loader, config):

    num_epochs = 2

    

    train_fn_with_parameters = tune.with_parameters(trainer_tuning,
                                                train_loader = train_loader,
                                                valid_loader = valid_loader,
                                                num_epochs=num_epochs,
                                                num_gpus=1
                                                )


    tuner = tune.Tuner(
        tune.with_resources(
            train_fn_with_parameters,
            resources= {"gpu": 1}
        ),
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            scheduler = ASHAScheduler(
                max_t=num_epochs,
                grace_period=1,
                reduction_factor=2),
            search_alg = tune.search.basic_variant.BasicVariantGenerator(),
            #search_alg= TuneBOHB(),
            num_samples=2
            
        ),
        run_config=air.RunConfig(
            name="tuning",
            verbose = 3
        ),
        param_space=config,
    )

    results = tuner.fit()

    print("Best hyperparameters found were: ", results.get_best_result().config)

config = {
        "len_forecast" : len_forecast,
        "col_out" : 1,
        "lr": tune.loguniform(1e-5, 1e-1),
        "p_dropout": tune.uniform(0,1),
        "hidden_size_enc": tune.qloguniform(16, 32, base = 2, q =1)
    }

tuner(energy_train_loader, energy_valid_loader, config)
'''
forecaster_energy = CNNEncDecAttn(len_forecast,len(col_out), 
                                lr = 3e-4, 
                                conv_layers = [(512, 3, 1, 1)],
                                linear_layers=[250, 100, 50, 10],
                                hidden_size_enc=246,
                                scheduler_patience=5,
                                p_dropout=0.5).to(dtype = torch.float)

early_stopper = EarlyStopping(monitor = 'val_loss', mode = 'min', patience = 10, verbose = True)
checkpoint_callback = ModelCheckpoint(dirpath=main_dir/'results'/'energy'/'checkpoint', save_top_k=1, monitor="val_loss")

trainer = Trainer(accelerator='auto', default_root_dir=main_dir/'checkpoint'/'energy', auto_lr_find=False, callbacks=[early_stopper, checkpoint_callback], max_epochs=50)
trainer.tune(forecaster_energy, energy_train_loader, energy_valid_loader)
trainer.fit(forecaster_energy, energy_train_loader, energy_valid_loader)

forecaster_temp = CNNEncDecAttn(len_forecast,len(col_out), 
                                lr = 3e-4, 
                                conv_layers = [(512, 3, 1, 1)],
                                linear_layers=[250, 100, 50, 10],
                                hidden_size_enc=246,
                                scheduler_patience=5,
                                p_dropout=0.5).to(dtype = torch.float)

checkpoint_callback = ModelCheckpoint(dirpath=main_dir/'results'/'temp'/'checkpoint', save_top_k=1, monitor="val_loss")
trainer = Trainer(accelerator='auto', default_root_dir=main_dir/'checkpoint'/'temp', auto_lr_find=False, callbacks=[early_stopper, checkpoint_callback], max_epochs=50)
trainer.tune(forecaster_energy, temp_train_loader, temp_valid_loader)
trainer.fit(temp_train_loader, temp_valid_loader, 2)

torch.save(forecaster_energy.state_dict(),main_dir/'results'/'models'/'forecaster_energy.pt')
torch.save(forecaster_temp.state_dict(),main_dir/'results'/'models'/'forecaster_temp.pt')
'''