
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
from ray.tune.integration.pytorch_lightning import TuneReportCallback, TuneReportCheckpointCallback
from ray.tune.schedulers import ASHAScheduler
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
from ray.tune.search.bohb import TuneBOHB
from ray.tune.search.bayesopt.bayesopt_search import BayesOptSearch
from data.dataset import collate_fn
import pandas as pd
from pickle import dump
checkpoint_callback = ModelCheckpoint(
    dirpath="my/path/", save_top_k=2, monitor="val_loss")

main_dir = Path(__file__).parent.parent
with open('config.json') as f:
    config = json.load(f)
    col_out = config['PARAMETER']['DATA']['col_temp_in'] + \
        config['PARAMETER']['DATA']['col_temp_ext']
    len_forecast = config['PARAMETER']['DATA']['len_forecast']
    time_window = config['PARAMETER']['DATA']['time_window']

energy_train_set = pd.read_csv(
    main_dir/'data'/'cleaned'/'energy'/'train_set_imp.csv')
energy_valid_set = pd.read_csv(
    main_dir/'data'/'cleaned'/'energy'/'valid_set_imp.csv')
energy_train_loader = DataLoader(Dataset(energy_train_set, time_window, len_forecast, [
                                 'hvac']), batch_size=64, collate_fn=collate_fn, num_workers=14)
energy_valid_loader = DataLoader(Dataset(energy_valid_set, time_window, len_forecast, [
                                 'hvac']), batch_size=64, collate_fn=collate_fn, num_workers=14)


temp_train_set = pd.read_csv(
    main_dir/'data'/'cleaned'/'temp'/'train_set_imp.csv')
temp_valid_set = pd.read_csv(
    main_dir/'data'/'cleaned'/'temp'/'valid_set_imp.csv')
temp_train_loader = DataLoader(Dataset(temp_train_set, time_window, len_forecast,
                               col_out), batch_size=64, collate_fn=collate_fn, num_workers=14)
temp_valid_loader = DataLoader(Dataset(temp_valid_set, time_window, len_forecast,
                               col_out), batch_size=64, collate_fn=collate_fn, num_workers=14)


def trainer_tuning(config, train_loader, valid_loader, num_epochs, num_gpus, log_path):
    model = CNNEncDecAttn(config)
    trainer = Trainer(
        max_epochs=num_epochs,
        gpus=num_gpus,
        enable_progress_bar=True,
        logger=TensorBoardLogger(
            save_dir=(log_path/'tb'), name="", version="."),
        callbacks=[
            TuneReportCallback(
                {
                    "loss": "val_loss"
                },
                on="epoch_end"
            )
        ]
    )
    trainer.fit(model, train_loader, valid_loader)


def tuner(train_loader, valid_loader, config, log_path):

    num_epochs = 100

    train_fn_with_parameters = tune.with_parameters(trainer_tuning,
                                                    train_loader=train_loader,
                                                    valid_loader=valid_loader,
                                                    num_epochs=num_epochs,
                                                    num_gpus=1,
                                                    log_path=log_path
                                                    )

    tuner = tune.Tuner(
        tune.with_resources(
            train_fn_with_parameters,
            resources={"cpu": 14, "gpu": 1}
        ),
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            scheduler=ASHAScheduler(),
            #search_alg=tune.search.basic_variant.BasicVariantGenerator(),
            search_alg= BayesOptSearch(),
            num_samples=10

        ),
        run_config=air.RunConfig(
            name="tuning",
            verbose=3,
            local_dir=log_path
        ),
        param_space=config,
    )

    results = tuner.fit()

    print("Best hyperparameters found were: ",
          results.get_best_result().config)
    with open(log_path/'best_results.pkl', 'wb') as f:
        dump(results.get_best_result(), f)


config_en = {
    "len_forecast": len_forecast,
    "col_out": 1,
    "lr": tune.loguniform(1e-5, 1e-1),
    "p_dropout": tune.uniform(0, 1),
    "hidden_size_enc": tune.qloguniform(16, 1024, base=2, q=1),
    "conv_features": tune.qloguniform(16, 1024, base=2, q=1),
    "kernel_size": tune.choice([3, 5, 7, 9, 11]),
    "linear_layer1": tune.qrandint(10, 1000, q=10),
    "linear_layer2": tune.qrandint(10, 1000, q=10),
    "linear_layer3": tune.qrandint(10, 1000, q=10),
    "linear_layer4": tune.qrandint(10, 1000, q=10)
}


config_temp = config_en
config_temp['col_out'] = len(col_out)


tuner(energy_train_loader, energy_valid_loader,
      config_en, main_dir/'tuning'/'energy'/'cnn_lstm')
tuner(temp_train_loader, temp_valid_loader,
      config_temp, main_dir/'tuning'/'temp'/'cnn_lstm')
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
