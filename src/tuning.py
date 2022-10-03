
from utils import MAIN_DIR
import torch
from model.cnn_enc_dec_attn import CNNEncDecAttn
from torch.utils.data import DataLoader
from data.dataset import Dataset
from pathlib import Path
from pytorch_lightning.loggers import TensorBoardLogger
import json
import os
import wandb
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from ray.air.callbacks.wandb import WandbLoggerCallback
from ray import air, tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback, TuneReportCheckpointCallback
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
from ray.tune.search.bohb import TuneBOHB
from ray.tune.search.bayesopt.bayesopt_search import BayesOptSearch
from data.dataset import collate_fn
import pandas as pd
from pickle import dump
import numpy as np

checkpoint_callback = ModelCheckpoint(
    dirpath="my/path/", save_top_k=2, monitor="val_loss")

main_dir = Path(__file__).parent.parent
with open('config.json') as f:
    config = json.load(f)

with open("cleaned_csv.json") as f:
    cleaned_csv = json.load(f)
len_forecast = config['len_forecast']
time_window = config['time_window']
col_out_temp = cleaned_csv['temp']['col_out']


energy_train_set = pd.read_csv(
    main_dir/'data'/'cleaned'/'energy'/'train_set_imp.csv')
energy_valid_set = pd.read_csv(
    main_dir/'data'/'cleaned'/'energy'/'valid_set_imp.csv')
energy_train_loader = DataLoader(Dataset(energy_train_set, time_window,
                                 len_forecast, "energy"), batch_size=64, collate_fn=collate_fn, num_workers=0)
energy_valid_loader = DataLoader(Dataset(energy_valid_set, time_window,
                                 len_forecast, "energy"), batch_size=64, collate_fn=collate_fn, num_workers=0)


temp_train_set = pd.read_csv(
    main_dir/'data'/'cleaned'/'temp'/'train_set_imp.csv')
temp_valid_set = pd.read_csv(
    main_dir/'data'/'cleaned'/'temp'/'valid_set_imp.csv')
temp_train_loader = DataLoader(Dataset(temp_train_set, time_window, len_forecast,
                               'temp'), batch_size=64, collate_fn=collate_fn, num_workers=0)
temp_valid_loader = DataLoader(Dataset(temp_valid_set, time_window, len_forecast,
                               'temp'), batch_size=64, collate_fn=collate_fn, num_workers=0)


def trainer_tuning(config, train_loader, valid_loader, num_epochs, num_gpus, log_path):
    model = CNNEncDecAttn(config)
    kwargs = {
        "max_epochs": num_epochs,
        "gpus": num_gpus,
        "enable_progress_bar": True,
        "callbacks": [
            TuneReportCheckpointCallback(
                {
                    "val_loss": "val_loss", "train_loss": "train_loss"
                },
                filename="checkpoint",
                on="epoch_end"
            ),
            EarlyStopping(monitor="val_loss", min_delta=0.00, patience=5, verbose=True, mode="min")

        ]
    }
    trainer = Trainer(
        **kwargs
    )

    if log_path:
        kwargs["resume_from_checkpoint"] = os.path.join(log_path, "checkpoint")

    trainer.fit(model, train_loader, valid_loader)


def tuner(train_loader, valid_loader, config, name, log_path):

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
            resources={"cpu": 4, "gpu": 1}
        ),
        tune_config=tune.TuneConfig(
            metric="val_loss",
            mode="min",
            scheduler=ASHAScheduler(),
            
            # search_alg= BayesOptSearch(),
            search_alg=tune.search.basic_variant.BasicVariantGenerator(),

            num_samples=100

        ),
        run_config=air.RunConfig(
            name=name,
            verbose=2,
            local_dir=log_path,
            callbacks=[
                WandbLoggerCallback(
                    api_key="86a2ba8c8e41892c3c639a87cdd7c01bd034116f", project="MPC", log_config=True, group = "name")
            ],
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
    "lr": 1e-3,
    "p_dropout_conv": tune.uniform(0, 1),
    "p_dropout_fc": tune.uniform(0, 1),
    "hidden_size_enc": tune.qloguniform(16, 1024, base=2, q=1),
    "conv_layers": tune.randint(1,4),
    "linear_layers":tune.randint(1,11),
    "conv_features" :tune.sample_from(lambda spec: 2**np.random.randint(1,10, size = spec.config.conv_layers)),
    "conv_kernels": tune.sample_from(lambda spec: 2*np.random.randint(1,5, size = spec.config.conv_layers)+1),
    "linear_neurons":tune.sample_from(lambda spec: 10*np.random.randint(1,100, size = spec.config.linear_layers))
}


config_temp = config_en.copy()
config_temp['col_out'] = len(col_out_temp)


#tuner(energy_train_loader, energy_valid_loader,
      #config_en, 'energy', main_dir/'tuning'/'energy'/'cnn_lstm')
tuner(temp_train_loader, temp_valid_loader,
      config_temp, 'temp', main_dir/'tuning'/'temp'/'cnn_lstm')
'''
forecaster_energy = CNNEncDecAttn(len_forecast,len(col_out),
                                lr = 3e-4,
                                conv_layers = [(512, 3, 1, 1)],
                                linear_layers=[250, 100, 50, 10],
                                hidden_size_enc=246,
                                scheduler_patience=5,
                                p_dropout=0.5).to(dtype = torch.float)

early_stopper = EarlyStopping(
    monitor = 'val_loss', mode = 'min', patience = 10, verbose = True)
checkpoint_callback = ModelCheckpoint(
    dirpath=main_dir/'results'/'energy'/'checkpoint', save_top_k=1, monitor="val_loss")

trainer = Trainer(accelerator='auto', default_root_dir=main_dir/'checkpoint'/'energy',
                  auto_lr_find=False, callbacks=[early_stopper, checkpoint_callback], max_epochs=50)
trainer.tune(forecaster_energy, energy_train_loader, energy_valid_loader)
trainer.fit(forecaster_energy, energy_train_loader, energy_valid_loader)

forecaster_temp = CNNEncDecAttn(len_forecast,len(col_out),
                                lr = 3e-4,
                                conv_layers = [(512, 3, 1, 1)],
                                linear_layers=[250, 100, 50, 10],
                                hidden_size_enc=246,
                                scheduler_patience=5,
                                p_dropout=0.5).to(dtype = torch.float)

checkpoint_callback = ModelCheckpoint(
    dirpath=main_dir/'results'/'temp'/'checkpoint', save_top_k=1, monitor="val_loss")
trainer = Trainer(accelerator='auto', default_root_dir=main_dir/'checkpoint'/'temp',
                  auto_lr_find=False, callbacks=[early_stopper, checkpoint_callback], max_epochs=50)
trainer.tune(forecaster_energy, temp_train_loader, temp_valid_loader)
trainer.fit(temp_train_loader, temp_valid_loader, 2)

torch.save(forecaster_energy.state_dict(),main_dir/'results'/'models'/'forecaster_energy.pt')
torch.save(forecaster_temp.state_dict(),main_dir/'results'/'models'/'forecaster_temp.pt')
'''
