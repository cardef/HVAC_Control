
from utils import MAIN_DIR
import torch
from model.cnn_enc_dec_attn import CNNEncDecAttn
from model.seq2seq_attn import Seq2SeqAttn
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
from ray.tune.schedulers.pb2 import PB2

import numpy as np

checkpoint_callback = ModelCheckpoint(
    dirpath="my/path/", save_top_k=2, monitor="val_loss")

main_dir = Path(__file__).parent.parent
with open('config.json') as f:
    config_json = json.load(f)

with open("cleaned_csv.json") as f:
    cleaned_csv = json.load(f)

len_forecast = config_json['len_forecast']
time_window = config_json['time_window']
col_out_temp = cleaned_csv['temp']['col_out']


energy_train_set = pd.read_csv(
    main_dir/'data'/'cleaned'/'energy'/'train_set_imp.csv')
energy_valid_set = pd.read_csv(
    main_dir/'data'/'cleaned'/'energy'/'valid_set_imp.csv')
energy_train_loader = DataLoader(Dataset(energy_train_set, time_window,
                                 config_json["len_forecast"], "energy"), batch_size=64, collate_fn=collate_fn, num_workers=24)
energy_valid_loader = DataLoader(Dataset(energy_valid_set, time_window,
                                 config_json["len_forecast"], "energy"), batch_size=64, collate_fn=collate_fn, num_workers=24)


temp_train_set = pd.read_csv(
    main_dir/'data'/'cleaned'/'temp'/'train_set_imp.csv')
temp_valid_set = pd.read_csv(
    main_dir/'data'/'cleaned'/'temp'/'valid_set_imp.csv')
temp_train_ds = Dataset(temp_train_set, time_window, config_json["len_forecast"],
                               'temp')
temp_valid_ds = Dataset(temp_valid_set, time_window, config_json["len_forecast"],
                               'temp')
temp_train_loader = DataLoader(temp_train_ds, batch_size=64, collate_fn=collate_fn, num_workers=24)
temp_valid_loader = DataLoader(temp_valid_ds, batch_size=64, collate_fn=collate_fn, num_workers=24)


def trainer_tuning(config,model_class,  train_loader, valid_loader, num_epochs, num_gpus, log_path, kwargs_model):
    
    model = model_class(config = config, **kwargs_model)
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
                on="validation_end"
            ),
            EarlyStopping(monitor="val_loss", min_delta=0.00,
                          patience=10, verbose=True, mode="min")

        ]
    }
    trainer = Trainer(
        **kwargs,
        profiler="simple"
    )

    if log_path:
        kwargs["resume_from_checkpoint"] = os.path.join(log_path, "checkpoint")

    trainer.fit(model, train_loader, valid_loader)


def tuner(model_class, train_loader, valid_loader, config, name, log_path, kwargs_model):

    num_epochs = 50
    

    train_fn_with_parameters = tune.with_parameters(trainer_tuning,
                                                    model_class = model_class,
                                                    train_loader=train_loader,
                                                    valid_loader=valid_loader,
                                                    num_epochs=num_epochs,
                                                    num_gpus=0,
                                                    log_path=log_path,
                                                    kwargs_model=kwargs_model
                                                    )

    tuner = tune.Tuner(
        tune.with_resources(
            train_fn_with_parameters,
            resources={"cpu": 1, "gpu": 0}
        ),
        tune_config=tune.TuneConfig(
            metric="val_loss",
            mode="min",
            scheduler=ASHAScheduler(reduction_factor = 4),
            #scheduler=PB2(time_attr='training_iteration', perturbation_interval=10,hyperparam_bounds={"lr": [1e-5, 1]}),
            # search_alg= BayesOptSearch(),
            search_alg=tune.search.basic_variant.BasicVariantGenerator(),

            num_samples=60

        ),
        run_config=air.RunConfig(
            name=name,
            verbose=2,
            local_dir=log_path,
            callbacks=[
                WandbLoggerCallback(
                    api_key="86a2ba8c8e41892c3c639a87cdd7c01bd034116f", project="MPC", log_config=True, group=name)
            ],
        ),
        param_space=config,
    )

    results = tuner.fit()

    print("Best hyperparameters found were: ",
          results.get_best_result().config)
    with open(log_path/name/'best_results.pkl', 'wb') as f:
        dump(results.get_best_result(), f)


config = {
    "lr": tune.loguniform(1e-5, 1e-1),
    "p_dropout_conv": tune.uniform(0, 1),
    "p_dropout_fc": tune.uniform(0, 1),
    "hidden_size_enc": tune.sample_from(lambda spec: 2**np.random.randint(4, 10)),
    "conv_layers": tune.randint(1, 4),
    "linear_layers": tune.randint(1, 11),
    "conv_features": tune.sample_from(lambda spec: 2**np.random.randint(4, 10, size=spec.config.conv_layers)),
    "conv_kernels": tune.sample_from(lambda spec: 2*np.random.randint(1, 5, size=spec.config.conv_layers)+1),
    "linear_neurons": tune.sample_from(lambda spec: 10*np.random.randint(1, 100, size=spec.config.linear_layers))
}


tuner(CNNEncDecAttn,energy_train_loader, energy_valid_loader,config, 'energy', main_dir/'tuning'/'cnn_bilstm',  kwargs_model = {'len_forecast':len_forecast, 'col_out':1, 'bidirectional' : True})
tuner(CNNEncDecAttn,temp_train_loader, temp_valid_loader, config, 'temp', main_dir/'tuning'/'cnn_bilstm',kwargs_model = {'len_forecast':len_forecast, 'col_out':len(col_out_temp), 'bidirectional' : True})
