from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss, MAE
import torch.nn as nn
import pandas as pd
from pathlib import Path
import pytorch_lightning as pl

main_dir = Path(__file__).parent.parent



training_data = pd.read_csv(main_dir/'data'/'cleaned'/'energy/train_set_imp.csv')
training_data['date'] = pd.to_datetime(training_data['date'])

training_data['time_idx'] = training_data['date'].astype(int)/1e9
training_data['time_idx'] -= training_data['time_idx'].min()
training_data['group_ids'] = 0
training_data['time_idx'] = (training_data['time_idx']/900).astype(int)
training_data['month'] = training_data['date'].dt.month.astype(str).astype("category")
training_data['hour'] = training_data['date'].dt.hour.astype(str).astype("category")
training_data['day'] = training_data['date'].dt.dayofweek.astype(str).astype("category")
training_data.drop('date', axis = 1, inplace = True)

validation_data = pd.read_csv(main_dir/'data'/'cleaned'/'energy/valid_set_imp.csv')
validation_data['date'] = pd.to_datetime(validation_data['date'])

validation_data['time_idx'] = validation_data['date'].astype(int)/1e9
validation_data['time_idx'] -= validation_data['time_idx'].min()
validation_data['group_ids'] = 0
validation_data['time_idx'] = (validation_data['time_idx']/900).astype(int)
validation_data['month'] = validation_data['date'].dt.month.astype(str).astype("category")
validation_data['hour'] = validation_data['date'].dt.hour.astype(str).astype("category")
validation_data['day'] = validation_data['date'].dt.dayofweek.astype(str).astype("category")
validation_data.drop('date', axis = 1, inplace = True)

scalers = {}

for x in training_data.columns:
    if x not in ['month', 'day', 'hour', 'hvac']:
        scalers[x] = None

training = TimeSeriesDataSet(
    training_data,
    time_idx = 'time_idx',
    group_ids = ['group_ids'],
    target = 'hvac',
    max_encoder_length=96,
    max_prediction_length = 4,
    time_varying_unknown_categoricals = ['month', 'hour', 'day'],
    time_varying_unknown_reals = [x for x in training_data.columns if (x not in ['month', 'day', 'hour','air_temp_set_1', 'air_temp_set_2', 'dew_point_temperature_set_1d', 'relative_humidity_set_1', 'solar_radiation_set_1'])],
    time_varying_known_reals = ['air_temp_set_1', 'air_temp_set_2', 'dew_point_temperature_set_1d', 'relative_humidity_set_1', 'solar_radiation_set_1'],
    target_normalizer=None,
    scalers = scalers,
)

validation = TimeSeriesDataSet.from_dataset(training, validation_data, stop_randomization=True)
train_dataloader = training.to_dataloader(train=True, batch_size = 128, num_workers = 24)
validation_dataloader = validation.to_dataloader(train=False, batch_size = 128, num_workers = 24)
tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate = 3e-1,
    hidden_size = 8,
    lstm_layers = 1,
    attention_head_size = 1,
    dropout = 0.2,
    hidden_continuous_size = 4,
    output_size = 1,
    loss = MAE(),
    reduce_on_plateau_patience = 4,
)

'''
print(f"suggested learning rate: {res.suggestion()}")
fig = res.plot(show=True, suggest=True)
fig.show()
res = trainer.tuner.lr_find(
    tft,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
    max_lr=10.0,
    min_lr=1e-6,
)
'''
trainer = pl.Trainer(
    accelerator = 'gpu',
    devices = 1,
    #gradient_clip_val = 0.1,
    max_epochs = 10,
    limit_train_batches = 0.1
)

trainer.fit(tft, train_dataloader, validation_dataloader)
predictions, x = tft.predict(validation_dataloader, mode = 'raw', fast_dev_run = 'True', return_x=True)
plot = tft.plot_prediction(x, predictions, idx=0, add_loss_to_title=True)
plot.savefig("plot.jpg")