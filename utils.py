import numpy as np
import pandas as pd
import torch


def split_seq(df, time_window, len_forecast, col_out):
    seq = []
   
    df = torch.tensor(df.values.astype(float))
    for i in range(len(df)):
        
        end_ix = i + time_window
        
        if end_ix+len_forecast > len(df):
            break
        seq_x, seq_y = df[i:end_ix], df[end_ix:end_ix+len_forecast, col_out]
        seq.append((seq_x.transpose(0,1), seq_y))
    return seq


def merge(path, csv_list):
    df = pd.read_csv(path + '\\'+ csv_list[0])
    df['date'] = pd.to_datetime(df['date'])
    df = df.drop_duplicates(subset = ['date'], keep='first')
    df['date'] = pd.Series(pd.date_range(min(df['date']), max(df['date']), freq = '15min'))
    for dataframe_name in csv_list[1:]:
        dataframe = pd.read_csv(path + '\\' + dataframe_name)
        dataframe['date']  = pd.to_datetime(dataframe['date'])
        dataframe = dataframe.drop_duplicates(subset = ['date'], keep = 'first')
        df = pd.merge_asof(df, dataframe, on = 'date', tolerance = pd.Timedelta('10min'))
        df.drop(df.filter(regex="Unnamed"),axis=1, inplace=True)

        
    return df

def remove_outliers(df):
    for col in list(df.columns):
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_lim = Q1 - 1.5*IQR
        upper_lim = Q3 + 1.5*IQR
        outliers = (df[col] < lower_lim) | (df[col] > upper_lim)
        df[col] = df[col].where(~outliers, np.nan)
    return df

def imputation(df, imputer, fit = False):
    if fit:
        return pd.DataFrame(imputer.fit_transform(df), columns = df.columns)
    else:
        return pd.DataFrame(imputer.transform(df), columns = df.columns)


def split(df, train_size = 0.7, test_size = 0.2):
    train_indices = np.ceil(len(df)*train_size).astype(int)
    test_indices = np.ceil(len(df)*test_size).astype(int)
    train_set = df[:train_indices]
    test_set = df[train_indices:train_indices+test_indices]
    valid_set = df[train_indices+test_indices:]

    return train_set, valid_set, test_set

def col_out_to_index(df, col_out):
    col_out_in = {}
    for i, col in enumerate(col_out):
        in_x = df.columns.get_loc(col)
        col_out_in[col] = (in_x, i)

    return col_out_in