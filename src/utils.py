import numpy as np
import pandas as pd
from pathlib import Path
import tqdm

MAIN_DIR = Path(__file__).parent.parent

def merge(path, csv_list):
    df = pd.read_csv(path/csv_list[0])
    df['date'] = pd.to_datetime(df['date'])
    df = df.drop_duplicates(subset = ['date'], keep='first')
    df['date'] = pd.Series(pd.date_range(min(df['date']), max(df['date']), freq = '15min'))
    for dataframe_name in tqdm.tqdm(csv_list[1:], desc = 'Merge', unit = 'file'):
        dataframe = pd.read_csv(path/dataframe_name)
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


def split(df, train_size = 0.8):
    train_indices = np.ceil(len(df)*train_size).astype(int)
    train_set = df[:train_indices].reset_index()
    test_set = df[train_indices:].reset_index()

    return train_set, test_set

def col_out_to_index(df, col_out):
    col_out_in = {}
    for i, col in enumerate(col_out):
        in_x = df.columns.get_loc(col)
        col_out_in[col] = (in_x, i)

    return col_out_in
'''
def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file: the path of the config file
    :return: config(namespace), config(dictionary)
    """

    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        try:
            config_dict = json.load(config_file)
            # EasyDict allows to access dict values as attributes (works recursively).
            config = EasyDict(config_dict)
            return config, config_dict
        except ValueError:
            print("INVALID JSON file format.. Please provide a good json file")
            exit(-1)
'''