# Data Handler functions to create tensorflow datasets
# 2023-12-05

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

def build_dataset(path,
                  train_fraq=0.65, 
                  n_steps=72, 
                  n_horizon=24, 
                  batch_size=256, 
                  shuffle_buffer=500, 
                  expand_dims=False):
    """
    This function creates a windowed dataset for tensorflow models
    :param path: str - dataset file
    :param train_fraq: float - train size. Default = 0.65
    :param n_steps: int - size of input interval. Default = 72
    :param n_horizon: int - size of future prediction interval. Default = 24
    :param batch_size: int - batch size. Default = 256
    :param shuffle_buffer: int - buffer when splitting the data
    :return: tuple(tf.Dataset, tf.Dataset, tf.Dataset) - train, validation, test datasets 
    """
    
    tf.random.set_seed(23)
    
    data = load_data(col=['RV', 'logRet', 'sentiment_score'], path=path)
    # hours, day, minute = make_time_features(data.date)
    # data = pd.concat([data.drop(['date'], axis=1), hours, day, minute], axis=1)
        
    mm = MinMaxScaler()
    data = mm.fit_transform(data)

    train_data, val_data, test_data = split_data(data, train_fraq=train_fraq, test_len=200)    
    train_ds = window_dataset(train_data, n_steps, n_horizon, batch_size, shuffle_buffer, expand_dims=expand_dims)
    val_ds = window_dataset(val_data, n_steps, n_horizon, batch_size, shuffle_buffer, expand_dims=expand_dims)
    test_ds = window_dataset(test_data, n_steps, n_horizon, batch_size, shuffle_buffer, expand_dims=expand_dims)
    
    
    print(f"Prediction lookback (n_steps): {n_steps}")
    print(f"Prediction horizon (n_horizon): {n_horizon}")
    print(f"Batch Size: {batch_size}")
    print("Datasets:")
    print(train_ds.element_spec)
    
    return train_ds, val_ds, test_ds

def load_data(col=None, path="./src/rv_sentiment.csv"):
    """
    This function reads the input dataframe
    :param col: list - ordered list of columns
    :param path: str - file path
    :return: pd.DataFrame - input data
    """
    df = pd.read_csv(path)
    if col is not None:
        df = df[col]
    return df

def min_max_scale(dataframe):
    """ Applies MinMax Scaling
    
        Wrapper for sklearn's MinMaxScaler class.
    """
    mm = MinMaxScaler()
    return mm.fit_transform(dataframe)

def make_time_features(series):
    """
    Converts timestamp into three features
    """
    #convert series to datetimes
    times = series.apply(lambda x: x.split('+')[0])
    datetimes = pd.DatetimeIndex(times)
    
    hours = datetimes.hour.values
    day = datetimes.dayofweek.values
    minutes = datetimes.minute.values
    
    hour = pd.Series(hours, name='hours')
    dayofw = pd.Series(day, name='dayofw')
    month = pd.Series(minutes, name='minutes')
    
    return hour, dayofw, month

def split_data(series, train_fraq, test_len=200):
    """Splits input series into train, val and test.
    
        Default to 200 observations of test data.
    """
    #slice the last year of data for testing 1 year has 8760 hours
    test_slice = len(series)-test_len

    test_data = series[test_slice:]
    train_val_data = series[:test_slice]

    #make train and validation from the remaining
    train_size = int(len(train_val_data) * train_fraq)
    
    train_data = train_val_data[:train_size]
    val_data = train_val_data[train_size:]
    
    return train_data, val_data, test_data

def window_dataset(data, n_steps, n_horizon, batch_size, shuffle_buffer, expand_dims=False):
    """ Create a windowed tensorflow dataset
    """
    #create a window with n steps back plus the size of the prediction length
    window = n_steps + n_horizon
    
    #expand dimensions to 3D to fit with LSTM inputs
    #creat the inital tensor dataset
    if expand_dims:
        ds = tf.expand_dims(data, axis=-1)
        ds = tf.data.Dataset.from_tensor_slices(ds)
    else:
        ds = tf.data.Dataset.from_tensor_slices(data)
    
    #create the window function shifting the data by the prediction length
    ds = ds.window(window, shift=n_horizon, drop_remainder=True)
    
    #flatten the dataset and batch into the window size
    ds = ds.flat_map(lambda x : x.batch(window))
    ds = ds.shuffle(shuffle_buffer)    
    
    #create the supervised learning problem x and y and batch
    ds = ds.map(lambda x : (x[:-n_horizon, 1:], x[-n_horizon:, :1]))
    ds = ds.batch(batch_size).prefetch(1)
    
    return ds