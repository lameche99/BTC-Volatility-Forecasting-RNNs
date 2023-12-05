# tensorflow models
# 2023-12-05

from keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from keras.models import Sequential
from keras.optimizers import AdamW
from keras.losses import Huber
from keras.backend import clear_session
from handler import build_dataset

def cnn_model(lr, wd, n_steps=72, n_horizon=24, n_features=5):
    """
    Create CNN tensorflow model with MaxPooling
    :param lr: float - learning rate
    :param wd: float - weight decay
    :param n_steps: int - size of input interval. Default = 72
    :param n_horizon: int - size of future prediction interval. Default = 24
    :param n_features: int - number of features
    :return: tf.models.Sequential - CNN model  
    """
    clear_session()

    model = Sequential([
        Conv1D(64, kernel_size=6, activation='relu', input_shape=(n_steps,n_features)),
        MaxPooling1D(2),
        Conv1D(64, kernel_size=3, activation='relu'),
        MaxPooling1D(2),
        Flatten(),
        Dropout(0.3),
        Dense(128),
        Dropout(0.3),
        Dense(n_horizon)
    ], name="CNN")
    
    optimizer = AdamW(learning_rate=lr,
                      weight_decay=wd)
    loss = Huber()
    model.compile(loss=loss, optimizer=optimizer, metrics=['mae'])
    
    return model

def lstm_model(lr, wd, n_steps=72, n_horizon=24, n_features=5):
    """
    Create LSTM tensorflow model with MaxPooling
    :param lr: float - learning rate
    :param wd: float - weight decay
    :param n_steps: int - size of input interval. Default = 72
    :param n_horizon: int - size of future prediction interval. Default = 24
    :param n_features: int - number of features
    :return: tf.models.Sequential - LSTM model  
    """
    clear_session()
    
    model = Sequential([
        LSTM(72, activation='relu', input_shape=(n_steps, n_features), return_sequences=True),
        LSTM(48, activation='relu', return_sequences=False),
        Flatten(),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(n_horizon)
    ], name='lstm')
    
    loss = Huber()
    optimizer = AdamW(learning_rate=lr,
                      weight_decay=wd)
    
    model.compile(loss=loss, optimizer=optimizer, metrics=['mae'])
    
    return model

def lstm_cnn_model(lr, wd, n_steps=72, n_horizon=24, n_features=5):
    """
    Create LSTM-CNN stacked tensorflow model with MaxPooling
    :param lr: float - learning rate
    :param wd: float - weight decay
    :param n_steps: int - size of input interval. Default = 72
    :param n_horizon: int - size of future prediction interval. Default = 24
    :param n_features: int - number of features
    :return: tf.models.Sequential - LSTM-CNN model  
    """
    clear_session()
    
    model = Sequential([
        Conv1D(64, kernel_size=6, activation='relu', input_shape=(n_steps,n_features)),
        MaxPooling1D(2),
        Conv1D(64, kernel_size=3, activation='relu'),
        MaxPooling1D(2),
        LSTM(72, activation='relu', return_sequences=True),
        LSTM(48, activation='relu', return_sequences=False),
        Flatten(),
        Dropout(0.3),
        Dense(128),
        Dropout(0.3),
        Dense(n_horizon)
    ], name="lstm_cnn")
    
    loss = Huber()
    optimizer = AdamW(learning_rate=lr,
                      weight_decay=wd)
    
    model.compile(loss=loss, optimizer=optimizer, metrics=['mae'])
    
    return model

def cfg_model_run(model, history, test_ds):
    """
    Get model configurations
    :param model: tf.models.Sequential - tensorflow model
    :param history: - model training history
    :param test_ds: tf.Dataset - test dataset
    :return: dict - model configurations dictionary
    """
    return {"model": model, "history" : history, "test_ds": test_ds}

def run_model(fname, model_name, model_func, model_configs, model_parms, n_steps=72, n_horizon=24, n_features=5):
    """
    This function builds a dataset, trains it and updates the model configuration dictionary.
    :param fname: str - file path
    :param model_name: str - model name
    :param model_func: func - function to compile model
    :param model_configs: dict - model configurations
    :param model_parms: dict - tuned hyperparamaeters
    :param n_steps: int - size of input interval. Default = 72
    :param n_horizon: int - size of future prediction interval. Default = 24
    :param n_features: int - number of features
    :return: tf.Dataset - test dataset
    """
    train_ds, val_ds, test_ds = build_dataset(path=fname, n_steps=n_steps, n_horizon=n_horizon)

    model = model_func(lr=model_parms['learning_rate'], wd=model_parms['weight_decay'], n_steps=n_steps, n_horizon=n_horizon, n_features=n_features)
    model_hist = model.fit(train_ds, validation_data=val_ds, epochs=150)

    model_configs[model_name] = cfg_model_run(model, model_hist, test_ds)
    return test_ds