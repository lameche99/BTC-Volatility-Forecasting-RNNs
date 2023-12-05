from keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from keras.models import Sequential
from keras.optimizers import AdamW
from keras.losses import Huber
from keras.backend import clear_session
from handler import build_dataset

def cnn_model(lr, wd, n_steps=72, n_horizon=24, n_features=5):
    
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
    return {"model": model, "history" : history, "test_ds": test_ds}

def run_model(fname, model_name, model_func, model_configs, model_parms, n_steps=72, n_horizon=24):
    
    train_ds, val_ds, test_ds = build_dataset(path=fname, n_steps=n_steps, n_horizon=n_horizon)

    model = model_func(lr=model_parms['learning_rate'], wd=model_parms['weight_decay'], n_steps=n_steps, n_horizon=n_horizon)
    model_hist = model.fit(train_ds, validation_data=val_ds, epochs=150)

    model_configs[model_name] = cfg_model_run(model, model_hist, test_ds)
    return test_ds