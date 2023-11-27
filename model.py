import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras.layers import LSTM, GRU, Dense, Reshape
from keras.models import Sequential
from keras.optimizers import AdamW
from keras.backend import clear_session
from optuna_integration.tfkeras import TFKerasPruningCallback
import sqlite3
import optuna
from optuna.trial import TrialState
from optuna.samplers import TPESampler

# engine = sqlite3.connect('./btc-data.db')
# btc = pd.read_sql('SELECT * FROM BTCUSD', engine)
btc = pd.read_csv('./sentiment.csv')
scaler = MinMaxScaler()
scaler.fit(btc[['logRet', 'sentiment_score']].values)
n = len(btc)
dt = 44
scaled = scaler.transform(btc[['logRet', 'sentiment_score']].values)[:n-dt] # X matrix
logRet = scaler.transform(btc[['logRet', 'sentiment_score']].values)[dt:, 0] # y values
min_ = scaler.min_ # min scalers
scale_ = scaler.scale_ # scales
X = scaled.reshape(29, 22, 2)
Y = logRet.reshape(29, 22, 1)

train_size = int(0.6 * len(X))
val_size = int(0.8 * len(X))
train_x, train_y = X[:train_size], Y[:train_size]
val_x, val_y = X[train_size:val_size], Y[train_size:val_size]
test_x, test_y = X[val_size:], Y[val_size:]
full_train_x, full_train_y = X[:val_size], Y[:val_size]

def squared_epsilon_insensitive_loss(epsilon: float = 0.025):
    """
    Training loss function wrapper
    :param epsilon: float - tolerance (default = 0.025)
    :return: _loss - loss function
    """
    def _loss(y_true: float, y_pred: float):
        """
        Training loss function
        :param y_true: float - true RV
        :param y_pred: float - predicted RV
        :return: float - train loss
        """
        losses = tf.where(
            tf.abs(y_true - y_pred) > epsilon,
            tf.square(tf.abs(y_true - y_pred) - epsilon),
            0,
        )
        return tf.reduce_sum(losses)

    return _loss

def invTransform(sig: np.array):
    """
    Invert scaled signals
    :param sig: np.array - (3,) signals with resepct 
    to given dimensions (days, steps, channels)
    :param df_scale: pd.DataFrame - scales for data
    :return: np.array - transformed output
    """
    scalermin_, scalerscale_ = min_, scale_
    X = tf.reshape(sig[0,:,0],[sig.shape[1],1])
    X -= scalermin_
    X /= scalerscale_
    return tf.reshape(X, [1,44,1])

def calculateRV(signal: np.array):
    """
    Calculate Realized Volatility (RV)
    :param signal: np.array - log returns
    :return: float - Realized Volatility
    """
    return 100 * tf.math.sqrt(tf.math.reduce_sum(tf.math.square(signal),axis=-1))

def mape(y_true: np.array, y_pred: np.array):
    """
    Mean Absolute Percentage Error for CV metric
    :param y_true: np.array - true RV values
    :param y_pred: np.array - predicted RV values
    :return: float - mean absolute percentage error
    """
    actual = calculateRV(invTransform(y_true)[:,:,0])
    pred = calculateRV(invTransform(y_pred)[:,:,0])
    mape = tf.reduce_mean(tf.abs((actual - pred) / actual))
    return mape
mape.__name__ = "mape"

def create_model(trial, mode: int = 1):
    """
    Create RNN model
    """
    # Hyperparameters to be tuned by Optuna.
    learning_rate = trial.suggest_float("learning_rate", 1e-7, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-9, 1e-2, log=True)
    rec_units = trial.suggest_int("recurrent", 32, 512)
    dropout_units = trial.suggest_uniform("dropout", 0, 0.5)
    epsilon = trial.suggest_float("epsilon", 1e-2, 1e-1, log=True)

    model = Sequential()
    if mode == 1:
        model.add(LSTM(units=rec_units, dropout=dropout_units))
    else:
        model.add(GRU(units=rec_units, dropout=dropout_units))
    
    model.add(Dense(units=22))
    model.add(Reshape((22,1)))

    # Compile model.
    model.compile(
        optimizer=AdamW(
            learning_rate=learning_rate,
            weight_decay=weight_decay
        ),
        loss=squared_epsilon_insensitive_loss(epsilon=epsilon),
        metrics=mape
    )

    return model

def train_model(trial, train_x: np.array = train_x,
                train_y: np.array = train_y,
                valid_x: np.array = val_x,
                valid_y: np.array = val_y,
                mode: int = 1):
    # Clear clutter from previous TensorFlow graphs.
    clear_session()

    # Metrics to be monitored by Optuna.
    monitor = "val_mape"

    # Create tf.keras model instance.
    model = create_model(trial, mode)

    # Create callbacks for pruning.
    callbacks = [
        TFKerasPruningCallback(trial, monitor),
    ]

    # Train model.
    history = model.fit(
        x=train_x,
        y=train_y,
        epochs=15,
        validation_data=(valid_x, valid_y),
        callbacks=callbacks,
        shuffle=False,
        batch_size=1
    )

    return history.history[monitor][-1]

def create_study():
    study = optuna.create_study(
        direction="minimize", sampler=TPESampler(multivariate=True), storage=f"sqlite:///test_rnn.db",
        load_if_exists=True, pruner=optuna.pruners.NopPruner()
    )
    study.optimize(train_model, n_trials=25, catch=(Exception,))
    return study

def get_optimized_parameters(study):

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    return dict(trial.params.items())

def test_model(rec_units, dropout_units,
                learning_rate, weight_decay,
                epsilon,
                train_x: np.array = train_x,
                train_y: np.array = train_y,
                valid_x: np.array = val_x,
                valid_y: np.array = val_y,
                mode: int = 1):
    model = Sequential()
    if mode == 1:
        model.add(LSTM(units=rec_units, dropout=dropout_units))
    else:
        model.add(GRU(units=rec_units, dropout=dropout_units))
    
    model.add(Dense(units=22))
    model.add(Reshape((22,1)))

    # Compile model.
    model.compile(
        optimizer=AdamW(
            learning_rate=learning_rate,
            weight_decay=weight_decay
        ),
        loss=squared_epsilon_insensitive_loss(epsilon=epsilon),
        metrics=mape
    )
    model.fit(
        x=train_x,
        y=train_y,
        epochs=15,
        validation_data=(valid_x, valid_y),
        shuffle=False,
        batch_size=1
    )

    y_pred = model.predict(
        x=test_x
    )
    return y_pred

def invPred(sig):
    """Inverse transform scaled signals
    Parameters
    ----------
    sig : np.ndarray; (_days, _steps, _channels)
        Signals wrt given dimensions

    df_scale : DataFrame
        Respective scale dataframe with columns mins and scales necessary
        for inverse_transform function.

    Returns
    -------
    to_ret : np.ndarray; (_days, _steps, _channels)
        Converted back to Log-Return Scales
    """
    to_ret = np.zeros((sig.shape[0], sig.shape[1], sig.shape[2]))
    print(np.shape(to_ret))
    for i in range(sig.shape[0]):
        for j in range(sig.shape[2]):
            scaler = MinMaxScaler()
            scaler.min_, scaler.scale_ = min_[j], scale_[j]
            to_ret[i,:,j] = scaler.inverse_transform(sig[i,:,j].reshape(sig.shape[1],1)).reshape(sig.shape[1],)

    return to_ret

def calculate_RV(signal):
    """Calcualte RV from log returns
    Parameters
    ----------
    signal : np.ndarray
        Input array with first dimension as days and last dimension as log-returns
    Returns
    -------
    np.ndarray
        [description]
    """
    return 100 * np.sqrt(np.square(signal).sum(axis=-1))


def evaluate_forecast(forecast, actual):
    mape = np.mean(np.abs(forecast - actual) / np.abs(actual))
    me = np.mean(forecast - actual)
    mae = np.mean(np.abs(forecast - actual))
    mpe = np.mean((forecast - actual) / actual)
    rmse = np.mean((forecast - actual) ** 2) ** 0.5
    msle = np.mean(np.square(np.log(1 + forecast) - np.log(1 + actual)))
    return {"mape": mape, "me": me, "mae": mae, "mpe": mpe, "rmse": rmse, "msle": msle}
