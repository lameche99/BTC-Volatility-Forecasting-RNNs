import pandas as pd
import numpy as np
import tensorflow as tf
from keras.layers import LSTM, GRU, Dense, Reshape
from keras.models import Sequential
from keras.optimizers import AdamW
from keras.backend import clear_session
from optuna.integration import TFKerasPruningCallback


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

def invTransform(sig: np.array, df_scale: pd.DataFrame):
    """
    Invert scaled signals
    :param sig: np.array - (3,) signals with resepct 
    to given dimensions (days, steps, channels)
    :param df_scale: pd.DataFrame - scales for data
    :return: np.array - transformed output
    """
    scalermin_, scalerscale_ = df_scale["mins"][0], df_scale["scales"][0]
    X = tf.reshape(sig[0,:,0],[sig.shape[1],1])
    X -= scalermin_
    X /= scalerscale_
    return tf.reshape(X, [1,96,1])

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
    
    model.add(Dense(units=96))
    model.add(Reshape((96,1)))

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

def train_model(trial, train_x: np.array, train_y: np.array, valid_x: np.array, valid_y: np.array, mode: int = 1):
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
        epochs=30,
        validation_data=(valid_x, valid_y),
        callbacks=callbacks,
        shuffle=False,
        batch_size=1
    )

    return history.history[monitor][-1]

