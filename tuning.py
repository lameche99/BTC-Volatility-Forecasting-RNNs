# fine tune hyperparameters with Optuna
# 2023-12-05

from optuna_integration.tfkeras import TFKerasPruningCallback
import optuna
from optuna.trial import TrialState
from optuna.samplers import TPESampler

def hyper_tune(trial, model_fun, train, val, n_steps, n_horizon, n_features):
    """
    Wrapper function to tune hyperparameters
    :param trial: optuna.trial - optuna optimization trial
    :param model_fun: func - model compile function
    :param train: tf.Dataset - train set
    :param val: tf.Dataset - validation set
    :param n_steps: int - size of input interval. Default = 72
    :param n_horizon: int - size of future prediction interval. Default = 24
    :param n_features: int - number of features
    :return: model history for validation metric
    """
    monitor = 'val_mae'
    # Hyperparameters to be tuned by Optuna.
    learning_rate = trial.suggest_float("learning_rate", 3e-7, 3e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-9, 1e-2, log=True)

    model = model_fun(learning_rate, weight_decay, n_steps, n_horizon, n_features)
    callbacks = [TFKerasPruningCallback(trial, monitor)]
    history = model.fit(
        train,
        validation_data=val,
        epochs=150,
        callbacks=callbacks
    )
    return history.history[monitor][-1]

def create_study(model_fun, train, val, n_steps, n_horizon, n_features):
    """
    This function creates an Optuna study object for hyperparameter tuning
    :param model_fun: func - model compile function
    :param train: tf.Dataset - train set
    :param val: tf.Dataset - validation set
    :param n_steps: int - size of input interval. Default = 72
    :param n_horizon: int - size of future prediction interval. Default = 24
    :param n_features: int - number of features
    :return: optuna.Study - collection of tuning trials
    """
    study = optuna.create_study(
        direction="minimize", sampler=TPESampler(multivariate=True), storage=f"sqlite:///hypertune.db",
        load_if_exists=True, pruner=optuna.pruners.NopPruner()
    )
    study.optimize(lambda x: hyper_tune(trial=x,
                                        model_fun=model_fun,
                                        train=train, val=val,
                                        n_steps=n_steps,
                                        n_horizon=n_horizon,
                                        n_features=n_features),
                   n_trials=25, catch=(Exception,))
    return study

def get_optimized_parameters(study):
    """
    This function finds the optimal hyperparameters from an Optuna study
    :param study: optuna.Study - hyperparameter tuning study
    :return: dict - optimal parametrs for particular model
    """
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