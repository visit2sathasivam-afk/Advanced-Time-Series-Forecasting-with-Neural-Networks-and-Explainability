import numpy as np
from sklearn.metrics import mean_squared_error
import optuna

from model import build_lstm_model


def tune_lstm(X_train, y_train, X_val, y_val, input_shape, n_trials=10):
    """
    Tunes LSTM hyperparameters using Optuna.

    Parameters:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        input_shape: Shape of model input
        n_trials: Number of Optuna trials

    Returns:
        best_params (dict)
    """
    def objective(trial):
        units = trial.suggest_int("units", 32, 128)
        dropout = trial.suggest_float("dropout", 0.1, 0.5)

        model = build_lstm_model(input_shape, units, dropout)

        model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=20,
            batch_size=32,
            verbose=0
        )

        preds = model.predict(X_val, verbose=0)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        return rmse

    print("Starting Optuna tuning...")

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    print("Best parameters found:", study.best_params)

    return study.best_params
