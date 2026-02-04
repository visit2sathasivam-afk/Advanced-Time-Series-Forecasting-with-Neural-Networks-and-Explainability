import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def run_arima(train_series, test_series, order=(5, 1, 0)):
    """
    Runs ARIMA baseline model.

    Parameters:
        train_series: Training time series
        test_series: Test time series
        order: ARIMA order

    Returns:
        predictions, rmse, mae
    """
    model = ARIMA(train_series, order=order)
    fit = model.fit()

    preds = fit.forecast(len(test_series))

    rmse = np.sqrt(mean_squared_error(test_series, preds))
    mae = mean_absolute_error(test_series, preds)

    return preds, rmse, mae


def run_holt_winters(train_series, test_series, seasonal_periods=50):
    """
    Runs Holt-Winters Exponential Smoothing baseline.

    Parameters:
        train_series: Training time series
        test_series: Test time series
        seasonal_periods: Seasonality period

    Returns:
        predictions, rmse, mae
    """
    model = ExponentialSmoothing(
        train_series,
        seasonal="add",
        trend="add",
        seasonal_periods=seasonal_periods
    )

    fit = model.fit()
    preds = fit.forecast(len(test_series))

    rmse = np.sqrt(mean_squared_error(test_series, preds))
    mae = mean_absolute_error(test_series, preds)

    return preds, rmse, mae
