import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

import optuna
import shap

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing


# -----------------------------
# 1. DATA GENERATION
# -----------------------------
np.random.seed(42)

n = 1500
t = np.arange(n)

trend = t * 0.01
seasonal = 10 * np.sin(2 * np.pi * t / 50)
noise = np.random.normal(0, 1, n)

y = trend + seasonal + noise
x1 = np.roll(y, 1)
x2 = np.roll(y, 2)

df = pd.DataFrame({
    "target": y,
    "lag1": x1,
    "lag2": x2
}).dropna()

df["rolling_mean"] = df["target"].rolling(5).mean()
df["rolling_std"] = df["target"].rolling(5).std()
df = df.dropna()

print("Dataset shape:", df.shape)


# -----------------------------
# 2. PREPROCESSING
# -----------------------------
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df)


def create_sequences(data, seq_len=30):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len, 0])
    return np.array(X), np.array(y)


X, y = create_sequences(scaled, 30)

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)


# -----------------------------
# 3. MODEL
# -----------------------------
def build_model(units=64, dropout=0.2):
    model = Sequential()
    model.add(LSTM(units, return_sequences=True, input_shape=X_train.shape[1:]))
    model.add(Dropout(dropout))
    model.add(LSTM(units))
    model.add(Dropout(dropout))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    return model


# -----------------------------
# 4. OPTUNA TUNING
# -----------------------------
def objective(trial):
    units = trial.suggest_int("units", 32, 128)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)

    model = build_model(units, dropout)
    model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=20,
        batch_size=32,
        verbose=0
    )

    preds = model.predict(X_test, verbose=0)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    return rmse


print("\nRunning Optuna tuning...")
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=10)

print("Best Params:", study.best_params)


# -----------------------------
# 5. FINAL TRAINING
# -----------------------------
best = study.best_params
model = build_model(best["units"], best["dropout"])

es = EarlyStopping(patience=5, restore_best_weights=True)

model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    callbacks=[es],
    verbose=1
)


# -----------------------------
# 6. LSTM EVALUATION
# -----------------------------
pred_lstm = model.predict(X_test)

rmse_lstm = np.sqrt(mean_squared_error(y_test, pred_lstm))
mae_lstm = mean_absolute_error(y_test, pred_lstm)

print("\nLSTM RMSE:", rmse_lstm)
print("LSTM MAE :", mae_lstm)


# -----------------------------
# 7. BASELINES
# -----------------------------
train_series = df["target"][:split]
test_series = df["target"][split:]


# ARIMA
arima = ARIMA(train_series, order=(5, 1, 0))
arima_fit = arima.fit()
pred_arima = arima_fit.forecast(len(test_series))

rmse_arima = np.sqrt(mean_squared_error(test_series, pred_arima))
mae_arima = mean_absolute_error(test_series, pred_arima)


# HOLT-WINTERS
hw = ExponentialSmoothing(train_series, seasonal="add", seasonal_periods=50)
hw_fit = hw.fit()
pred_hw = hw_fit.forecast(len(test_series))

rmse_hw = np.sqrt(mean_squared_error(test_series, pred_hw))
mae_hw = mean_absolute_error(test_series, pred_hw)


print("\nARIMA RMSE:", rmse_arima, " MAE:", mae_arima)
print("HW RMSE   :", rmse_hw, " MAE:", mae_hw)


# -----------------------------
# 8. RESULTS TABLE
# -----------------------------
results = pd.DataFrame({
    "Model": ["LSTM", "ARIMA", "Holt-Winters"],
    "RMSE": [rmse_lstm, rmse_arima, rmse_hw],
    "MAE": [mae_lstm, mae_arima, mae_hw]
})

print("\nFinal Comparison:")
print(results)


# -----------------------------
# 9. EXPLAINABILITY (SHAP)
# -----------------------------
print("\nRunning SHAP Explainability...")

explainer = shap.KernelExplainer(model.predict, X_train[:100])
shap_values = explainer.shap_values(X_test[:10])

shap.summary_plot(shap_values, X_test[:10], show=False)
plt.savefig("shap_summary.png")
plt.close()

print("SHAP plot saved as shap_summary.png")


# -----------------------------
# 10. FORECAST PLOT
# -----------------------------
plt.figure(figsize=(10, 5))
plt.plot(y_test[:200], label="Actual")
plt.plot(pred_lstm[:200], label="LSTM Forecast")
plt.legend()
plt.title("LSTM Forecast vs Actual")
plt.savefig("forecast_plot.png")
plt.close()

print("Forecast plot saved as forecast_plot.png")


print("\n--- PIPELINE COMPLETED SUCCESSFULLY ---")
