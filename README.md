📈 Advanced Time Series Forecasting with Neural Networks and Explainability
-----------------------------------------------------------------------------

📌 Project Overview

This project implements an advanced multivariate time series forecasting system using a Long Short-Term Memory (LSTM) neural network. The objective is to model complex temporal dependencies, perform multi-step forecasting, optimize model performance through hyperparameter tuning, and interpret predictions using explainability techniques like SHAP.

The system is evaluated against traditional baseline models such as ARIMA and Exponential Smoothing (Holt-Winters) using metrics like RMSE and MAE.


🎯 Objectives

Generate or acquire a complex multivariate time series dataset.

Engineer temporal features (lags, rolling statistics).

Build and tune an LSTM forecasting model.

Perform hyperparameter optimization using Optuna.

Compare results with classical forecasting methods.

Apply explainability (SHAP) to interpret model behavior.

Provide production-style, reproducible code and analysis.


🗂 Project Structure

time_series_project/
│
├── main.py                # End-to-end pipeline
├── data_generation.py    # Dataset creation
├── preprocessing.py      # Feature engineering & scaling
├── model.py              # LSTM architecture
├── tuning.py             # Optuna tuning logic
├── baselines.py          # ARIMA & Holt-Winters models
├── explainability.py     # SHAP analysis
├── report.txt            # Text report
├── README.md             # Project documentation



⚙️ Tech Stack

Python

TensorFlow / Keras

NumPy, Pandas

Scikit-learn

Optuna

Statsmodels

SHAP

Matplotlib


📊 Dataset

The dataset is programmatically generated with:

Trend component

Seasonal component

Noise

Lagged features

Rolling mean and standard deviation

It contains 1500+ observations and supports multivariate forecasting.

Example features:

target

lag1, lag2

rolling_mean

rolling_std


🧠 Model Architecture

The LSTM network includes:

Two stacked LSTM layers

Dropout regularization

Dense output layer

Adam optimizer

Mean Squared Error loss

Example:

LSTM → Dropout → LSTM → Dropout → Dense


🔍 Hyperparameter Optimization

Hyperparameters are tuned using Optuna, including:

Number of LSTM units

Dropout rate

The objective minimizes RMSE on validation data.


📏 Evaluation Metrics

Models are evaluated using:

RMSE (Root Mean Squared Error)

MAE (Mean Absolute Error)

Compared models:

LSTM

ARIMA

Holt-Winters (Exponential Smoothing)

Results are summarized in a comparison table.


🧪 Baseline Models

ARIMA for statistical time series modeling.

Exponential Smoothing (Holt-Winters) for trend and seasonality capture.

These provide performance benchmarks against the LSTM model.


🧩 Explainability

To interpret predictions, the project applies SHAP (KernelExplainer) to:

Identify influential lag features.

Understand temporal importance.

Visualize feature contributions.

This improves transparency of the neural network forecasts.


▶️ How to Run

Install dependencies:

pip install numpy pandas scikit-learn tensorflow optuna shap statsmodels matplotlib


Run the pipeline:

python main.py


or open the notebook and run all cells.

📌 Outputs

Trained LSTM model

RMSE and MAE scores

Baseline comparison

SHAP summary plots

Forecast visualizations


📝 Report

The report.txt includes:

Dataset description

Feature engineering explanation

Optimization process

Model comparison

Explainability insights

Final conclusions


✅ Conclusion

The LSTM model effectively captures nonlinear temporal dependencies and outperforms classical approaches such as ARIMA and Holt-Winters. Explainability analysis confirms that recent lag values and rolling statistics significantly influence predictions, improving trust and interpretability in deep learning forecasts.


👨‍💻 Author

Sathasivam Murugesan


Advanced Time Series Forecasting Project
