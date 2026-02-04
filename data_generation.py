import numpy as np
import pandas as pd


def generate_time_series(n_points=1500, seasonal_period=50, noise_std=1.0, seed=42):
    """
    Generates a synthetic multivariate time series dataset.

    Parameters:
        n_points (int): Number of data points
        seasonal_period (int): Period for seasonality
        noise_std (float): Noise standard deviation
        seed (int): Random seed

    Returns:
        DataFrame with target and engineered features
    """
    np.random.seed(seed)

    t = np.arange(n_points)

    # Components
    trend = t * 0.01
    seasonal = 10 * np.sin(2 * np.pi * t / seasonal_period)
    noise = np.random.normal(0, noise_std, n_points)

    target = trend + seasonal + noise

    # Lag features
    lag1 = np.roll(target, 1)
    lag2 = np.roll(target, 2)

    df = pd.DataFrame({
        "target": target,
        "lag1": lag1,
        "lag2": lag2
    })

    # Feature engineering
    df["rolling_mean"] = df["target"].rolling(5).mean()
    df["rolling_std"] = df["target"].rolling(5).std()

    df = df.dropna().reset_index(drop=True)

    return df


if __name__ == "__main__":
    df = generate_time_series()
    print(df.head())
    print("Dataset shape:", df.shape)
