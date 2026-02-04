import numpy as np
from sklearn.preprocessing import MinMaxScaler


def scale_features(df):
    """
    Scales dataframe features using MinMaxScaler.

    Parameters:
        df (DataFrame): Input dataframe

    Returns:
        scaled_data (ndarray)
        scaler (MinMaxScaler)
    """
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)
    return scaled_data, scaler


def create_sequences(data, seq_len=30):
    """
    Converts time series into supervised learning sequences.

    Parameters:
        data (ndarray): Scaled data
        seq_len (int): Sequence length

    Returns:
        X, y arrays
    """
    X, y = [], []

    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len, 0])

    return np.array(X), np.array(y)


def train_test_split_ts(X, y, train_ratio=0.8):
    """
    Splits time series data sequentially.

    Parameters:
        X: Features
        y: Target
        train_ratio: Train size ratio

    Returns:
        X_train, X_test, y_train, y_test
    """
    split = int(len(X) * train_ratio)

    X_train = X[:split]
    X_test = X[split:]

    y_train = y[:split]
    y_test = y[split:]

    return X_train, X_test, y_train, y_test
