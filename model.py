from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


def build_lstm_model(input_shape, units=64, dropout=0.2, learning_rate=0.001):
    """
    Builds and compiles an LSTM model.

    Parameters:
        input_shape (tuple): Shape of input (timesteps, features)
        units (int): Number of LSTM units
        dropout (float): Dropout rate
        learning_rate (float): Learning rate for optimizer

    Returns:
        model (keras.Model): Compiled LSTM model
    """
    model = Sequential()
    model.add(LSTM(units, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(dropout))
    model.add(LSTM(units))
    model.add(Dropout(dropout))
    model.add(Dense(1))

    model.compile(
        optimizer="adam",
        loss="mse"
    )

    return model


def train_model(model, X_train, y_train, epochs=50, batch_size=32):
    """
    Trains the LSTM model with early stopping.

    Parameters:
        model: Compiled model
        X_train: Training features
        y_train: Training target
        epochs (int): Training epochs
        batch_size (int): Batch size

    Returns:
        history
    """
    es = EarlyStopping(patience=5, restore_best_weights=True)

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[es],
        verbose=1
    )

    return history


def predict_model(model, X_test):
    """
    Generates predictions from trained model.

    Parameters:
        model: Trained model
        X_test: Test features

    Returns:
        predictions
    """
    return model.predict(X_test, verbose=0)
