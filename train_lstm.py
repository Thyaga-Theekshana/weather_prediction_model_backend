import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

FEATURE_NAMES = [
    'temperature',
    'humidity',
    'pressure',
    'aqi',
    'pm2_5',
    'light',
]


def generate_dummy_data(num_samples=2000):
    """Generate a dummy multivariate time series dataset."""
    np.random.seed(42)
    timestamps = np.arange(num_samples)

    temperature = 20 + 5 * np.sin(2 * np.pi * timestamps / 144) + 0.5 * np.random.randn(num_samples)
    humidity = 50 + 10 * np.sin(2 * np.pi * timestamps / 72 + 1.0) + 2.0 * np.random.randn(num_samples)
    pressure = 1013 + 8 * np.sin(2 * np.pi * timestamps / 168 + 0.5) + 0.8 * np.random.randn(num_samples)
    aqi = 75 + 12 * np.sin(2 * np.pi * timestamps / 48 + 2.0) + 4.0 * np.random.randn(num_samples)
    pm2_5 = 18 + 6 * np.sin(2 * np.pi * timestamps / 36 + 0.8) + 2.5 * np.random.randn(num_samples)
    light = 200 + 150 * np.clip(np.sin(2 * np.pi * timestamps / 24), 0, 1) + 10 * np.random.randn(num_samples)

    return np.column_stack([temperature, humidity, pressure, aqi, pm2_5, light])


def create_sequences(data, lookback=24):
    """Convert raw data into LSTM-ready sequences."""
    X = []
    y = []
    for i in range(len(data) - lookback):
        X.append(data[i : i + lookback])
        y.append(data[i + lookback])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def build_lstm_model(input_shape, output_size):
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(32),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(output_size, activation='linear'),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae'],
    )
    return model


def main():
    model_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(model_dir, exist_ok=True)

    print('Generating dummy historical weather and air quality data...')
    raw_data = generate_dummy_data(num_samples=2200)
    lookback_steps = 24

    X, y = create_sequences(raw_data, lookback=lookback_steps)
    print(f'Generated dataset shape: X={X.shape}, y={y.shape}')

    split_index = int(len(X) * 0.8)
    X_train, X_val = X[:split_index], X[split_index:]
    y_train, y_val = y[:split_index], y[split_index:]

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    n_samples, n_timesteps, n_features = X_train.shape
    X_train_reshaped = X_train.reshape(-1, n_features)
    X_val_reshaped = X_val.reshape(-1, n_features)

    X_train_scaled = scaler_X.fit_transform(X_train_reshaped).reshape(n_samples, n_timesteps, n_features)
    X_val_scaled = scaler_X.transform(X_val_reshaped).reshape(X_val.shape[0], n_timesteps, n_features)
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_val_scaled = scaler_y.transform(y_val)

    model = build_lstm_model(input_shape=(lookback_steps, len(FEATURE_NAMES)), output_size=len(FEATURE_NAMES))
    model.summary()

    print('Training the LSTM model...')
    history = model.fit(
        X_train_scaled,
        y_train_scaled,
        validation_data=(X_val_scaled, y_val_scaled),
        epochs=20,
        batch_size=32,
        verbose=2,
    )

    model_path = os.path.join(model_dir, 'lstm_weather_model.keras')
    print(f'Saving model to {model_path}')
    model.save(model_path)

    scaler_path = os.path.join(model_dir, 'lstm_scalers.npz')
    print(f'Saving scalers to {scaler_path}')
    np.savez(scaler_path,
             scaler_X_mean=scaler_X.mean_,
             scaler_X_scale=scaler_X.scale_,
             scaler_y_mean=scaler_y.mean_,
             scaler_y_scale=scaler_y.scale_)

    print('Training complete.')
    print('Saved feature order:', FEATURE_NAMES)
    print('Model output order matches the feature order above.')


if __name__ == '__main__':
    main()
