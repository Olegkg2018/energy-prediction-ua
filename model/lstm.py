import os
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

MODEL_DIR = os.path.dirname(__file__)
LSTM_MODEL_PATH = os.path.join(MODEL_DIR, 'lstm_model.keras')
LSTM_CONFIG_PATH = os.path.join(MODEL_DIR, 'lstm_config.json')
LSTM_SCALER_PATH = os.path.join(MODEL_DIR, 'lstm_scaler.pkl')

SEQUENCE_LENGTH = 48
PREDICTION_HORIZON = 24

LSTM_FEATURE_COLS = [
    'hour', 'dayofweek', 'month',
    'sin_hour', 'cos_hour', 'sin_month', 'cos_month',
    'temperature', 'humidity', 'solar_radiation',
    'solar_index', 'wind_index', 'renewable_index',
    'nuclear_share', 'thermal_share', 'hydro_share',
    'solar_share', 'wind_share', 'res_share', 'total_gen_mw',
    'price', 'price_lag_24h', 'price_lag_168h',
    'price_rolling_mean_24h', 'price_rolling_std_24h',
    'price_delta_1h', 'price_vs_yesterday',
]


def _build_sequences(data_df, feature_cols, seq_length=SEQUENCE_LENGTH):
    """Build overlapping sequences for LSTM training."""
    values = data_df[feature_cols].values
    targets = data_df['price'].values

    X, y = [], []
    for i in range(seq_length, len(values)):
        X.append(values[i - seq_length:i])
        y.append(targets[i])

    return np.array(X), np.array(y)


def create_lstm_model(n_features, seq_length=SEQUENCE_LENGTH):
    """Create LSTM model architecture. Predicts single next-hour price."""
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential([
        layers.Input(shape=(seq_length, n_features)),
        layers.LSTM(128, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(64, return_sequences=False),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='huber',
        metrics=['mae']
    )
    return model


def train_lstm(data_df, force=False):
    """Train LSTM model on historical data."""
    if os.path.exists(LSTM_MODEL_PATH) and not force:
        try:
            import json
            with open(LSTM_CONFIG_PATH) as f:
                config = json.load(f)
            return config
        except Exception:
            pass

    try:
        import tensorflow as tf
        from tensorflow import keras
    except ImportError:
        print("[LSTM] TensorFlow not installed, skipping LSTM training")
        return None

    available = [c for c in LSTM_FEATURE_COLS if c in data_df.columns]
    if 'price' not in data_df.columns:
        print("[LSTM] No price column, skipping")
        return None

    df = data_df.dropna(subset=available + ['price']).copy()
    df = df.sort_values('datetime').reset_index(drop=True)

    if len(df) < SEQUENCE_LENGTH + PREDICTION_HORIZON + 500:
        print(f"[LSTM] Not enough data: {len(df)} rows, need {SEQUENCE_LENGTH + PREDICTION_HORIZON + 500}")
        return None

    from sklearn.preprocessing import StandardScaler

    feature_idx = [available.index(c) for c in available]
    price_idx = available.index('price') if 'price' in available else -1

    values = df[available].values.astype(np.float32)
    targets = df['price'].values.astype(np.float32)

    scaler = StandardScaler()
    values_scaled = scaler.fit_transform(values)

    split_idx = int(len(values_scaled) * 0.85)
    train_data = values_scaled[:split_idx]
    train_targets = targets[:split_idx]
    test_data = values_scaled[split_idx:]
    test_targets = targets[split_idx:]

    X_train, y_train = [], []
    for i in range(SEQUENCE_LENGTH, len(train_data)):
        X_train.append(train_data[i - SEQUENCE_LENGTH:i])
        y_train.append(train_targets[i])

    X_test, y_test = [], []
    for i in range(SEQUENCE_LENGTH, len(test_data)):
        X_test.append(test_data[i - SEQUENCE_LENGTH:i])
        y_test.append(test_targets[i])

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    if len(X_train) < 100 or len(X_test) < 20:
        print(f"[LSTM] Not enough sequences: train={len(X_train)}, test={len(X_test)}")
        return None

    print(f"[LSTM] Training: {len(X_train)} sequences, test: {len(X_test)} sequences")
    print(f"[LSTM] Features: {len(available)}, sequence length: {SEQUENCE_LENGTH}")

    n_features = X_train.shape[2]
    model = create_lstm_model(n_features)

    callbacks = [
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=64,
        callbacks=callbacks,
        verbose=1
    )

    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)

    joblib.dump(scaler, LSTM_SCALER_PATH)
    model.save(LSTM_MODEL_PATH)

    import json
    config = {
        'mae': round(float(test_mae), 2),
        'n_train': len(X_train),
        'n_test': len(X_test),
        'features': available,
        'sequence_length': SEQUENCE_LENGTH,
        'n_features': n_features,
        'epochs_trained': len(history.history['loss']),
    }
    with open(LSTM_CONFIG_PATH, 'w') as f:
        json.dump(config, f)

    print(f"[LSTM] Test MAE: {test_mae:.2f}")
    return config


def load_lstm():
    """Load trained LSTM model and scaler."""
    if not os.path.exists(LSTM_MODEL_PATH) or not os.path.exists(LSTM_SCALER_PATH):
        return None, None, None

    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(LSTM_MODEL_PATH)
        scaler = joblib.load(LSTM_SCALER_PATH)
        import json
        with open(LSTM_CONFIG_PATH) as f:
            config = json.load(f)
        return model, scaler, config
    except Exception as e:
        print(f"[LSTM] Load error: {e}")
        return None, None, None


def predict_lstm(model, scaler, recent_data, feature_cols):
    """Predict next 24 hours using LSTM autoregressively.

    Args:
        model: trained Keras model
        scaler: fitted StandardScaler
        recent_data: DataFrame with at least SEQUENCE_LENGTH hours of features
        feature_cols: list of feature column names
    Returns:
        array of 24 predicted prices, or None
    """
    if model is None or scaler is None:
        return None

    available = [c for c in feature_cols if c in recent_data.columns]
    if len(available) < 5:
        return None

    values = recent_data[available].values.astype(np.float32)
    if len(values) < SEQUENCE_LENGTH:
        return None

    seq = values[-SEQUENCE_LENGTH:].copy()
    price_idx = feature_cols.index('price') if 'price' in feature_cols else -1
    if price_idx < 0 or price_idx >= seq.shape[1]:
        return None

    preds = []
    for _ in range(PREDICTION_HORIZON):
        seq_scaled = scaler.transform(seq)
        seq_batch = seq_scaled[np.newaxis, :, :]
        pred = model.predict(seq_batch, verbose=0)[0, 0]
        preds.append(pred)

        new_row = seq[-1].copy()
        new_row[price_idx] = pred
        seq = np.vstack([seq[1:], new_row[np.newaxis, :]])

    return np.array(preds)
