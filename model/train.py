import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

MODEL_DIR = os.path.join(os.path.dirname(__file__))
MODEL_PATH = os.path.join(MODEL_DIR, 'model.pkl')
MODEL_CONFIG_PATH = os.path.join(MODEL_DIR, 'model_config.json')

def train_model(data_df, force=False):
    if os.path.exists(MODEL_PATH) and not force:
        try:
            model = joblib.load(MODEL_PATH)
            metrics = load_metrics()
            return model, metrics
        except Exception:
            pass

    feature_cols = [
        'hour', 'dayofweek', 'month', 'day',
        'is_weekend', 'is_holiday',
        'sin_hour', 'cos_hour', 'sin_month', 'cos_month',
        'temperature',
        'solar_radiation',
        'solar_index', 'wind_index', 'renewable_index',
        'nuclear_share', 'thermal_share', 'hydro_share',
        'solar_share', 'wind_share', 'res_share', 'total_gen_mw',
        'days_since_epoch',
    ]
    available = [c for c in feature_cols if c in data_df.columns]
    df = data_df.dropna(subset=['price'] + available).copy()
    if len(df) < 1000:
        return None, None

    X = df[available].values
    y = df['price'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, shuffle=True
    )

    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=2.0,
        min_child_weight=5,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    y_pred = model.predict(X_test)
    metrics = {
        'mae': round(float(mean_absolute_error(y_test, y_pred)), 2),
        'rmse': round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 2),
        'r2': round(float(r2_score(y_test, y_pred)), 4),
        'n_train': int(len(X_train)),
        'n_test': int(len(X_test)),
        'feature_cols': available
    }

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    import json
    with open(MODEL_CONFIG_PATH, 'w') as f:
        json.dump(metrics, f)

    return model, metrics

def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        print(f"[MODEL] Load error: {e}")
        return None

def load_metrics():
    if not os.path.exists(MODEL_CONFIG_PATH):
        return None
    import json
    with open(MODEL_CONFIG_PATH, 'r') as f:
        return json.load(f)

def predict_hourly(model, features_df):
    if model is None:
        return None
    metrics = load_metrics()
    feature_cols = metrics['feature_cols'] if metrics else [
        'hour', 'dayofweek', 'month', 'day',
        'is_weekend', 'is_holiday',
        'sin_hour', 'cos_hour', 'sin_month', 'cos_month',
        'temperature',
        'solar_radiation',
        'solar_index', 'wind_index', 'renewable_index',
        'nuclear_share', 'thermal_share', 'hydro_share',
        'solar_share', 'wind_share', 'res_share', 'total_gen_mw',
        'days_since_epoch',
    ]
    available = [c for c in feature_cols if c in features_df.columns]
    missing = [c for c in feature_cols if c not in features_df.columns]
    for c in missing:
        features_df[c] = 0
    X = features_df[feature_cols].values
    return model.predict(X)
