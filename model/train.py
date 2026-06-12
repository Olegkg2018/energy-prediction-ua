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
QUANTILE_MODEL_PATH = os.path.join(MODEL_DIR, 'quantile_model.pkl')
QUANTILE_CONFIG_PATH = os.path.join(MODEL_DIR, 'quantile_config.json')
QUANTILE_P10_PATH = os.path.join(MODEL_DIR, 'quantile_p10.pkl')
QUANTILE_P50_PATH = os.path.join(MODEL_DIR, 'quantile_p50.pkl')
QUANTILE_P90_PATH = os.path.join(MODEL_DIR, 'quantile_p90.pkl')

def train_quantile_models(data_df, force=False):
    """Train XGBoost quantile regression (P10, P50, P90) for midday solar dip hours."""
    models = {}
    for quantile, path in [(0.10, QUANTILE_P10_PATH), (0.50, QUANTILE_P50_PATH), (0.90, QUANTILE_P90_PATH)]:
        if os.path.exists(path) and not force:
            try:
                models[quantile] = joblib.load(path)
                continue
            except Exception:
                pass

    if all(os.path.exists(p) for p in [QUANTILE_P10_PATH, QUANTILE_P50_PATH, QUANTILE_P90_PATH]) and not force:
        try:
            return {0.10: joblib.load(QUANTILE_P10_PATH), 
                    0.50: joblib.load(QUANTILE_P50_PATH), 
                    0.90: joblib.load(QUANTILE_P90_PATH)}
        except Exception:
            pass

    feature_cols = [
        'hour', 'dayofweek', 'month', 'day',
        'is_weekend', 'is_holiday',
        'sin_hour', 'cos_hour', 'sin_month', 'cos_month',
        'sin_dayofyear', 'cos_dayofyear',
        'sin_hour_of_week', 'cos_hour_of_week',
        'temperature', 'temperature_squared',
        'humidity', 'solar_radiation',
        'solar_index', 'wind_index', 'renewable_index',
        'nuclear_share', 'thermal_share', 'hydro_share',
        'solar_share', 'wind_share', 'res_share', 'total_gen_mw',
        'days_since_epoch',
        'solar_irradiance', 'solar_intensity',
        'is_solar_dip_hour',
        'demand_proxy', 'cooling_demand', 'heating_demand',
        'price_lag_24h', 'price_lag_168h',
        'price_rolling_mean_24h', 'price_rolling_std_24h',
        'price_delta_1h', 'price_vs_yesterday',
        'solar_x_hour', 'wind_x_hour',
    ]

    available = [c for c in feature_cols if c in data_df.columns]
    df = data_df.dropna(subset=['price'] + available).copy()
    df = df[df['hour'].between(9, 19)]
    if len(df) < 500:
        return None

    X = df[available].values
    y = df['price'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, shuffle=True
    )

    quantiles = [0.10, 0.50, 0.90]
    for quantile in quantiles:
        model = xgb.XGBRegressor(
            objective='reg:quantileerror',
            quantile_alpha=quantile,
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=2.0,
            min_child_weight=5,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        
        pred_q = model.predict(X_test)
        mae_q = mean_absolute_error(y_test, pred_q)
        
        path = {0.10: QUANTILE_P10_PATH, 0.50: QUANTILE_P50_PATH, 0.90: QUANTILE_P90_PATH}[quantile]
        joblib.dump(model, path)
        models[quantile] = model
        print(f'Quantile P{int(quantile*100)} MAE: {mae_q:.1f}')

    os.makedirs(MODEL_DIR, exist_ok=True)
    import json
    with open(QUANTILE_CONFIG_PATH, 'w') as f:
        json.dump({
            'mae_p10': round(float(mean_absolute_error(y_test, models[0.10].predict(X_test))), 2),
            'mae_p50': round(float(mean_absolute_error(y_test, models[0.50].predict(X_test))), 2),
            'mae_p90': round(float(mean_absolute_error(y_test, models[0.90].predict(X_test))), 2),
            'n_train': int(len(X_train)),
            'n_test': int(len(X_test)),
            'feature_cols': available,
            'quantiles': [0.10, 0.50, 0.90],
            'hours': '9-19',
        }, f)

    return models

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
        'sin_dayofyear', 'cos_dayofyear',
        'sin_hour_of_week', 'cos_hour_of_week',
        'temperature', 'temperature_squared',
        'humidity', 'solar_radiation',
        'solar_index', 'wind_index', 'renewable_index',
        'nuclear_share', 'thermal_share', 'hydro_share',
        'solar_share', 'wind_share', 'res_share', 'total_gen_mw',
        'days_since_epoch',
        'solar_irradiance', 'solar_intensity',
        'is_solar_dip_hour',
        'demand_proxy', 'cooling_demand', 'heating_demand',
        'price_lag_24h', 'price_lag_168h',
        'price_rolling_mean_24h', 'price_rolling_std_24h',
        'price_delta_1h', 'price_vs_yesterday',
        'solar_x_hour', 'wind_x_hour',
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

    xgb_model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=2.0,
        min_child_weight=5,
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    preds = xgb_model.predict(X_test)
    mae_val = mean_absolute_error(y_test, preds)
    r2_val = r2_score(y_test, preds)

    model = xgb_model
    metrics = {
        'mae': round(float(mae_val), 2),
        'rmse': round(float(np.sqrt(mean_squared_error(y_test, preds))), 2),
        'r2': round(float(r2_val), 4),
        'n_train': int(len(X_train)),
        'n_test': int(len(X_test)),
        'feature_cols': available,
    }

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    import json
    with open(MODEL_CONFIG_PATH, 'w') as f:
        json.dump(metrics, f)

    # Train quantile models (P10, P50, P90) for midday solar dip
    train_quantile_models(data_df, force=force)

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

def load_quantile_models():
    models = {}
    for quantile, path in [(0.10, QUANTILE_P10_PATH), (0.50, QUANTILE_P50_PATH), (0.90, QUANTILE_P90_PATH)]:
        if os.path.exists(path):
            try:
                models[quantile] = joblib.load(path)
            except Exception as e:
                print(f"[QUANTILE P{int(quantile*100)}] Load error: {e}")
    return models if models else None

def predict_hourly(model, features_df):
    if model is None:
        return None
    metrics = load_metrics()
    feature_cols = metrics['feature_cols'] if metrics else [
        'hour', 'dayofweek', 'month', 'day',
        'is_weekend', 'is_holiday',
        'sin_hour', 'cos_hour', 'sin_month', 'cos_month',
        'temperature',
        'humidity',
        'solar_radiation',
        'solar_index', 'wind_index', 'renewable_index',
        'nuclear_share', 'thermal_share', 'hydro_share',
        'solar_share', 'wind_share', 'res_share', 'total_gen_mw',
        'days_since_epoch',
        'solar_irradiance', 'solar_intensity',
        'is_solar_dip_hour',
    ]
    available = [c for c in feature_cols if c in features_df.columns]
    missing = [c for c in feature_cols if c not in features_df.columns]
    for c in missing:
        features_df[c] = 0
    X = features_df[feature_cols].values

    if isinstance(model, dict) and 'xgb' in model:
        pred_xgb = model['xgb'].predict(X)
        pred_hgb = model['hgb'].predict(X)
        w_xgb = model.get('weight_xgb', 0.6)
        w_hgb = model.get('weight_hgb', 0.4)
        return pred_xgb * w_xgb + pred_hgb * w_hgb

    return model.predict(X)
