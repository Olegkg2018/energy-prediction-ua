import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import TimeSeriesSplit
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
ENSEMBLE_CONFIG_PATH = os.path.join(MODEL_DIR, 'ensemble_config.json')

FEATURE_COLS = [
    # Time encodings (14)
    'hour', 'dayofweek', 'month', 'day',
    'is_weekend', 'is_holiday',
    'sin_hour', 'cos_hour', 'sin_month', 'cos_month',
    'sin_dayofyear', 'cos_dayofyear',
    'sin_hour_of_week', 'cos_hour_of_week',
    # Fourier harmonics (10)
    'sin_hour_2', 'cos_hour_2', 'sin_hour_3', 'cos_hour_3',
    'sin_month_2', 'cos_month_2',
    'sin_dayofyear_2', 'cos_dayofyear_2',
    'sin_week_of_year', 'cos_week_of_year',
    # Calendar enhancements (12)
    'is_month_start', 'is_month_end', 'is_quarter_start', 'is_quarter_end',
    'day_of_month', 'week_of_year', 'quarter', 'season',
    'is_heating_season', 'is_cooling_season',
    'is_week_before_holiday', 'is_bridge_day',
    'day_before_holiday', 'day_after_holiday',
    'days_to_next_holiday', 'days_since_last_holiday',
    # Weather (10)
    'temperature', 'temperature_squared',
    'humidity', 'solar_radiation', 'wind_speed', 'clouds',
    'heating_degree_hour', 'cooling_degree_hour',
    'temp_anomaly', 'cloud_cover',
    # Weather interactions (6)
    'demand_proxy', 'cooling_demand', 'heating_demand',
    'temp_x_hour', 'temp_x_solar', 'solar_x_clouds',
    # Renewable & generation (14)
    'solar_index', 'wind_index', 'renewable_index',
    'nuclear_share', 'thermal_share', 'hydro_share',
    'solar_share', 'wind_share', 'res_share', 'total_gen_mw',
    'renewable_share_forecast', 'thermal_x_hour',
    'nuclear_x_hour', 'hydro_x_hour',
    # Solar/wind interactions (5)
    'solar_irradiance', 'solar_intensity', 'is_solar_dip_hour',
    'solar_x_hour', 'wind_x_hour',
    # Generation interactions (3)
    'res_x_temp', 'total_gen_x_hour', 'wind_x_renewable',
    # Trend (1)
    'days_since_epoch',
    # Extended price lags (9)
    'price_lag_2h', 'price_lag_3h', 'price_lag_6h',
    'price_lag_12h', 'price_lag_24h', 'price_lag_48h',
    'price_lag_168h', 'price_lag_336h', 'price_lag_504h',
    # Rolling statistics (17)
    'price_rolling_mean_24h', 'price_rolling_std_24h',
    'price_rolling_min_24h', 'price_rolling_max_24h',
    'price_rolling_mean_48h', 'price_rolling_std_48h',
    'price_rolling_min_48h', 'price_rolling_max_48h',
    'price_rolling_mean_168h', 'price_rolling_std_168h',
    'price_rolling_min_168h', 'price_rolling_max_168h',
    'price_rolling_median_24h',
    'price_rolling_skew_168h', 'price_rolling_kurt_168h',
    'price_range_48h', 'price_range_168h',
    # EWM (2)
    'price_ewm_12h', 'price_ewm_48h',
    # Price deltas (8)
    'price_delta_1h', 'price_delta_3h', 'price_delta_6h', 'price_delta_24h',
    'price_vs_yesterday', 'price_vs_last_week',
    'price_same_hour_yesterday', 'price_yoy_ratio',
    # Technical indicators (11)
    'price_ema_6', 'price_ema_12', 'price_ema_24', 'price_ema_diff',
    'price_bb_pctb_24', 'price_bb_pctb_48',
    'price_momentum_24', 'price_momentum_48',
    'price_roc_12', 'price_roc_24',
    # Gas & fuel (6)
    'ttf_eur_mwh', 'gas_uah_mwh',
    'gas_momentum_7d', 'gas_rolling_std_7d',
    # Market structure (2)
    'vrd_rdn_spread', 'vrd_rdn_ratio',
]

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

    available = [c for c in FEATURE_COLS if c in data_df.columns]
    df = data_df.dropna(subset=['price'] + available).copy()
    df = df[df['hour'].between(9, 19)]
    if len(df) < 500:
        return None

    X = df[available].values
    y = df['price'].values

    # TEMPORAL split: train on first 85%, test on last 15% (no shuffle = no future leak)
    split_idx = int(len(X) * 0.85)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    quantiles = [0.10, 0.50, 0.90]
    for quantile in quantiles:
        model = xgb.XGBRegressor(
            objective='reg:quantileerror',
            quantile_alpha=quantile,
            n_estimators=800,
            max_depth=8,
            learning_rate=0.03,
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


def cross_validate_timeseries(data_df, n_splits=5):
    """TimeSeriesSplit cross-validation for robust model evaluation."""
    available = [c for c in FEATURE_COLS if c in data_df.columns]
    df = data_df.dropna(subset=['price'] + available).copy()
    if len(df) < 2000:
        print(f"[CV] Not enough data: {len(df)} rows")
        return None

    X = df[available].values
    y = df['price'].values

    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = xgb.XGBRegressor(
            n_estimators=400,
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
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        fold_metrics.append({
            'fold': fold + 1,
            'mae': round(float(mae), 2),
            'rmse': round(float(rmse), 2),
            'r2': round(float(r2), 4),
            'n_train': len(train_idx),
            'n_test': len(test_idx),
        })
        print(f"[CV] Fold {fold+1}/{n_splits}: MAE={mae:.2f}, R2={r2:.4f}")

    avg_mae = np.mean([m['mae'] for m in fold_metrics])
    avg_rmse = np.mean([m['rmse'] for m in fold_metrics])
    avg_r2 = np.mean([m['r2'] for m in fold_metrics])

    result = {
        'folds': fold_metrics,
        'avg_mae': round(float(avg_mae), 2),
        'avg_rmse': round(float(avg_rmse), 2),
        'avg_r2': round(float(avg_r2), 4),
        'n_splits': n_splits,
    }
    print(f"[CV] Average: MAE={avg_mae:.2f}, RMSE={avg_rmse:.2f}, R2={avg_r2:.4f}")
    return result


def train_model(data_df, force=False):
    if os.path.exists(MODEL_PATH) and not force:
        try:
            model = joblib.load(MODEL_PATH)
            metrics = load_metrics()
            return model, metrics
        except Exception:
            pass

    available = [c for c in FEATURE_COLS if c in data_df.columns]
    df = data_df.dropna(subset=['price'] + available).copy()
    if len(df) < 1000:
        return None, None

    X = df[available].values
    y = df['price'].values

    # TEMPORAL split: train on first 85%, test on last 15% (no shuffle = no future leak)
    split_idx = int(len(X) * 0.85)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    xgb_model = xgb.XGBRegressor(
        n_estimators=800,
        max_depth=8,
        learning_rate=0.03,
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

    # Train LSTM
    from model.lstm import train_lstm, load_lstm, predict_lstm, LSTM_FEATURE_COLS
    lstm_config = train_lstm(data_df, force=force)
    lstm_model, lstm_scaler, _ = load_lstm()

    # Ensemble weights: find optimal on test set
    w_xgb, w_lstm = 0.6, 0.4
    if lstm_model is not None:
        lstm_features = [c for c in LSTM_FEATURE_COLS if c in df.columns]
        recent_for_lstm = df.iloc[split_idx - 48:split_idx + len(y_test)]
        lstm_preds_all = []

        for start in range(0, len(y_test), 24):
            end = min(start + 24, len(y_test))
            recent_slice = recent_for_lstm.iloc[start:start + 48] if start + 48 <= len(recent_for_lstm) else recent_for_lstm.iloc[-48:]
            if len(recent_slice) >= 48:
                lp = predict_lstm(lstm_model, lstm_scaler, recent_slice, lstm_features)
                if lp is not None:
                    lstm_preds_all.extend(lp[:end - start])
                else:
                    lstm_preds_all.extend(preds[start:end])
            else:
                lstm_preds_all.extend(preds[start:end])

        if len(lstm_preds_all) == len(y_test):
            best_mae = mae_val
            best_w = (1.0, 0.0)
            for w in np.arange(0.3, 0.9, 0.05):
                ensemble_pred = w * preds + (1 - w) * np.array(lstm_preds_all)
                ens_mae = mean_absolute_error(y_test, ensemble_pred)
                if ens_mae < best_mae:
                    best_mae = ens_mae
                    best_w = (w, 1 - w)
            w_xgb, w_lstm = best_w
            ensemble_pred = w_xgb * preds + w_lstm * np.array(lstm_preds_all)
            mae_val = mean_absolute_error(y_test, ensemble_pred)
            r2_val = r2_score(y_test, ensemble_pred)
            print(f"[ENSEMBLE] Optimal weights: XGB={w_xgb:.2f}, LSTM={w_lstm:.2f}, MAE={mae_val:.2f}")

            import json
            with open(ENSEMBLE_CONFIG_PATH, 'w') as f:
                json.dump({
                    'w_xgb': round(float(w_xgb), 3),
                    'w_lstm': round(float(w_lstm), 3),
                    'mae_ensemble': round(float(mae_val), 2),
                    'mae_xgb_only': round(float(mean_absolute_error(y_test, preds)), 2),
                    'lstm_mae': lstm_config.get('mae', None) if lstm_config else None,
                }, f)

    model = {
        'xgb': xgb_model,
        'weight_xgb': w_xgb,
        'weight_lstm': w_lstm,
    }
    metrics = {
        'mae': round(float(mae_val), 2),
        'rmse': round(float(np.sqrt(mean_squared_error(y_test, preds))), 2),
        'r2': round(float(r2_val), 4),
        'n_train': int(len(X_train)),
        'n_test': int(len(X_test)),
        'feature_cols': available,
        'has_lstm': lstm_model is not None,
        'ensemble_weights': {'xgb': w_xgb, 'lstm': w_lstm},
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
    feature_cols = metrics['feature_cols'] if metrics else FEATURE_COLS
    available = [c for c in feature_cols if c in features_df.columns]
    missing = [c for c in feature_cols if c not in features_df.columns]
    for c in missing:
        features_df[c] = 0
    X = features_df[feature_cols].values

    if isinstance(model, dict) and 'xgb' in model:
        pred_xgb = model['xgb'].predict(X)
        w_xgb = model.get('weight_xgb', 0.6)
        w_lstm = model.get('weight_lstm', 0.4)

        if w_lstm > 0:
            from model.lstm import load_lstm, predict_lstm, LSTM_FEATURE_COLS
            lstm_model, lstm_scaler, _ = load_lstm()
            if lstm_model is not None:
                lstm_features = [c for c in LSTM_FEATURE_COLS if c in features_df.columns]
                lstm_preds = predict_lstm(lstm_model, lstm_scaler, features_df, lstm_features)
                if lstm_preds is not None and len(lstm_preds) >= len(pred_xgb):
                    return w_xgb * pred_xgb + w_lstm * lstm_preds[:len(pred_xgb)]

        return pred_xgb

    return model.predict(X)
