import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from model.train import load_model, predict_hourly, load_metrics, load_quantile_models
from collectors.generation_mix import get_generation_mix

_FEATURE_CACHE = None

def prepare_prediction_features(target_date):
    """Generate features matching training data using the same synthetic generators."""
    from data.loader import (generate_synthetic_weather_for_range,
                              generate_synthetic_renewable_for_range,
                              generate_synthetic_genmix_for_range,
                              build_features)
    global _FEATURE_CACHE
    
    pred_date = pd.Timestamp(target_date)
    
    # Generate weather for the full range matching training seed state
    # then select only the target date
    np.random.seed(42)
    full_start = '2025-12-01'
    weather = generate_synthetic_weather_for_range(full_start, target_date)
    weather = weather[pd.to_datetime(weather['date']) == pred_date].copy()
    weather = weather.sort_values('hour').reset_index(drop=True)
    df = weather.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # Renewable indices (synthetic, same as training)
    np.random.seed(42)
    syn_renewable = generate_synthetic_renewable_for_range(2025, 2026)
    syn_renewable = syn_renewable[pd.to_datetime(syn_renewable['date']) == pred_date]
    ren_cols = ['solar_index', 'wind_index', 'renewable_index']
    for c in ren_cols:
        if c in df.columns:
            del df[c]
    df = pd.merge(df, syn_renewable[['datetime'] + ren_cols], on='datetime', how='left')
    for c in ren_cols:
        if c not in df.columns:
            df[c] = 0
        df[c] = df[c].fillna(0)

    # Generation mix (real ENTSO-E if recent, else synthetic matching training)
    genmix_cols = ['nuclear_share', 'thermal_share', 'hydro_share',
                   'solar_share', 'wind_share', 'res_share', 'total_gen_mw']
    genmix = get_generation_mix(days=14)
    genmix_ok = False
    if genmix is not None and len(genmix) > 0:
        genmix_dates = set(pd.to_datetime(genmix['datetime']).dt.strftime('%Y-%m-%d'))
        if target_date in genmix_dates:
            gm = genmix[pd.to_datetime(genmix['datetime']).dt.strftime('%Y-%m-%d') == target_date].copy()
            gm['datetime'] = pd.to_datetime(gm['date'].astype(str) + ' ' + gm['hour'].astype(int).astype(str) + ':00:00')
            for c in genmix_cols:
                if c in df.columns:
                    del df[c]
            df = pd.merge(df, gm[['datetime'] + genmix_cols], on='datetime', how='left')
            genmix_ok = not df[genmix_cols].isna().any().any()

    if not genmix_ok:
        np.random.seed(99)
        syn_genmix = generate_synthetic_genmix_for_range(2025, 2026)
        syn_genmix = syn_genmix[pd.to_datetime(syn_genmix['date']) == pred_date]
        for c in genmix_cols:
            if c in df.columns:
                del df[c]
        df = pd.merge(df, syn_genmix[['datetime'] + genmix_cols], on='datetime', how='left')

    for c in genmix_cols:
        if c not in df.columns:
            df[c] = 0
        df[c] = df[c].fillna(0)

    # Build time features (hour, dayofweek, month, etc.)
    df = build_features(df)
    df['solar_irradiance'] = df['solar_share'] * df['solar_radiation']
    df['solar_intensity'] = df['solar_share'] * df['sin_hour'].clip(lower=0)
    df['is_solar_dip_hour'] = ((df['hour'] >= 10) & (df['hour'] <= 15)).astype(int)
    df['solar_x_hour'] = df['solar_radiation'] * df['hour']
    df['wind_x_hour'] = df['wind_index'] * df['hour']

    # Price lag features from real OREE data
    try:
        oree_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'oree_prices.feather')
        if os.path.exists(oree_path):
            oree_lags = pd.read_feather(oree_path)[['datetime', 'price']].copy()
            oree_lags['datetime'] = pd.to_datetime(oree_lags['datetime'])
            oree_lags = oree_lags.drop_duplicates(subset='datetime')
            for lag_hours, col_name in [(24, 'price_lag_24h'), (168, 'price_lag_168h')]:
                ref = oree_lags.copy()
                ref['datetime'] = ref['datetime'] + pd.Timedelta(hours=lag_hours)
                ref.rename(columns={'price': col_name}, inplace=True)
                df = pd.merge(df, ref[['datetime', col_name]], on='datetime', how='left')
                df[col_name] = df[col_name].fillna(0)

            oree_sorted = oree_lags.sort_values('datetime').set_index('datetime')
            df = df.set_index('datetime', drop=False)
            price_series = oree_sorted['price']
            rolling_24 = price_series.rolling(24, min_periods=1).mean()
            rolling_std_24 = price_series.rolling(24, min_periods=1).std().fillna(0)
            price_delta = price_series.diff(1).fillna(0)
            price_yesterday = price_series.shift(24)
            df['price_rolling_mean_24h'] = df.index.map(lambda x: rolling_24.get(x, 0))
            df['price_rolling_std_24h'] = df.index.map(lambda x: rolling_std_24.get(x, 0))
            df['price_delta_1h'] = df.index.map(lambda x: price_delta.get(x, 0))
            df['price_vs_yesterday'] = df.apply(lambda r: r['price'] - price_yesterday.get(r['name'], r['price']) if r['name'] in price_yesterday.index else 0, axis=1) if len(price_yesterday) > 0 else 0
            df = df.reset_index(drop=True)
        else:
            df['price_lag_24h'] = 0
            df['price_lag_168h'] = 0
            df['price_rolling_mean_24h'] = 0
            df['price_rolling_std_24h'] = 0
            df['price_delta_1h'] = 0
            df['price_vs_yesterday'] = 0
    except Exception:
        df['price_lag_24h'] = 0
        df['price_lag_168h'] = 0
        df['price_rolling_mean_24h'] = 0
        df['price_rolling_std_24h'] = 0
        df['price_delta_1h'] = 0
        df['price_vs_yesterday'] = 0

    return df



def _get_quantile_predictions(features, midday_mask):
    import json as _json
    qm = load_quantile_models()
    if qm is None or not midday_mask.any():
        return None, None, None
    _qc_path = os.path.join(os.path.dirname(__file__), 'quantile_config.json')
    if os.path.exists(_qc_path):
        with open(_qc_path) as _qf:
            quantile_features = _json.load(_qf).get('feature_cols', [])
    else:
        quantile_features = list(features.columns)
    available_q = [c for c in quantile_features if c in features.columns]
    midday_features = features[midday_mask]
    p10 = qm.get(0.10).predict(midday_features[available_q].values) if 0.10 in qm else None
    p50 = qm.get(0.50).predict(midday_features[available_q].values) if 0.50 in qm else None
    p90 = qm.get(0.90).predict(midday_features[available_q].values) if 0.90 in qm else None
    return p10, p50, p90

def _build_results(features, predictions, p10_all=None, p50_all=None, p90_all=None, midday_mask=None):
    results = []
    q_idx = 0
    for idx, (_, row) in enumerate(features.iterrows()):
        hour = int(row['hour'])
        price = float(predictions[idx]) if predictions is not None and idx < len(predictions) else 0
        entry = {
            'hour': f"{(hour % 24) + 1:02d}:00",
            'hour_num': hour,
            'price': max(price, 0.01),
            'temperature': round(float(row.get('temperature', 15) if pd.notna(row.get('temperature', 15)) else 15), 1),
            'humidity': int(row.get('humidity', 50) if pd.notna(row.get('humidity', 50)) else 50),
            'clouds': int(row.get('clouds', 50) if pd.notna(row.get('clouds', 50)) else 50),
            'wind_speed': round(float(row.get('wind_speed', 0) if pd.notna(row.get('wind_speed', 0)) else 0), 1),
            'solar_radiation': round(float(row.get('solar_radiation', 0) if pd.notna(row.get('solar_radiation', 0)) else 0), 1),
        }
        if midday_mask is not None and midday_mask.iloc[idx] and p50_all is not None:
            entry['price_p10'] = round(float(p10_all[q_idx]), 2) if p10_all is not None else None
            entry['price_p50'] = round(float(p50_all[q_idx]), 2) if p50_all is not None else None
            entry['price_p90'] = round(float(p90_all[q_idx]), 2) if p90_all is not None else None
            q_idx += 1
        results.append(entry)
    results.sort(key=lambda x: x['hour_num'])
    return results

def predict_next_day_prices(target_date=None):
    if target_date is None:
        target_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')

    model = load_model()
    if model is None:
        return None

    features = prepare_prediction_features(target_date)
    if features is None:
        return None

    predictions = predict_hourly(model, features)
    midday_mask = features['hour'].between(9, 19)
    p10, p50, p90 = _get_quantile_predictions(features, midday_mask)

    return _build_results(features, predictions, p10, p50, p90, midday_mask)

def predict_with_dates(dates):
    model = load_model()
    if model is None:
        return None

    all_predictions = {}
    for target_date in dates:
        features = prepare_prediction_features(target_date)
        if features is None:
            all_predictions[target_date] = []
            continue
        preds = predict_hourly(model, features)
        midday_mask = features['hour'].between(9, 19)
        p10, p50, p90 = _get_quantile_predictions(features, midday_mask)
        all_predictions[target_date] = _build_results(features, preds, p10, p50, p90, midday_mask)
    return all_predictions

def get_model_stats():
    metrics = load_metrics()
    if metrics is None:
        return {'status': 'not_trained'}
    return {
        'status': 'trained',
        'mae': metrics.get('mae', 0),
        'rmse': metrics.get('rmse', 0),
        'r2': metrics.get('r2', 0),
        'n_train': metrics.get('n_train', 0),
        'n_test': metrics.get('n_test', 0)
    }
