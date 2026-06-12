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

    # Price lag + rolling features from real OREE data
    try:
        oree_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'oree_prices.feather')
        if os.path.exists(oree_path):
            oree_lags = pd.read_feather(oree_path)[['datetime', 'price']].copy()
            oree_lags['datetime'] = pd.to_datetime(oree_lags['datetime'])
            oree_lags = oree_lags.drop_duplicates(subset='datetime').sort_values('datetime')
            for lag_hours, col_name in [(24, 'price_lag_24h'), (168, 'price_lag_168h')]:
                ref = oree_lags.copy()
                ref['datetime'] = ref['datetime'] + pd.Timedelta(hours=lag_hours)
                ref.rename(columns={'price': col_name}, inplace=True)
                df = pd.merge(df, ref[['datetime', col_name]], on='datetime', how='left')
                df[col_name] = df[col_name].fillna(0)
            # Rolling stats from real OREE data
            oree_hist = oree_lags[['datetime', 'price']].copy()
            oree_hist['price_rolling_mean_24h'] = oree_hist['price'].rolling(window=24, min_periods=1).mean()
            oree_hist['price_rolling_std_24h'] = oree_hist['price'].rolling(window=24, min_periods=1).std().fillna(0)
            oree_hist['price_delta_1h'] = oree_hist['price'].diff(1).fillna(0)
            # For price_vs_yesterday, shift by 24h
            oree_for_vs = oree_lags[['datetime', 'price']].copy()
            oree_for_vs['datetime'] = oree_for_vs['datetime'] + pd.Timedelta(hours=24)
            oree_for_vs.rename(columns={'price': 'price_lag_24h_tmp'}, inplace=True)
            oree_hist = pd.merge(oree_hist, oree_for_vs, on='datetime', how='left')
            oree_hist['price_vs_yesterday'] = (oree_hist['price'] - oree_hist['price_lag_24h_tmp']).fillna(0)
            oree_hist.drop(columns=['price_lag_24h_tmp'], inplace=True)
            for col in ['price_rolling_mean_24h', 'price_rolling_std_24h', 'price_delta_1h', 'price_vs_yesterday']:
                df = pd.merge(df, oree_hist[['datetime', col]], on='datetime', how='left')
                df[col] = df[col].fillna(0)
        else:
            for col in ['price_lag_24h', 'price_lag_168h', 'price_rolling_mean_24h',
                        'price_rolling_std_24h', 'price_delta_1h', 'price_vs_yesterday']:
                df[col] = 0
    except Exception:
        for col in ['price_lag_24h', 'price_lag_168h', 'price_rolling_mean_24h',
                    'price_rolling_std_24h', 'price_delta_1h', 'price_vs_yesterday']:
            df[col] = 0

    return df



def _smooth_prices(results):
    prices = [r['price'] for r in results]
    n = len(prices)
    smoothed = []
    for i in range(n):
        w = prices[i]
        t = 1
        for d in [1, 2]:
            if i - d >= 0:
                w += prices[i - d] * (1 / d)
                t += 1 / d
            if i + d < n:
                w += prices[i + d] * (1 / d)
                t += 1 / d
        smoothed.append(w / t)
    for i, r in enumerate(results):
        r['price'] = round(smoothed[i], 2)

def predict_next_day_prices(target_date=None):
    if target_date is None:
        target_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')

    model = load_model()
    quantile_models = load_quantile_models()
    if model is None:
        return None

    features = prepare_prediction_features(target_date)
    if features is None:
        return None

    predictions = predict_hourly(model, features)
    # Use quantile models (P10/P50/P90) for extended midday hours (9-19) with boundary blending
    if quantile_models is not None:
        midday_mask = features['hour'].between(9, 19)
        if midday_mask.any():
            quantile_features = [
                'hour', 'dayofweek', 'month', 'day',
                'is_weekend', 'is_holiday',
                'sin_hour', 'cos_hour', 'sin_month', 'cos_month',
                'sin_dayofyear', 'cos_dayofyear',
                'sin_hour_of_week', 'cos_hour_of_week',
                'temperature', 'temperature_squared',
                'humidity',
                'solar_radiation',
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
            available_q = [c for c in quantile_features if c in features.columns]
            midday_features = features[midday_mask]
            
            p10_preds = quantile_models[0.10].predict(midday_features[available_q].values) if 0.10 in quantile_models else None
            p50_preds = quantile_models[0.50].predict(midday_features[available_q].values) if 0.50 in quantile_models else None
            p90_preds = quantile_models[0.90].predict(midday_features[available_q].values) if 0.90 in quantile_models else None
            
            pred_idx = 0
            for idx in features.index:
                if midday_mask.loc[idx]:
                    h = features.loc[idx, 'hour']
                    p50 = p50_preds[pred_idx] if p50_preds is not None else None
                    p10 = p10_preds[pred_idx] if p10_preds is not None else None
                    p90 = p90_preds[pred_idx] if p90_preds is not None else None
                    pred_idx += 1
                    
                    # Boundary blending: 9,19 use 50% quantile + 50% ensemble
                    if h in [9, 19]:
                        if p50 is not None:
                            predictions[idx] = 0.5 * p50 + 0.5 * predictions[idx]
                    else:
                        if p50 is not None:
                            predictions[idx] = p50

    results = []
    for idx, (_, row) in enumerate(features.iterrows()):
        hour = int(row['hour'])
        price = float(predictions[idx]) if predictions is not None and idx < len(predictions) else 0
        results.append({
            'hour': f"{(hour % 24) + 1:02d}:00",
            'hour_num': hour,
            'price': max(price, 0.01),
            'temperature': round(float(row.get('temperature', 15) if pd.notna(row.get('temperature', 15)) else 15), 1),
            'humidity': int(row.get('humidity', 50) if pd.notna(row.get('humidity', 50)) else 50),
            'clouds': int(row.get('clouds', 50) if pd.notna(row.get('clouds', 50)) else 50),
            'wind_speed': round(float(row.get('wind_speed', 0))),
            'solar_radiation': round(float(row.get('solar_radiation', 0)), 1)
        })
    results.sort(key=lambda x: x['hour_num'])
    return results

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
        results = []
        for idx, (_, row) in enumerate(features.iterrows()):
            price = float(preds[idx]) if preds is not None and idx < len(preds) else 0
            results.append({
                'hour': f"{int(row['hour']) + 1:02d}:00",
                'price': max(price, 0.01),
                'temperature': round(float(row.get('temperature', 15)), 1)
            })
        results = sorted(results, key=lambda x: x['hour'])
        all_predictions[target_date] = results
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
