import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from model.train import load_model, predict_hourly, load_metrics, load_quantile_models
from collectors.generation_mix import get_generation_mix

_FEATURE_CACHE = None

def _get_real_weather_for_target(target_date):
    """Try to get real weather forecast for target date from OpenWeather API."""
    try:
        from collectors.weather import get_forecast
        forecast = get_forecast()
        if forecast is not None and len(forecast) > 0:
            forecast['date'] = pd.to_datetime(forecast['date']).dt.strftime('%Y-%m-%d')
            target_weather = forecast[forecast['date'] == target_date]
            if len(target_weather) >= 20:
                return target_weather.sort_values('hour').reset_index(drop=True)
    except Exception:
        pass
    return None


def _get_real_renewable_for_target(target_date):
    """Try to get real renewable index forecast for target date."""
    try:
        from collectors.renewable_index import get_renewable_forecast
        forecast = get_renewable_forecast(days=3)
        if forecast is not None and len(forecast) > 0:
            target_ren = [r for r in forecast if r.get('date') == target_date]
            if len(target_ren) >= 20:
                return pd.DataFrame(target_ren).sort_values('hour').reset_index(drop=True)
    except Exception:
        pass
    return None


def _compute_hour_specific_price_features(oree_lags, pred_date):
    """Compute hour-specific price features from historical patterns (no look-ahead)."""
    result = {}
    if len(oree_lags) == 0:
        for c in ['price_rolling_mean_24h', 'price_rolling_std_24h', 'price_delta_1h', 'price_vs_yesterday']:
            result[c] = [0] * 24
        return result

    oree_sorted = oree_lags.sort_values('datetime').set_index('datetime')
    price_series = oree_sorted['price']

    for hour in range(24):
        last_ts = oree_sorted.index[-1]
        last_price = price_series.get(last_ts, 0)

        same_hour_prices = []
        for day_offset in range(1, 8):
            check_date = pred_date - pd.Timedelta(days=day_offset)
            ts = check_date + pd.Timedelta(hours=hour)
            if ts in price_series.index:
                same_hour_prices.append(price_series.get(ts, 0))

        if same_hour_prices:
            result.setdefault('price_rolling_mean_24h', []).append(np.mean(same_hour_prices))
            deltas = [same_hour_prices[i] - same_hour_prices[i-1] for i in range(1, len(same_hour_prices))]
            result.setdefault('price_delta_1h', []).append(np.mean(deltas) if deltas else 0)
            yesterday_same_hour = price_series.get(last_ts - pd.Timedelta(hours=24), None)
            if yesterday_same_hour is not None:
                result.setdefault('price_vs_yesterday', []).append(last_price - yesterday_same_hour)
            else:
                result.setdefault('price_vs_yesterday', []).append(0)
        else:
            result.setdefault('price_rolling_mean_24h', []).append(
                price_series.rolling(24, min_periods=1).mean().get(last_ts, 0) if len(price_series) > 0 else 0)
            result.setdefault('price_delta_1h', []).append(
                price_series.diff(1).fillna(0).get(last_ts, 0) if len(price_series) > 0 else 0)
            result.setdefault('price_vs_yesterday', []).append(0)

    rolling_all = price_series.rolling(24, min_periods=1).mean()
    rolling_std_all = price_series.rolling(24, min_periods=1).std().fillna(0)
    last_ts = oree_sorted.index[-1]
    result['price_rolling_std_24h'] = [rolling_std_all.get(last_ts, 0)] * 24

    return result


def prepare_prediction_features(target_date):
    """Generate features using real weather/renewable data when available."""
    from data.loader import (generate_synthetic_weather_for_range,
                              generate_synthetic_renewable_for_range,
                              generate_synthetic_genmix_for_range,
                              build_features)
    global _FEATURE_CACHE

    pred_date = pd.Timestamp(target_date)

    # --- Weather: try real forecast first, fall back to synthetic ---
    real_weather = _get_real_weather_for_target(target_date)
    if real_weather is not None:
        df = real_weather.copy()
        if 'datetime' not in df.columns:
            df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['hour'].astype(str) + ':00:00')
        df['datetime'] = pd.to_datetime(df['datetime'])
        if 'solar_radiation' not in df.columns:
            df['solar_radiation'] = 0.0
    else:
        np.random.seed(42)
        full_start = '2025-12-01'
        weather = generate_synthetic_weather_for_range(full_start, target_date)
        weather = weather[pd.to_datetime(weather['date']) == pred_date].copy()
        weather = weather.sort_values('hour').reset_index(drop=True)
        df = weather.copy()
        df['datetime'] = pd.to_datetime(df['datetime'])

    # --- Renewable indices: try real forecast first, fall back to synthetic ---
    real_ren = _get_real_renewable_for_target(target_date)
    ren_cols = ['solar_index', 'wind_index', 'renewable_index']
    if real_ren is not None and len(real_ren) > 0:
        for c in ren_cols:
            if c in df.columns:
                del df[c]
        if 'datetime' not in real_ren.columns:
            real_ren['datetime'] = pd.to_datetime(real_ren['date'] + ' ' + real_ren['hour'].astype(str) + ':00:00')
        df = pd.merge(df, real_ren[['datetime'] + ren_cols], on='datetime', how='left')
    else:
        np.random.seed(42)
        syn_renewable = generate_synthetic_renewable_for_range(2025, 2026)
        syn_renewable = syn_renewable[pd.to_datetime(syn_renewable['date']) == pred_date]
        for c in ren_cols:
            if c in df.columns:
                del df[c]
        df = pd.merge(df, syn_renewable[['datetime'] + ren_cols], on='datetime', how='left')

    for c in ren_cols:
        if c not in df.columns:
            df[c] = 0
        df[c] = df[c].fillna(0)

    # --- Generation mix (real ENTSO-E if available, else synthetic) ---
    genmix_cols = ['nuclear_share', 'thermal_share', 'hydro_share',
                   'solar_share', 'wind_share', 'res_share', 'total_gen_mw']
    genmix = get_generation_mix(days=14)
    genmix_ok = False
    if genmix is not None and len(genmix) > 0:
        genmix['date_str'] = pd.to_datetime(genmix['datetime']).dt.strftime('%Y-%m-%d')
        if target_date in genmix['date_str'].values:
            gm = genmix[genmix['date_str'] == target_date].copy()
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

    # Build time features
    df = build_features(df)
    df['solar_irradiance'] = df['solar_share'] * df['solar_radiation']
    df['solar_intensity'] = df['solar_share'] * df['sin_hour'].clip(lower=0)
    df['is_solar_dip_hour'] = ((df['hour'] >= 10) & (df['hour'] <= 15)).astype(int)
    df['solar_x_hour'] = df['solar_radiation'] * df['hour']
    df['wind_x_hour'] = df['wind_index'] * df['hour']

    # --- Price lag features (NO look-ahead: exclude target date) ---
    try:
        oree_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'oree_prices.feather')
        if os.path.exists(oree_path):
            oree_lags = pd.read_feather(oree_path)[['datetime', 'price']].copy()
            oree_lags['datetime'] = pd.to_datetime(oree_lags['datetime'])
            oree_lags = oree_lags.drop_duplicates(subset='datetime')
            oree_lags = oree_lags[oree_lags['datetime'] < pred_date].copy()

            for lag_hours, col_name in [(24, 'price_lag_24h'), (168, 'price_lag_168h')]:
                ref = oree_lags.copy()
                ref['datetime'] = ref['datetime'] + pd.Timedelta(hours=lag_hours)
                ref.rename(columns={'price': col_name}, inplace=True)
                df = pd.merge(df, ref[['datetime', col_name]], on='datetime', how='left')
                df[col_name] = df[col_name].fillna(0)

            hour_features = _compute_hour_specific_price_features(oree_lags, pred_date)
            df = df.sort_values('hour').reset_index(drop=True)
            for col_name, values in hour_features.items():
                df[col_name] = values
        else:
            df['price_lag_24h'] = 0
            df['price_lag_168h'] = 0
            df['price_rolling_mean_24h'] = 0
            df['price_rolling_std_24h'] = 0
            df['price_delta_1h'] = 0
            df['price_vs_yesterday'] = 0
    except Exception as e:
        import traceback
        traceback.print_exc()
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
