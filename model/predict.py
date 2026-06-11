import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from model.train import load_model, predict_hourly, load_metrics, load_quantile_models
from collectors.weather import get_forecast
from collectors.renewable_index import get_renewable_indices
from collectors.generation_mix import get_generation_mix
from data.loader import build_features, HOLIDAYS_2025

def prepare_prediction_features(weather_forecast):
    df = pd.DataFrame(weather_forecast)
    if df.empty:
        return None
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['hour'].astype(str) + ':00:00')
    # Build time features first (before adding renewable/genmix)
    # Remove any pre-existing renewable/genmix columns to avoid merge suffix issues
    skip_on_merge = {'solar_index', 'wind_index', 'renewable_index',
                     'nuclear_share', 'thermal_share', 'hydro_share',
                     'solar_share', 'wind_share', 'res_share', 'total_gen_mw'}
    for c in skip_on_merge:
        if c in df.columns:
            del df[c]
    df = build_features(df)
    if 'temperature' not in df.columns:
        df['temperature'] = 15
    df['solar_radiation'] = df.get('solar_radiation', 0)
    df['clouds'] = df.get('clouds', 50)
    df['wind_speed'] = df.get('wind_speed', 0)
    df['humidity'] = df.get('humidity', 50)

    # Helper: merge and overwrite existing columns with same names
    def merge_overwrite(df, right, on, columns):
        """Merge and ensure right-side columns overwrite left-side ones."""
        cols_to_overwrite = [c for c in right.columns if c != on and c in df.columns]
        for c in cols_to_overwrite:
            del df[c]
        return pd.merge(df, right, on=on, how='left')

    # --- Renewable indices ---
    renewable_idx = get_renewable_indices()
    if renewable_idx is not None and len(renewable_idx) > 0:
        ri = renewable_idx.copy()
        ri['date'] = pd.to_datetime(ri['date']).dt.strftime('%Y-%m-%d')
        ri['datetime'] = pd.to_datetime(ri['date'] + ' ' + ri['hour'].astype(str) + ':00:00')
        ri_sub = ri[['datetime', 'solar_index', 'wind_index', 'renewable_index']]
        ri_sub = ri_sub[[c for c in ri_sub.columns if c in ri_sub.columns]]
        df = merge_overwrite(df, ri_sub, 'datetime', ['solar_index', 'wind_index', 'renewable_index'])
        ren_cols = ['solar_index', 'wind_index', 'renewable_index']
        if all(c in df.columns for c in ren_cols) and df[ren_cols].isna().any().any():
            ri['hour'] = ri['datetime'].dt.hour
            all_dates = sorted(ri['date'].unique(), reverse=True)
            proxy = {}
            for date in all_dates:
                day_df = ri[ri['date'] == date]
                for _, row in day_df.iterrows():
                    h = int(row['hour'])
                    if h not in proxy:
                        proxy[h] = {c: row[c] for c in ren_cols}
                if len(proxy) >= 24:
                    break
            for idx in df[df[ren_cols[0]].isna()].index:
                h = int(df.loc[idx, 'hour'])
                if h in proxy:
                    for c in ren_cols:
                        df.loc[idx, c] = proxy[h][c]
    for c in ['solar_index', 'wind_index', 'renewable_index']:
        if c not in df.columns:
            df[c] = 0
        df[c] = df[c].fillna(0)

    # --- Generation mix ---
    genmix = get_generation_mix(days=14)
    if genmix is not None and len(genmix) > 0:
        gm = genmix.copy()
        gm['date'] = pd.to_datetime(gm['date']).dt.strftime('%Y-%m-%d')
        gm['datetime'] = pd.to_datetime(gm['date'] + ' ' + gm['hour'].astype(int).astype(str) + ':00:00')
        gm_sub = gm[['datetime', 'nuclear_share', 'thermal_share', 'hydro_share',
                      'solar_share', 'wind_share', 'res_share', 'total_gen_mw']]
        gm_sub = gm_sub[[c for c in gm_sub.columns if c in gm_sub.columns]]
        df = merge_overwrite(df, gm_sub, 'datetime',
                             ['nuclear_share', 'thermal_share', 'hydro_share',
                              'solar_share', 'wind_share', 'res_share', 'total_gen_mw'])
        gm_fill = ['nuclear_share', 'thermal_share', 'hydro_share',
                   'solar_share', 'wind_share', 'res_share', 'total_gen_mw']
        has_gm = [c for c in gm_fill if c in df.columns]
        if has_gm and df[has_gm].isna().any().any():
            all_dates = sorted(gm['date'].unique(), reverse=True)
            proxy = {}
            for date in all_dates:
                day_df = gm[gm['date'] == date]
                for _, row in day_df.iterrows():
                    h = int(row['hour'])
                    if h not in proxy:
                        proxy[h] = {c: row[c] for c in has_gm}
                if len(proxy) >= 24:
                    break
            for idx in df[df[has_gm[0]].isna()].index:
                h = int(df.loc[idx, 'hour'])
                if h in proxy:
                    for c in has_gm:
                        df.loc[idx, c] = proxy[h][c]
    for c in ['nuclear_share', 'thermal_share', 'hydro_share',
              'solar_share', 'wind_share', 'res_share', 'total_gen_mw']:
        if c not in df.columns:
            df[c] = 0
        df[c] = df[c].fillna(0)

    df['solar_irradiance'] = df['solar_share'] * df.get('solar_radiation', 0)
    df['solar_intensity'] = df['solar_share'] * df['sin_hour'].clip(lower=0)
    df['is_solar_dip_hour'] = ((df['hour'] >= 10) & (df['hour'] <= 15)).astype(int)

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
        else:
            df['price_lag_24h'] = 0
            df['price_lag_168h'] = 0
    except Exception:
        df['price_lag_24h'] = 0
        df['price_lag_168h'] = 0

    return df

def _generate_synthetic_day(target_date, use_existing_as_ref=False):
    from datetime import datetime as dt_mod
    try:
        target_dt = dt_mod.strptime(target_date, '%Y-%m-%d')
    except ValueError:
        target_dt = datetime.now()
    month = target_dt.month
    rng = np.random.RandomState(hash(target_date) % (2**31))
    rows = []
    for hour in range(24):
        base_temp = 5 + 15 * np.sin(np.pi * (month - 3) / 6)
        daily_variation = 7 * np.sin(np.pi * (hour - 6) / 12)
        noise = rng.normal(0, 2)
        temp = round(base_temp + daily_variation + noise, 1)
        clouds = int(np.clip(rng.normal(50, 25), 0, 100))
        wind = round(np.random.exponential(3), 1)
        humidity = int(np.clip(rng.normal(65 - 10 * np.sin(np.pi * (hour - 6) / 12), 15), 30, 95))
        solar_radiation = 0
        if 5 <= hour <= 21:
            elevation = np.sin(np.radians(50)) * np.sin(np.radians(23.45 * np.sin(np.radians(360/365 * (month * 30.5 - 81))))) + \
                np.cos(np.radians(50)) * np.cos(np.radians(23.45 * np.sin(np.radians(360/365 * (month * 30.5 - 81))))) * \
                np.cos(np.radians(15 * (hour - 12)))
            if elevation > 0:
                solar_radiation = round(max(0, 1000 * elevation * (1 - (clouds / 100) * 0.75)), 1)
        rows.append({
            'date': target_date,
            'hour': hour,
            'temperature': temp,
            'humidity': humidity,
            'clouds': clouds,
            'wind_speed': wind,
            'solar_radiation': solar_radiation,
            'weather_id': 800 if clouds < 30 else (802 if clouds < 70 else 804),
            'datetime': pd.Timestamp(f'{target_date} {hour:02d}:00:00')
        })
    return pd.DataFrame(rows)

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

    weather = get_forecast()
    if weather is None or len(weather) == 0:
        return None

    day_weather = weather[weather['date'] == target_date].copy()
    if len(day_weather) < 24:
        synthetic = _generate_synthetic_day(target_date, len(day_weather) > 0)
        if len(day_weather) > 0:
            existing_hours = set(day_weather['hour'])
            extra = synthetic[~synthetic['hour'].isin(existing_hours)]
            day_weather = pd.concat([day_weather, extra])
        else:
            day_weather = synthetic
        day_weather = day_weather.sort_values('hour').reset_index(drop=True)

    features = prepare_prediction_features(day_weather)
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
                'temperature',
                'humidity',
                'solar_radiation',
                'solar_index', 'wind_index', 'renewable_index',
                'nuclear_share', 'thermal_share', 'hydro_share',
                'solar_share', 'wind_share', 'res_share', 'total_gen_mw',
                'days_since_epoch',
                'solar_irradiance', 'solar_intensity',
                'is_solar_dip_hour',
                'price_lag_24h', 'price_lag_168h',
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
    for idx, (_, row) in enumerate(day_weather.iterrows()):
        hour = int(row['hour'])
        price = float(predictions[idx]) if predictions is not None and idx < len(predictions) else 0
        results.append({
            'hour': f"{(hour % 24) + 1:02d}:00",
            'hour_num': hour,
            'price': max(price, 0.01),
            'temperature': round(float(row.get('temperature', 15) if pd.notna(row.get('temperature', 15)) else 15), 1),
            'humidity': int(row.get('humidity', 50) if pd.notna(row.get('humidity', 50)) else 50),
            'clouds': int(row.get('clouds', 50) if pd.notna(row.get('clouds', 50)) else 50),
            'wind_speed': round(float(row.get('wind_speed', 0)), 1),
            'solar_radiation': round(float(row.get('solar_radiation', 0)), 1)
        })
    results.sort(key=lambda x: x['hour_num'])
    return results

def predict_with_dates(dates):
    model = load_model()
    if model is None:
        return None
    weather = get_forecast()
    if weather is None:
        return None

    all_predictions = {}
    for target_date in dates:
        day_weather = weather[weather['date'] == target_date].copy()
        if len(day_weather) == 0:
            all_predictions[target_date] = []
            continue
        features = prepare_prediction_features(day_weather)
        if features is None:
            all_predictions[target_date] = []
            continue
        preds = predict_hourly(model, features)
        results = []
        for idx, (_, row) in enumerate(day_weather.iterrows()):
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
