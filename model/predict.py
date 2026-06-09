import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from model.train import load_model, predict_hourly, load_metrics
from collectors.weather import get_forecast
from collectors.renewable_index import get_renewable_indices
from collectors.generation_mix import get_generation_mix
from data.loader import build_features, HOLIDAYS_2025

def prepare_prediction_features(weather_forecast):
    df = pd.DataFrame(weather_forecast)
    if df.empty:
        return None
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['hour'].astype(str) + ':00:00')
    df = build_features(df)
    if 'temperature' not in df.columns:
        df['temperature'] = 15
    df['solar_radiation'] = df.get('solar_radiation', 0)
    df['clouds'] = df.get('clouds', 50)
    df['wind_speed'] = df.get('wind_speed', 0)

    renewable_idx = get_renewable_indices()
    if renewable_idx is not None and len(renewable_idx) > 0:
        renewable_idx = renewable_idx.copy()
        renewable_idx['date'] = pd.to_datetime(renewable_idx['date']).dt.strftime('%Y-%m-%d')
        renewable_idx['datetime'] = pd.to_datetime(
            renewable_idx['date'] + ' ' + renewable_idx['hour'].astype(str) + ':00:00')
        merge_cols = ['datetime', 'solar_index', 'wind_index', 'renewable_index']
        renewable_idx = renewable_idx[[c for c in merge_cols if c in renewable_idx.columns]]
        df = pd.merge(df, renewable_idx, on='datetime', how='left')
    for c in ['solar_index', 'wind_index', 'renewable_index']:
        if c not in df.columns:
            df[c] = 0
        df[c] = df[c].fillna(0)

    genmix = get_generation_mix(days=14)
    if genmix is not None and len(genmix) > 0:
        genmix = genmix.copy()
        genmix['date'] = pd.to_datetime(genmix['date']).dt.strftime('%Y-%m-%d')
        genmix['datetime'] = pd.to_datetime(
            genmix['date'] + ' ' + genmix['hour'].astype(int).astype(str) + ':00:00')
        gm_cols = ['datetime', 'nuclear_share', 'thermal_share', 'hydro_share',
                    'solar_share', 'wind_share', 'res_share', 'total_gen_mw']
        genmix = genmix[[c for c in gm_cols if c in genmix.columns]]
        df = pd.merge(df, genmix, on='datetime', how='left')
    for c in ['nuclear_share', 'thermal_share', 'hydro_share',
              'solar_share', 'wind_share', 'res_share', 'total_gen_mw']:
        if c not in df.columns:
            df[c] = 0
        df[c] = df[c].fillna(0)

    return df

def predict_next_day_prices(target_date=None):
    if target_date is None:
        target_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')

    model = load_model()
    if model is None:
        return None

    weather = get_forecast()
    if weather is None or len(weather) == 0:
        return None

    day_weather = weather[weather['date'] == target_date].copy()
    if len(day_weather) < 24:
            remaining = 24 - len(day_weather)
            extra_rows = []
            for i in range(remaining):
                extra_rows.append({
                    'date': target_date,
                    'hour': len(day_weather) + i,
                    'temperature': day_weather['temperature'].mean() if len(day_weather) > 0 else 15,
                    'clouds': 50,
                    'wind_speed': 3,
                    'solar_radiation': 0
                })
            extra_df = pd.DataFrame(extra_rows)
            day_weather = pd.concat([day_weather, extra_df])

    features = prepare_prediction_features(day_weather)
    if features is None:
        return None

    predictions = predict_hourly(model, features)

    results = []
    for idx, (_, row) in enumerate(day_weather.iterrows()):
        hour = int(row['hour'])
        price = float(predictions[idx]) if predictions is not None and idx < len(predictions) else 0
        results.append({
            'hour': f"{hour:02d}:00",
            'hour_num': hour,
            'price': round(max(price, 0.01), 2),
            'temperature': round(float(row.get('temperature', 15)), 1),
            'clouds': int(row.get('clouds', 50)),
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
                'hour': f"{int(row['hour']):02d}:00",
                'price': round(max(price, 0.01), 2),
                'temperature': round(float(row.get('temperature', 15)), 1)
            })
        all_predictions[target_date] = sorted(results, key=lambda x: x['hour'])
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
