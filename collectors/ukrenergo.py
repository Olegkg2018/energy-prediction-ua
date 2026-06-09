import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
CACHE_FILE = os.path.join(CACHE_DIR, 'ukrenergo_genmix.feather')

HISTORICAL_URL = 'https://raw.githubusercontent.com/Aranaur/aranaur.rbind.io/main/datasets/energy_ua/energy_ua_2014_2021.csv'

COLUMN_MAP = {
    'AES': 'nuclear_mw',
    'TEC': 'thermal_mw',
    'TES': 'thermal_chp_mw',
    'GES': 'hydro_mw',
    'GAES_GEN': 'hydro_pumped_gen_mw',
    'GAES_PUMP': 'hydro_pumped_load_mw',
    'VDE': 'res_mw',
    'CONSUMPTION': 'consumption_mw',
}

def fetch_historical_csv():
    try:
        df = pd.read_csv(HISTORICAL_URL)
        df.rename(columns={'time': 'datetime'}, inplace=True)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['date'] = df['datetime'].dt.strftime('%Y-%m-%d')
        df['hour'] = df['datetime'].dt.hour
        for col in ['AES', 'TEC', 'TES', 'GES', 'GAES_GEN', 'GAES_PUMP', 'VDE', 'CONSUMPTION']:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        df['thermal_mw'] = df['TEC'] + df['TES']
        df['nuclear_mw'] = df['AES']
        df['hydro_mw'] = df['GES'] + df['GAES_GEN'] - df['GAES_PUMP'].clip(lower=0)
        df['hydro_mw'] = df['hydro_mw'].clip(lower=0)
        df['res_mw'] = df['VDE']
        df['solar_mw'] = df['VDE'] * 0.55
        df['wind_mw'] = df['VDE'] * 0.35
        df['other_res_mw'] = df['VDE'] * 0.10
        df['total_gen_mw'] = df['AES'] + df['thermal_mw'] + df['hydro_mw'] + df['VDE']
        df['import_balance_mw'] = df.get('UK_EURO', 0) + df.get('UK_BLR_RUS', 0) + df.get('UK_MLD', 0)
        result = df[['datetime', 'date', 'hour', 'nuclear_mw', 'thermal_mw', 'hydro_mw',
                      'solar_mw', 'wind_mw', 'other_res_mw', 'total_gen_mw', 'import_balance_mw']]
        result['source'] = 'ukrenergo_historical'
        return result
    except Exception as e:
        print(f"[UKRENERGO] Historical fetch error: {e}")
        return None

def generate_synthetic_for_gap(start_date, end_date):
    np.random.seed(99)
    rows = []
    ts = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    while ts <= end:
        hour = ts.hour
        month = ts.month
        is_winter = month in [12, 1, 2]
        is_summer = month in [6, 7, 8]
        nuclear = np.random.normal(7500, 300)
        thermal = np.random.normal(3000 if is_winter else 2000, 400)
        if 8 <= hour <= 11 or 17 <= hour <= 21:
            thermal += 500
        hydro = np.random.normal(800, 150)
        solar = 0
        if 6 <= hour <= 19:
            hour_factor = np.sin(np.pi * (hour - 6) / 13)
            solar = np.random.normal(2500 if is_summer else 1500, 500) * max(0, hour_factor)
            solar += np.random.normal(0, 200)
        wind = abs(np.random.normal(800, 300)) * (1.3 if hour >= 22 or hour <= 4 else 1.0)
        other_res = abs(np.random.normal(200, 50))
        total = nuclear + thermal + hydro + solar + wind + other_res
        rows.append({
            'datetime': ts,
            'date': ts.strftime('%Y-%m-%d'),
            'hour': hour,
            'nuclear_mw': round(nuclear, 1),
            'thermal_mw': round(thermal, 1),
            'hydro_mw': round(hydro, 1),
            'solar_mw': round(solar, 1),
            'wind_mw': round(wind, 1),
            'other_res_mw': round(other_res, 1),
            'total_gen_mw': round(total, 1),
            'import_balance_mw': round(np.random.normal(-500, 300), 1),
            'source': 'synthetic',
        })
        ts += timedelta(hours=1)
    return pd.DataFrame(rows)

def to_aggregated(df):
    agg = df.copy()
    agg['res_share'] = np.where(agg['total_gen_mw'] > 0,
                                 (agg['solar_mw'] + agg['wind_mw'] + agg['other_res_mw']) / agg['total_gen_mw'], 0)
    agg['nuclear_share'] = np.where(agg['total_gen_mw'] > 0,
                                     agg['nuclear_mw'] / agg['total_gen_mw'], 0)
    agg['thermal_share'] = np.where(agg['total_gen_mw'] > 0,
                                     agg['thermal_mw'] / agg['total_gen_mw'], 0)
    agg['hydro_share'] = np.where(agg['total_gen_mw'] > 0,
                                   agg['hydro_mw'] / agg['total_gen_mw'], 0)
    agg['solar_share'] = np.where(agg['total_gen_mw'] > 0,
                                   agg['solar_mw'] / agg['total_gen_mw'], 0)
    agg['wind_share'] = np.where(agg['total_gen_mw'] > 0,
                                  agg['wind_mw'] / agg['total_gen_mw'], 0)
    return agg

def load_cache():
    if not os.path.exists(CACHE_FILE):
        return None
    try:
        df = pd.read_feather(CACHE_FILE)
        latest = pd.to_datetime(df['datetime']).max()
        if datetime.now() - latest < timedelta(days=7):
            return df
    except Exception:
        pass
    return None

def save_cache(df):
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        df.to_feather(CACHE_FILE)
    except Exception:
        pass

def get_ukrenergo_genmix():
    cached = load_cache()
    if cached is not None:
        return to_aggregated(cached)

    df_hist = fetch_historical_csv()
    if df_hist is not None and len(df_hist) > 0:
        last_hist = pd.to_datetime(df_hist['datetime']).max()
        synthetic = generate_synthetic_for_gap(last_hist + timedelta(hours=1), datetime.now())
        combined = pd.concat([df_hist, synthetic], ignore_index=True)
    else:
        combined = generate_synthetic_for_gap(datetime(2022, 1, 1), datetime.now())

    combined.sort_values('datetime', inplace=True)
    combined.reset_index(drop=True, inplace=True)
    save_cache(combined)
    return to_aggregated(combined)

def get_ukrenergo_stats():
    df = get_ukrenergo_genmix()
    if df is None or len(df) == 0:
        return {}
    latest = df.sort_values('datetime').tail(1).iloc[0]
    recent_24h = df.tail(24)
    source = 'ukrenergo_historical' if 'ukrenergo' in str(df['source'].iloc[0]) else 'synthetic'
    has_real_data = (pd.to_datetime(df['datetime']).max().year >= 2021)
    return {
        'updated': datetime.now().isoformat(),
        'latest': {
            'date': latest['date'],
            'hour': int(latest['hour']),
            'nuclear_mw': round(float(latest['nuclear_mw']), 0),
            'thermal_mw': round(float(latest['thermal_mw']), 0),
            'hydro_mw': round(float(latest['hydro_mw']), 0),
            'solar_mw': round(float(latest['solar_mw']), 0),
            'wind_mw': round(float(latest['wind_mw']), 0),
            'total_gen_mw': round(float(latest['total_gen_mw']), 0),
        },
        'averages': {
            'nuclear_mw': round(float(recent_24h['nuclear_mw'].mean()), 0),
            'thermal_mw': round(float(recent_24h['thermal_mw'].mean()), 0),
            'hydro_mw': round(float(recent_24h['hydro_mw'].mean()), 0),
            'solar_mw': round(float(recent_24h['solar_mw'].mean()), 0),
            'wind_mw': round(float(recent_24h['wind_mw'].mean()), 0),
            'total_gen_mw': round(float(recent_24h['total_gen_mw'].mean()), 0),
        },
        'shares': {
            'res_share': round(float(recent_24h['res_share'].mean() * 100), 1),
            'nuclear_share': round(float(recent_24h['nuclear_share'].mean() * 100), 1),
            'thermal_share': round(float(recent_24h['thermal_share'].mean() * 100), 1),
            'hydro_share': round(float(recent_24h['hydro_share'].mean() * 100), 1),
            'solar_share': round(float(recent_24h['solar_share'].mean() * 100), 1),
            'wind_share': round(float(recent_24h['wind_share'].mean() * 100), 1),
        },
        'source': 'ukrenergo (2014-2021 real + 2022+ synthetic)' if has_real_data else 'synthetic',
        'date_range': f"{df['date'].min()} - {df['date'].max()}",
    }

def get_ukrenergo_timeseries(days=7):
    df = get_ukrenergo_genmix()
    if df is None or len(df) == 0:
        return []
    return df.tail(24 * days).to_dict('records')
