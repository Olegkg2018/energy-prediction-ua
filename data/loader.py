import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

DATA_DIR = os.path.join(os.path.dirname(__file__))
PRICES_FILE = os.path.join(DATA_DIR, 'prices_2025.csv')
WEATHER_FILE = os.path.join(DATA_DIR, 'weather_2025.csv')
CACHE_FILE = os.path.join(DATA_DIR, 'cache.feather')
RENEWABLE_CACHE = os.path.join(DATA_DIR, 'renewable_2025.feather')
GENMIX_CACHE = os.path.join(DATA_DIR, 'genmix_2025.feather')

_cache = None

HOLIDAYS_2025 = [
    '2025-01-01', '2025-01-07', '2025-03-08', '2025-04-20',
    '2025-05-01', '2025-05-05', '2025-05-06', '2025-06-08',
    '2025-06-28', '2025-08-24', '2025-10-14', '2025-12-25'
]

SOLAR_ZONE_SOUTH_COEF = 1.3
WIND_ZONE_SOUTH_COEF = 1.1

def generate_synthetic_renewable_indices(year=2025):
    cache_file = os.path.join(DATA_DIR, f'renewable_{year}.feather')
    if os.path.exists(cache_file):
        try:
            return pd.read_feather(cache_file)
        except Exception:
            pass
    np.random.seed(42 + year - 2020)
    rows = []
    start = datetime(year, 1, 1)
    end = datetime(year, 12, 31, 23)
    ts = start
    while ts <= end:
        hour = ts.hour
        month = ts.month
        cloud_base = 50 + 20 * np.sin(np.pi * (month - 3) / 6)
        clouds = int(np.clip(np.random.normal(cloud_base, 25), 0, 100))
        wind_base = 4 + 2 * np.sin(np.pi * (month - 1) / 6)
        wind_speed = round(abs(np.random.normal(wind_base, 2)), 1)
        declination = 23.45 * np.sin(np.radians(360 / 365 * (month * 30.5 - 81)))
        hour_angle = 15 * (hour - 12)
        lat_rad = np.radians(47)
        elevation = np.sin(lat_rad) * np.sin(np.radians(declination)) + \
                    np.cos(lat_rad) * np.cos(np.radians(declination)) * np.cos(np.radians(hour_angle))
        solar = 0
        if elevation > 0 and 5 <= hour <= 21:
            clear_sky = 1000 * elevation
            cloud_factor = 1 - (clouds / 100) * 0.75
            solar = max(0, clear_sky * cloud_factor * SOLAR_ZONE_SOUTH_COEF)
        wind_gen = 0
        if wind_speed >= 3:
            if wind_speed > 25:
                wind_gen = 1.0
            elif wind_speed > 12:
                wind_gen = 1.0
            else:
                wind_gen = ((wind_speed - 3) / 9) ** 3
            wind_gen *= WIND_ZONE_SOUTH_COEF
        rows.append({
            'datetime': ts,
            'date': ts.strftime('%Y-%m-%d'),
            'hour': hour,
            'solar_index': round(solar, 1),
            'wind_index': round(wind_gen, 4),
            'renewable_index': round(solar + wind_gen * 500, 1),
        })
        ts += timedelta(hours=1)
    df = pd.DataFrame(rows)
    try:
        df.to_feather(cache_file)
    except Exception:
        pass
    return df

def generate_synthetic_generation_mix(year=2025):
    cache_file = os.path.join(DATA_DIR, f'genmix_{year}.feather')
    if os.path.exists(cache_file):
        try:
            return pd.read_feather(cache_file)
        except Exception:
            pass
    np.random.seed(99 + year - 2020)
    rows = []
    start = datetime(year, 1, 1)
    end = datetime(year, 12, 31, 23)
    ts = start
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
        res_share = (solar + wind + other_res) / total if total > 0 else 0
        rows.append({
            'datetime': ts,
            'date': ts.strftime('%Y-%m-%d'),
            'hour': hour,
            'nuclear_share': round(nuclear / total, 4) if total > 0 else 0,
            'thermal_share': round(thermal / total, 4) if total > 0 else 0,
            'hydro_share': round(hydro / total, 4) if total > 0 else 0,
            'solar_share': round(solar / total, 4) if total > 0 else 0,
            'wind_share': round(wind / total, 4) if total > 0 else 0,
            'res_share': round(res_share, 4),
            'total_gen_mw': round(total, 1),
        })
        ts += timedelta(hours=1)
    df = pd.DataFrame(rows)
    try:
        df.to_feather(cache_file)
    except Exception:
        pass
    return df

def load_price_file():
    df = pd.read_csv(PRICES_FILE)
    df.columns = df.columns.str.strip()
    if 'date' not in df.columns:
        df.rename(columns={df.columns[0]: 'date'}, inplace=True)
    id_vars = [c for c in df.columns if c != 'date']
    melted = df.melt(id_vars=['date'], var_name='hour', value_name='price')
    melted['hour'] = melted['hour'].astype(str).str.strip().str.zfill(2).str.replace('h', '', regex=False)
    melted['hour'] = melted['hour'].astype(int)
    melted['hour'] = melted['hour'] % 24
    melted['datetime'] = pd.to_datetime(melted['date'] + ' ' + melted['hour'].astype(str) + ':00:00', dayfirst=True, format='mixed')
    melted.dropna(subset=['price'], inplace=True)
    melted['price'] = pd.to_numeric(melted['price'], errors='coerce')
    melted.dropna(subset=['price'], inplace=True)
    return melted[['datetime', 'date', 'hour', 'price']]

def load_weather_file():
    df = pd.read_csv(WEATHER_FILE)
    df.columns = df.columns.str.strip()
    id_vars = [c for c in df.columns if c not in ('date',) and c not in ('hour', 'temp', 'temperature')]
    if 'temp' in df.columns:
        temp_col = 'temp'
    elif 'temperature' in df.columns:
        temp_col = 'temperature'
    else:
        temp_col = None
    if 'hour' not in df.columns and len(df.columns) >= 3:
        possible = [c for c in df.columns if c != 'date' and c != temp_col]
        if possible:
            df.rename(columns={possible[0]: 'hour'}, inplace=True)
    df['hour'] = df['hour'].astype(str).str.strip().str.zfill(2)
    df['hour'] = df['hour'].str.replace('h', '', regex=False).astype(int)
    df['hour'] = df['hour'] % 24
    temp = temp_col or [c for c in df.columns if c not in ('date', 'hour')][0]
    df.rename(columns={temp: 'temperature'}, inplace=True)
    df['temperature'] = pd.to_numeric(df['temperature'], errors='coerce')
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['hour'].astype(str) + ':00:00', dayfirst=True, format='mixed')
    df.dropna(subset=['temperature'], inplace=True)
    return df[['datetime', 'date', 'hour', 'temperature']]

def build_features(df):
    df = df.copy()
    dt = pd.to_datetime(df['datetime'])
    df['hour'] = dt.dt.hour
    df['dayofweek'] = dt.dt.dayofweek
    df['month'] = dt.dt.month
    df['day'] = dt.dt.day
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    df['is_holiday'] = dt.dt.strftime('%Y-%m-%d').isin(HOLIDAYS_2025).astype(int)
    df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
    df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)

    for col in ['solar_index', 'wind_index', 'renewable_index',
                'nuclear_share', 'thermal_share', 'hydro_share',
                'solar_share', 'wind_share', 'res_share',
                'total_gen_mw']:
        if col not in df.columns:
            df[col] = 0
    return df

def get_combined_dataset():
    global _cache
    if _cache is not None:
        return _cache
    if os.path.exists(CACHE_FILE):
        try:
            _cache = pd.read_feather(CACHE_FILE)
            return _cache
        except Exception:
            pass
    prices = load_price_file()
    weather = load_weather_file()
    merged = pd.merge(prices, weather, on=['datetime', 'date', 'hour'], how='left')
    merged['temperature'] = merged['temperature'].interpolate(limit_direction='both')

    renewable = generate_synthetic_renewable_indices(2025)
    merged = pd.merge(merged, renewable[['datetime', 'solar_index', 'wind_index', 'renewable_index']],
                      on='datetime', how='left')
    for c in ['solar_index', 'wind_index', 'renewable_index']:
        merged[c] = merged[c].fillna(0)

    genmix = generate_synthetic_generation_mix(2025)
    merged = pd.merge(merged, genmix[['datetime', 'nuclear_share', 'thermal_share', 'hydro_share',
                                       'solar_share', 'wind_share', 'res_share', 'total_gen_mw']],
                      on='datetime', how='left')
    for c in ['nuclear_share', 'thermal_share', 'hydro_share',
              'solar_share', 'wind_share', 'res_share', 'total_gen_mw']:
        merged[c] = merged[c].fillna(0)

    merged = build_features(merged)
    merged.sort_values('datetime', inplace=True)
    merged.reset_index(drop=True, inplace=True)
    try:
        merged.to_feather(CACHE_FILE)
    except Exception:
        pass
    _cache = merged
    return merged

def get_data_statistics():
    df = get_combined_dataset()
    stats = {
        'total_rows': len(df),
        'date_range': f"{df['date'].min()} - {df['date'].max()}",
        'price_min': round(float(df['price'].min()), 2),
        'price_max': round(float(df['price'].max()), 2),
        'price_mean': round(float(df['price'].mean()), 2),
        'price_std': round(float(df['price'].std()), 2),
        'temp_min': round(float(df['temperature'].min()), 1),
        'temp_max': round(float(df['temperature'].max()), 1),
    }
    surplus_hours = df[df['price'] < 100]
    stats['surplus_hours_count'] = len(surplus_hours)
    stats['surplus_min_price'] = round(float(surplus_hours['price'].min()), 2) if len(surplus_hours) > 0 else 0
    return stats

def clear_cache():
    global _cache
    _cache = None
    if os.path.exists(CACHE_FILE):
        os.remove(CACHE_FILE)
