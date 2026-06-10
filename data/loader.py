import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

DATA_DIR = os.path.join(os.path.dirname(__file__))
PRICES_FILE = os.path.join(DATA_DIR, 'prices_2025.csv')
WEATHER_FILE = os.path.join(DATA_DIR, 'weather_2025.csv')
CACHE_FILE = os.path.join(DATA_DIR, 'cache.feather')
OREE_CACHE = os.path.join(os.path.dirname(DATA_DIR), 'data', 'oree_prices.feather')

_cache = None

HOLIDAYS_2025 = [
    '2025-01-01', '2025-01-07', '2025-03-08', '2025-04-20',
    '2025-05-01', '2025-05-05', '2025-05-06', '2025-06-08',
    '2025-06-28', '2025-08-24', '2025-10-14', '2025-12-25'
]

def load_oree_prices():
    if not os.path.exists(OREE_CACHE):
        return None
    try:
        df = pd.read_feather(OREE_CACHE)
        df['hour'] = pd.to_numeric(df['hour'], errors='coerce').astype(int)
        df['hour'] = df['hour'] % 24
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['date'] = pd.to_datetime(df['datetime']).dt.strftime('%Y-%m-%d')
        return df[['datetime', 'date', 'hour', 'price']]
    except Exception:
        return None

def generate_synthetic_weather_for_range(start_date, end_date):
    np.random.seed(42)
    rows = []
    ts = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    while ts <= end:
        hour = ts.hour
        month = ts.month
        day_of_year = ts.dayofyear
        base_temp = 8 + 15 * np.sin(np.pi * (month - 3) / 6)
        daily_variation = 6 * np.sin(np.pi * (hour - 6) / 12)
        temp = round(base_temp + daily_variation + np.random.normal(0, 2), 1)
        month = ts.month
        clouds = int(np.clip(np.random.normal(50 + 20 * np.sin(np.pi * (month - 3) / 6), 25), 0, 100))
        declination = 23.45 * np.sin(np.radians(360 / 365 * (ts.dayofyear - 81)))
        hour_angle = 15 * (hour - 12)
        lat_rad = np.radians(47)
        elevation = np.sin(lat_rad) * np.sin(np.radians(declination)) + \
                    np.cos(lat_rad) * np.cos(np.radians(declination)) * np.cos(np.radians(hour_angle))
        solar = 0
        if elevation > 0 and 5 <= hour <= 21:
            clear_sky = 1000 * elevation
            cloud_factor = 1 - (clouds / 100) * 0.75
            solar = max(0, clear_sky * cloud_factor)
        rows.append({
            'datetime': ts,
            'date': ts.strftime('%Y-%m-%d'),
            'hour': hour,
            'temperature': temp,
            'clouds': clouds,
            'solar_radiation': round(solar, 1),
        })
        ts += timedelta(hours=1)
    return pd.DataFrame(rows)

def generate_synthetic_renewable_for_range(start_year, end_year):
    np.random.seed(42)
    rows = []
    for year in range(start_year, end_year + 1):
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
                solar = max(0, clear_sky * cloud_factor * 1.3)
            wind_gen = 0
            if wind_speed >= 3:
                if wind_speed > 25:
                    wind_gen = 1.0
                elif wind_speed > 12:
                    wind_gen = 1.0
                else:
                    wind_gen = ((wind_speed - 3) / 9) ** 3
                wind_gen *= 1.1
            rows.append({
                'datetime': ts,
                'date': ts.strftime('%Y-%m-%d'),
                'hour': hour,
                'solar_index': round(solar, 1),
                'wind_index': round(wind_gen, 4),
                'renewable_index': round(solar + wind_gen * 500, 1),
            })
            ts += timedelta(hours=1)
    return pd.DataFrame(rows)

def generate_synthetic_genmix_for_range(start_year, end_year):
    np.random.seed(99)
    rows = []
    for year in range(start_year, end_year + 1):
        start = datetime(year, 1, 1)
        end = datetime(year, 12, 31, 23)
        ts = start
        year_solar_mult = 1.0 + max(0, year - 2021) * 0.5
        nuclear_base = 4500 if year >= 2024 else 7500
        while ts <= end:
            hour = ts.hour
            month = ts.month
            is_winter = month in [12, 1, 2]
            is_summer = month in [6, 7, 8]
            nuclear = np.random.normal(nuclear_base, 300)
            thermal = np.random.normal(2000 if is_winter else 1200, 400)
            if 8 <= hour <= 11 or 17 <= hour <= 21:
                thermal += 500
            hydro = np.random.normal(600, 150)
            solar = 0
            if 6 <= hour <= 19:
                hour_factor = np.sin(np.pi * (hour - 6) / 13)
                solar = np.random.normal(2500 * year_solar_mult if is_summer else 1500 * year_solar_mult, 500) * max(0, hour_factor)
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
    return pd.DataFrame(rows)

DATA_EPOCH = pd.Timestamp('2023-01-01')

def _days_since_epoch(dt_series):
    return (dt_series - DATA_EPOCH).dt.days.astype(int)

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
    df['days_since_epoch'] = _days_since_epoch(dt)
    if 'solar_radiation' not in df.columns:
        df['solar_radiation'] = 0
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

    prices = load_oree_prices()
    if prices is None or len(prices) < 1000:
        from collectors.oree import get_sample_prices
        prices = get_sample_prices(365)

    price_min_date = prices['date'].min()
    price_max_date = prices['date'].max()
    years = set(pd.to_datetime(prices['datetime']).dt.year)
    min_year = min(years)
    max_year = max(years)

    weather = generate_synthetic_weather_for_range(price_min_date, price_max_date)
    merged = pd.merge(prices, weather, on=['datetime', 'date', 'hour'], how='left')
    merged['temperature'] = merged['temperature'].interpolate(limit_direction='both')

    oree_years = set(pd.to_datetime(prices['datetime']).dt.year)
    renewable = generate_synthetic_renewable_for_range(min(oree_years), max(oree_years))
    merged = pd.merge(merged, renewable[['datetime', 'solar_index', 'wind_index', 'renewable_index']],
                      on='datetime', how='left')
    for c in ['solar_index', 'wind_index', 'renewable_index']:
        merged[c] = merged[c].fillna(0)

    try:
        from collectors.ukrenergo import get_ukrenergo_genmix
        real_genmix = get_ukrenergo_genmix()
        if real_genmix is not None and len(real_genmix) > 0:
            genmix = real_genmix[real_genmix['datetime'].isin(prices['datetime'])]
            if len(genmix) == 0:
                genmix = generate_synthetic_genmix_for_range(min(oree_years), max(oree_years))
        else:
            genmix = generate_synthetic_genmix_for_range(min(oree_years), max(oree_years))
    except Exception:
        genmix = generate_synthetic_genmix_for_range(min(oree_years), max(oree_years))
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
        os.makedirs(DATA_DIR, exist_ok=True)
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

def get_oree_prices():
    df = load_oree_prices()
    if df is not None:
        return df[['date', 'hour', 'price']].to_dict('records')
    return []
