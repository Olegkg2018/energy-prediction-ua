import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json

RENEWABLE_ZONES = [
    {'name': 'pivden', 'lat': 47.0, 'lon': 32.0, 'label': 'Південь (Миколаїв)', 'solar': 1.3, 'wind': 1.2},
    {'name': 'zaporizhzhia', 'lat': 47.8, 'lon': 35.0, 'label': 'Запоріжжя', 'solar': 1.2, 'wind': 0.8},
    {'name': 'dnipro', 'lat': 48.5, 'lon': 35.0, 'label': 'Дніпро', 'solar': 1.1, 'wind': 0.9},
    {'name': 'odesa', 'lat': 46.5, 'lon': 30.5, 'label': 'Одеса', 'solar': 1.3, 'wind': 1.0},
    {'name': 'carpathians', 'lat': 49.0, 'lon': 24.0, 'label': 'Карпати', 'solar': 0.7, 'wind': 1.4},
]

API_KEY_ENV = 'OPENWEATHER_API_KEY'
DEFAULT_API_KEY = 'demo_key'
CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
CACHE_FILE = os.path.join(CACHE_DIR, 'renewable_index.json')

def get_api_key():
    return os.environ.get(API_KEY_ENV, DEFAULT_API_KEY)

def fetch_weather_for_zone(lat, lon, api_key):
    url = 'https://api.openweathermap.org/data/2.5/forecast'
    params = {'lat': lat, 'lon': lon, 'units': 'metric', 'appid': api_key}
    try:
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None

def estimate_solar_generation(dt, clouds, zone_factor):
    hour = dt.hour
    month = dt.month
    if hour < 5 or hour > 21:
        return 0
    declination = 23.45 * np.sin(np.radians(360 / 365 * (month * 30.5 - 81)))
    hour_angle = 15 * (hour - 12)
    lat_rad = np.radians(47)
    elevation = np.sin(lat_rad) * np.sin(np.radians(declination)) + \
                np.cos(lat_rad) * np.cos(np.radians(declination)) * np.cos(np.radians(hour_angle))
    if elevation < 0:
        return 0
    clear_sky = 1000 * elevation
    cloud_factor = 1 - (clouds / 100) * 0.75
    return max(0, clear_sky * cloud_factor * zone_factor)

def estimate_wind_generation(wind_speed, zone_factor):
    v = wind_speed
    if v < 3:
        return 0
    if v > 25:
        return 1.0 * zone_factor
    if v > 12:
        return 1.0 * zone_factor
    return ((v - 3) / 9) ** 3 * zone_factor

def parse_forecast_to_rows(data, zone):
    rows = []
    for item in data.get('list', []):
        dt = datetime.fromtimestamp(item['dt'])
        clouds = item.get('clouds', {}).get('all', 50)
        wind = item.get('wind', {}).get('speed', 0)
        temp = item['main'].get('temp', 15)
        solar = estimate_solar_generation(dt, clouds, zone['solar'])
        wind_gen = estimate_wind_generation(wind, zone['wind'])
        rows.append({
            'datetime': dt.isoformat(),
            'date': dt.strftime('%Y-%m-%d'),
            'hour': dt.hour,
            'zone': zone['name'],
            'zone_label': zone['label'],
            'temperature': round(temp, 1),
            'clouds': clouds,
            'wind_speed': round(wind, 1),
            'solar_gen_index': round(solar, 1),
            'wind_gen_index': round(wind_gen, 4),
        })
    return rows

def generate_synthetic_rows(zone, days=2):
    now = datetime.now().replace(minute=0, second=0, microsecond=0)
    rows = []
    for i in range(48):
        dt = now + timedelta(hours=i)
        hour = dt.hour
        month = dt.month
        base_temp = 5 + 15 * np.sin(np.pi * (month - 3) / 6)
        daily_variation = 7 * np.sin(np.pi * (hour - 6) / 12)
        temp = round(base_temp + daily_variation + np.random.normal(0, 2) + (zone['lat'] - 47), 1)
        clouds = int(np.clip(np.random.normal(50, 25), 0, 100))
        wind = round(np.random.exponential(3) * zone['wind'], 1)
        solar = estimate_solar_generation(dt, clouds, zone['solar'])
        wind_gen = estimate_wind_generation(wind, zone['wind'])
        rows.append({
            'datetime': dt.isoformat(),
            'date': dt.strftime('%Y-%m-%d'),
            'hour': hour,
            'zone': zone['name'],
            'zone_label': zone['label'],
            'temperature': round(temp, 1),
            'clouds': clouds,
            'wind_speed': round(wind, 1),
            'solar_gen_index': round(solar, 1),
            'wind_gen_index': round(wind_gen, 4),
        })
    return rows

def fetch_all_zones():
    api_key = get_api_key()
    all_rows = []
    for zone in RENEWABLE_ZONES:
        if api_key and api_key != 'demo_key':
            data = fetch_weather_for_zone(zone['lat'], zone['lon'], api_key)
            if data:
                all_rows.extend(parse_forecast_to_rows(data, zone))
                continue
        all_rows.extend(generate_synthetic_rows(zone))
    df = pd.DataFrame(all_rows)
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        df.to_json(CACHE_FILE, orient='records', date_format='iso')
    except Exception:
        pass
    return df

def get_aggregated_indices(df):
    agg = df.groupby(['date', 'hour']).agg({
        'solar_gen_index': 'sum',
        'wind_gen_index': 'sum',
        'temperature': 'mean',
        'clouds': 'mean',
        'wind_speed': 'mean',
    }).reset_index()
    agg.rename(columns={
        'solar_gen_index': 'solar_index',
        'wind_gen_index': 'wind_index',
        'temperature': 'avg_temp_vde',
        'clouds': 'avg_clouds_vde',
        'wind_speed': 'avg_wind_vde',
    }, inplace=True)
    agg['renewable_index'] = agg['solar_index'] + agg['wind_index']
    agg['solar_ratio'] = np.where(agg['renewable_index'] > 0,
                                   agg['solar_index'] / agg['renewable_index'], 0.5)
    return agg

def get_cached_indices():
    if not os.path.exists(CACHE_FILE):
        return None
    try:
        df = pd.read_json(CACHE_FILE)
        ts = pd.to_datetime(df['datetime']).max()
        if datetime.now() - ts < timedelta(hours=3):
            return df
    except Exception:
        pass
    return None

def get_renewable_indices():
    cached = get_cached_indices()
    if cached is not None:
        return get_aggregated_indices(cached)
    df = fetch_all_zones()
    return get_aggregated_indices(df)

def get_renewable_forecast(days=2):
    df = get_renewable_indices()
    if df is None or len(df) == 0:
        return []
    now = datetime.now()
    target_dates = [(now + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(days)]
    df['date_str'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    filtered = df[df['date_str'].isin(target_dates)]
    return filtered.drop(columns=['date_str']).to_dict('records') if len(filtered) > 0 else []

def get_zone_details():
    cached = get_cached_indices()
    if cached is not None:
        df = cached
    else:
        df = fetch_all_zones()
    if df is None or len(df) == 0:
        df = fetch_all_zones()
    now = datetime.now()
    cutoff = now - timedelta(hours=48)
    recent = df[pd.to_datetime(df['datetime']) >= cutoff].sort_values('datetime')
    if len(recent) < 1:
        recent = df.sort_values('datetime')
    result = {}
    for zone in RENEWABLE_ZONES:
        zdata = recent[recent['zone'] == zone['name']]
        if len(zdata) > 0:
            result[zone['name']] = {
                'label': zone['label'],
                'data': zdata[['date', 'hour', 'solar_gen_index', 'wind_gen_index',
                               'temperature', 'clouds', 'wind_speed']].tail(48).to_dict('records')
            }
    return result
