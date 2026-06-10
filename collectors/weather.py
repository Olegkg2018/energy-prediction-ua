import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json

API_KEY_ENV = 'OPENWEATHER_API_KEY'
DEFAULT_API_KEY = 'demo_key'
LAT, LON = 49.53, 30.40
CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
WEATHER_CACHE = os.path.join(CACHE_DIR, 'forecast_cache.json')

def get_api_key():
    return os.environ.get(API_KEY_ENV, DEFAULT_API_KEY)

def fetch_forecast():
    api_key = get_api_key()
    if not api_key or api_key == 'demo_key':
        return generate_synthetic_forecast()

    url = 'https://api.openweathermap.org/data/3.0/onecall'
    params = {
        'lat': LAT,
        'lon': LON,
        'exclude': 'current,minutely,alerts',
        'units': 'metric',
        'appid': api_key
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            df = parse_openweather_response(data)
            if df is not None and len(df) > 0:
                save_forecast_cache(df)
                return df
    except Exception:
        pass

    url_25 = 'https://api.openweathermap.org/data/2.5/forecast'
    params = {
        'lat': LAT,
        'lon': LON,
        'units': 'metric',
        'appid': api_key
    }
    try:
        resp = requests.get(url_25, params=params, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            df = parse_25_forecast(data)
            if df is not None and len(df) > 0:
                df = interpolate_forecast(df)
                save_forecast_cache(df)
                return df
    except Exception:
        pass
    return generate_synthetic_forecast()

def parse_openweather_response(data):
    rows = []
    for d in data.get('hourly', []):
        dt = datetime.fromtimestamp(d['dt'])
        temp = d.get('temp', 0)
        clouds = d.get('clouds', 50)
        wind = d.get('wind_speed', 0)
        weather_id = d.get('weather', [{}])[0].get('id', 800)
        solar_radiation = estimate_solar_radiation(dt, clouds)
        rows.append({
            'datetime': dt,
            'date': dt.strftime('%Y-%m-%d'),
            'hour': dt.hour,
            'temperature': temp,
            'clouds': clouds,
            'wind_speed': wind,
            'weather_id': weather_id,
            'solar_radiation': solar_radiation
        })
    return pd.DataFrame(rows)

def parse_25_forecast(data):
    rows = []
    for item in data.get('list', []):
        dt = datetime.fromtimestamp(item['dt'])
        temp = item['main'].get('temp', 0)
        clouds = item.get('clouds', {}).get('all', 50)
        wind = item.get('wind', {}).get('speed', 0)
        weather_id = item.get('weather', [{}])[0].get('id', 800)
        solar_radiation = estimate_solar_radiation(dt, clouds)
        rows.append({
            'datetime': dt,
            'date': dt.strftime('%Y-%m-%d'),
            'hour': dt.hour,
            'temperature': temp,
            'clouds': clouds,
            'wind_speed': wind,
            'weather_id': weather_id,
            'solar_radiation': solar_radiation
        })
    return pd.DataFrame(rows)

def interpolate_forecast(df):
    if df is None or len(df) == 0:
        return df
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    full = pd.DataFrame({'datetime': pd.date_range(
        df['datetime'].min().floor('h'),
        df['datetime'].max().ceil('h') - pd.Timedelta(hours=1),
        freq='h'
    )})
    full['hour'] = full['datetime'].dt.hour
    full['date'] = full['datetime'].dt.strftime('%Y-%m-%d')
    for col in ['temperature', 'clouds', 'wind_speed', 'solar_radiation', 'weather_id']:
        if col in df.columns:
            full[col] = full[['datetime']].merge(
                df[['datetime', col]], on='datetime', how='left'
            )[col]
            full[col] = full[col].interpolate(method='linear').bfill().ffill()
    return full

def estimate_solar_radiation(dt, clouds):
    hour = dt.hour
    month = dt.month
    if hour < 5 or hour > 21:
        return 0
    latitude = LAT
    declination = 23.45 * np.sin(np.radians(360 / 365 * (month * 30.5 - 81)))
    hour_angle = 15 * (hour - 12)
    elevation = np.sin(np.radians(latitude)) * np.sin(np.radians(declination)) + \
                np.cos(np.radians(latitude)) * np.cos(np.radians(declination)) * np.cos(np.radians(hour_angle))
    if elevation < 0:
        return 0
    clear_sky = 1000 * elevation
    cloud_factor = 1 - (clouds / 100) * 0.75
    return max(0, clear_sky * cloud_factor)

def generate_synthetic_forecast():
    now = datetime.now()
    rows = []
    for i in range(48):
        dt = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=i)
        hour = dt.hour
        month = dt.month
        base_temp = 5 + 15 * np.sin(np.pi * (month - 3) / 6)
        daily_variation = 7 * np.sin(np.pi * (hour - 6) / 12)
        noise = np.random.normal(0, 2)
        temp = round(base_temp + daily_variation + noise, 1)
        clouds = int(np.clip(np.random.normal(50, 25), 0, 100))
        wind = round(np.random.exponential(3), 1)
        weather_id = 800 if clouds < 30 else (802 if clouds < 70 else 804)
        solar_radiation = estimate_solar_radiation(dt, clouds)
        rows.append({
            'datetime': dt,
            'date': dt.strftime('%Y-%m-%d'),
            'hour': hour,
            'temperature': temp,
            'clouds': clouds,
            'wind_speed': wind,
            'weather_id': weather_id,
            'solar_radiation': round(solar_radiation, 1)
        })
    df = pd.DataFrame(rows)
    save_forecast_cache(df)
    return df

def save_forecast_cache(df):
    try:
        rows = df.to_dict('records')
        cache = {'timestamp': datetime.now().isoformat(), 'data': rows}
        os.makedirs(os.path.dirname(WEATHER_CACHE), exist_ok=True)
        tmp = WEATHER_CACHE + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(cache, f)
        os.replace(tmp, WEATHER_CACHE)
    except Exception:
        pass

def get_cached_forecast():
    if not os.path.exists(WEATHER_CACHE):
        return None
    try:
        with open(WEATHER_CACHE, 'r') as f:
            data = json.load(f)
        ts = datetime.fromisoformat(data['timestamp'])
        if datetime.now() - ts < timedelta(hours=3):
            return pd.DataFrame(data['data'])
    except Exception:
        pass
    return None

def get_forecast():
    cached = get_cached_forecast()
    if cached is not None:
        rows = len(cached)
        if rows > 0 and 'hour' in cached.columns:
            unique_hours = cached['hour'].nunique()
        else:
            unique_hours = 0
        if unique_hours < 18:
            return interpolate_forecast(cached)
        return cached
    return fetch_forecast()

def get_forecast_for_dates(days_ahead=2):
    df = get_forecast()
    if df is None or len(df) == 0:
        return []
    now = datetime.now()
    today = now.strftime('%Y-%m-%d')
    target_dates = [(now + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(days_ahead)]
    result = df[df['date'].isin(target_dates)][['date', 'hour', 'temperature', 'clouds', 'wind_speed', 'solar_radiation']].to_dict('records')
    return result
