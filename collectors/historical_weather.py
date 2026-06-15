import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json
import time

CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
HISTORICAL_CACHE = os.path.join(CACHE_DIR, 'historical_weather.feather')

LAT, LON = 49.53, 30.40

OPEN_METEO_URL = 'https://archive-api.open-meteo.com/v1/archive'


def fetch_open_meteo(start_date, end_date, lat=LAT, lon=LON):
    """Fetch hourly historical weather from Open-Meteo (free, no API key)."""
    params = {
        'latitude': lat,
        'longitude': lon,
        'start_date': start_date,
        'end_date': end_date,
        'hourly': 'temperature_2m,relative_humidity_2m,cloud_cover,wind_speed_10m,shortwave_radiation',
        'timezone': 'Europe/Kyiv',
    }
    try:
        resp = requests.get(OPEN_METEO_URL, params=params, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            hourly = data.get('hourly', {})
            if not hourly or 'time' not in hourly:
                return None
            times = hourly['time']
            rows = []
            for i, t in enumerate(times):
                dt = pd.Timestamp(t)
                temp = hourly.get('temperature_2m', [None] * len(times))[i]
                humidity = hourly.get('relative_humidity_2m', [None] * len(times))[i]
                clouds = hourly.get('cloud_cover', [None] * len(times))[i]
                wind = hourly.get('wind_speed_10m', [None] * len(times))[i]
                solar = hourly.get('shortwave_radiation', [None] * len(times))[i]
                if temp is not None:
                    rows.append({
                        'datetime': dt,
                        'date': dt.strftime('%Y-%m-%d'),
                        'hour': dt.hour,
                        'temperature': round(float(temp), 1),
                        'humidity': int(round(float(humidity))) if humidity is not None else 50,
                        'clouds': int(round(float(clouds))) if clouds is not None else 50,
                        'wind_speed': round(float(wind), 1) if wind is not None else 3.0,
                        'solar_radiation': round(float(solar), 1) if solar is not None else 0.0,
                    })
            return pd.DataFrame(rows) if rows else None
        else:
            print(f'[HIST-WEATHER] Open-Meteo HTTP {resp.status_code}')
    except Exception as e:
        print(f'[HIST-WEATHER] Error: {e}')
    return None


def fetch_historical_weather_in_chunks(start_date, end_date, chunk_days=90):
    """Fetch historical weather in chunks to avoid API limits."""
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    all_chunks = []
    current = start
    while current <= end:
        chunk_end = min(current + pd.Timedelta(days=chunk_days - 1), end)
        chunk_df = fetch_open_meteo(
            current.strftime('%Y-%m-%d'),
            chunk_end.strftime('%Y-%m-%d')
        )
        if chunk_df is not None and len(chunk_df) > 0:
            all_chunks.append(chunk_df)
            print(f'[HIST-WEATHER] Fetched {current.strftime("%Y-%m-%d")} to {chunk_end.strftime("%Y-%m-%d")} ({len(chunk_df)} rows)')
        else:
            print(f'[HIST-WEATHER] Failed chunk {current.strftime("%Y-%m-%d")} to {chunk_end.strftime("%Y-%m-%d")}')
        current = chunk_end + pd.Timedelta(days=1)
        time.sleep(0.5)
    if not all_chunks:
        return None
    combined = pd.concat(all_chunks, ignore_index=True)
    combined = combined.drop_duplicates(subset='datetime').sort_values('datetime').reset_index(drop=True)
    return combined


def get_historical_weather(start_date, end_date, force=False):
    """Get historical weather, using cache when possible."""
    if not force and os.path.exists(HISTORICAL_CACHE):
        try:
            cached = pd.read_feather(HISTORICAL_CACHE)
            cached_min = cached['date'].min()
            cached_max = cached['date'].max()
            if cached_min <= start_date and cached_max >= end_date:
                mask = (cached['date'] >= start_date) & (cached['date'] <= end_date)
                subset = cached[mask]
                if len(subset) > 0:
                    return subset
        except Exception:
            pass

    print(f'[HIST-WEATHER] Fetching {start_date} to {end_date} from Open-Meteo...')
    df = fetch_historical_weather_in_chunks(start_date, end_date, chunk_days=90)

    if df is not None and len(df) > 0:
        if os.path.exists(HISTORICAL_CACHE):
            try:
                old = pd.read_feather(HISTORICAL_CACHE)
                df = pd.concat([old, df]).drop_duplicates(subset='datetime').sort_values('datetime').reset_index(drop=True)
            except Exception:
                pass
        try:
            df.to_feather(HISTORICAL_CACHE)
        except Exception:
            pass
        return df

    return None


def compute_solar_index_from_weather(dt, clouds, solar_radiation):
    """Compute solar generation index from real weather data."""
    hour = dt.hour
    if hour < 5 or hour > 21 or solar_radiation <= 0:
        return 0.0
    month = dt.month
    declination = 23.45 * np.sin(np.radians(360 / 365 * (month * 30.5 - 81)))
    hour_angle = 15 * (hour - 12)
    lat_rad = np.radians(LAT)
    elevation = np.sin(lat_rad) * np.sin(np.radians(declination)) + \
                np.cos(lat_rad) * np.cos(np.radians(declination)) * np.cos(np.radians(hour_angle))
    if elevation <= 0:
        return 0.0
    clear_sky = 1000 * elevation
    cloud_factor = 1 - (clouds / 100) * 0.75
    expected = max(0, clear_sky * cloud_factor)
    if expected > 0:
        return min(solar_radiation / expected * 1.3, 1.5)
    return 0.0


def compute_wind_index_from_weather(wind_speed):
    """Compute wind generation index from real weather data."""
    if wind_speed < 3:
        return 0.0
    if wind_speed > 25:
        return 1.0
    if wind_speed > 12:
        return 1.0
    return ((wind_speed - 3) / 9) ** 3


def compute_renewable_indices_from_weather(weather_df):
    """Compute solar_index, wind_index, renewable_index from real weather data."""
    rows = []
    for _, row in weather_df.iterrows():
        dt = row['datetime']
        clouds = row.get('clouds', 50)
        solar_rad = row.get('solar_radiation', 0)
        wind_speed = row.get('wind_speed', 3)

        solar_idx = compute_solar_index_from_weather(dt, clouds, solar_rad)
        wind_idx = compute_wind_index_from_weather(wind_speed)

        rows.append({
            'datetime': dt,
            'date': row.get('date', dt.strftime('%Y-%m-%d')),
            'hour': dt.hour,
            'solar_index': round(solar_idx, 4),
            'wind_index': round(wind_idx, 4),
            'renewable_index': round(solar_idx * 1000 + wind_idx * 500, 1),
        })
    return pd.DataFrame(rows)


def clear_historical_cache():
    global _cache
    if os.path.exists(HISTORICAL_CACHE):
        os.remove(HISTORICAL_CACHE)
