import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json
from xml.etree import ElementTree as ET
from collectors.ukrenergo import get_ukrenergo_genmix as get_ukrenergo_data

CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
CACHE_FILE = os.path.join(CACHE_DIR, 'generation_mix.feather')

UKRAINE_DOMAIN = '10YUA-WINXX--L'
ENTSOE_API_ENV = 'ENTSOE_API_KEY'

GENERATION_LABELS = {
    'B01': 'АЕС',
    'B02': 'ТЕС',
    'B03': 'ГЕС',
    'B04': 'ГАЕС',
    'B05': 'ВЕС',
    'B06': 'СЕС',
    'B07': 'Інші ВДЕ',
    'B08': 'ТЕЦ',
    'B09': 'Біомаса',
    'B10': 'Інші',
}

GENERATION_TYPES = {
    'nuclear': ['B01'],
    'thermal': ['B02', 'B08', 'B10'],
    'hydro': ['B03', 'B04'],
    'wind': ['B05'],
    'solar': ['B06'],
    'other_res': ['B07', 'B09'],
}

def get_entsoe_api_key():
    return os.environ.get(ENTSOE_API_ENV, '')

def parse_entsoe_xml(xml_text):
    root = ET.fromstring(xml_text)
    ns = {'ns': 'urn:iec62325.351:tc57wg16:451-1:generationnotification:7:0'}
    rows = []
    for ts in root.findall('.//ns:TimeSeries', ns):
        mkt_psr = ts.find('.//ns:MktPSRType', ns)
        psr_type = mkt_psr.find('ns:PSRType', ns)
        fuel_code = psr_type.find('ns:psrType', ns).text if psr_type is not None else 'B10'
        period = ts.find('ns:Period', ns)
        if period is None:
            continue
        start = period.find('ns:timeInterval/ns:start', ns)
        resolution = period.find('ns:resolution', ns)
        if start is None or resolution is None:
            continue
        start_time = datetime.fromisoformat(start.text)
        res_minutes = 60
        res_text = resolution.text.lower()
        if 'pt60m' in res_text or 'pt1h' in res_text or 'pt60' in res_text:
            res_minutes = 60
        elif 'pt30m' in res_text:
            res_minutes = 30
        elif 'pt15m' in res_text:
            res_minutes = 15
        for i, pt in enumerate(period.findall('ns:Point', ns)):
            position = pt.find('ns:position', ns)
            quantity = pt.find('ns:quantity', ns)
            if quantity is not None and quantity.text:
                ts_dt = start_time + timedelta(minutes=res_minutes * (int(position.text) - 1))
                rows.append({
                    'datetime': ts_dt,
                    'fuel_code': fuel_code,
                    'fuel_label': GENERATION_LABELS.get(fuel_code, fuel_code),
                    'generation_mw': round(float(quantity.text), 1),
                })
    return rows

def fetch_entsoe_data(days_back=7):
    api_key = get_entsoe_api_key()
    if not api_key:
        return None
    end = datetime.now()
    start = end - timedelta(days=days_back)
    url = 'https://web-api.tp.entsoe.eu/api'
    params = {
        'securityToken': api_key,
        'documentType': 'A75',
        'processType': 'A16',
        'in_Domain': UKRAINE_DOMAIN,
        'out_Domain': UKRAINE_DOMAIN,
        'periodStart': start.strftime('%Y%m%d%H%M'),
        'periodEnd': end.strftime('%Y%m%d%H%M'),
    }
    try:
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code == 200:
            rows = parse_entsoe_xml(resp.text)
            return rows
    except Exception as e:
        print(f"[GENMIX] ENTSO-E fetch error: {e}")
    return None

OPEN_METEO_URL = 'https://api.open-meteo.com/v1/forecast'
UA_LAT, UA_LON = 48.5, 31.0
TOTAL_SOLAR_CAPACITY_MW = 15000
TOTAL_WIND_CAPACITY_MW = 2000
WIND_CF = 0.30

def fetch_open_meteo_weather():
    params = {
        'latitude': UA_LAT, 'longitude': UA_LON,
        'hourly': 'shortwave_radiation,wind_speed_10m,cloud_cover',
        'timezone': 'Europe/Kyiv',
        'past_days': 3, 'forecast_days': 1,
    }
    try:
        resp = requests.get(OPEN_METEO_URL, params=params, timeout=15)
        if resp.status_code == 200:
            return resp.json().get('hourly', None)
    except Exception as e:
        print(f"[GENMIX] Open-Meteo error: {e}")
    return None

def estimate_solar_mw(radiation, hour):
    if radiation <= 0 or hour < 5 or hour > 21:
        return 0
    capacity_factor = min(radiation / 1000, 1.0)
    return round(TOTAL_SOLAR_CAPACITY_MW * capacity_factor, 1)

def estimate_wind_mw(wind_speed):
    v = wind_speed
    if v < 3:
        return 0
    if v > 25:
        cf = 0.0
    elif v > 12:
        cf = 0.35
    else:
        cf = 0.35 * ((v - 3) / 9) ** 2
    return round(TOTAL_WIND_CAPACITY_MW * min(cf, 0.35), 1)

def fetch_real_solar_wind(days=7):
    hourly = fetch_open_meteo_weather()
    if not hourly or 'time' not in hourly:
        return None
    times = hourly['time']
    radiation = hourly.get('shortwave_radiation', [0]*len(times))
    wind_speed = hourly.get('wind_speed_10m', [0]*len(times))
    rows = []
    for i, t in enumerate(times):
        dt = pd.Timestamp(t)
        r = radiation[i] if i < len(radiation) else 0
        w = wind_speed[i] if i < len(wind_speed) else 0
        rows.append({
            'datetime': dt,
            'date': dt.strftime('%Y-%m-%d'),
            'hour': dt.hour,
            'solar_radiation': r,
            'wind_speed': w,
            'solar_mw': estimate_solar_mw(r, dt.hour),
            'wind_mw': estimate_wind_mw(w),
        })
    df = pd.DataFrame(rows)
    end = datetime.now().replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(days=days)
    df = df[(df['datetime'] >= start) & (df['datetime'] <= end)]
    return df if len(df) > 0 else None

def generate_sample_mix(days=7):
    np.random.seed(42)
    rows = []
    end = datetime.now().replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(days=days)
    ts = start
    while ts < end:
        hour = ts.hour
        month = ts.month
        is_winter = month in [12, 1, 2]
        is_summer = month in [6, 7, 8]
        nuclear = np.random.normal(4500, 200)
        thermal = np.random.normal(1800 if is_winter else 1300, 300)
        if 8 <= hour <= 11 or 17 <= hour <= 21:
            thermal += 400
        hydro = np.random.normal(600, 120)
        solar = 0
        if 6 <= hour <= 19:
            solar = np.random.normal(6000, 1200) * np.sin(np.pi * (hour - 6) / 13)
            solar = max(0, solar)
            if is_summer:
                solar *= 1.4
        wind = np.random.normal(800, 300) * (1 + 0.3 if hour >= 22 or hour <= 4 else 1)
        wind = max(0, wind)
        other_res = np.random.normal(200, 50)
        import_balance = np.random.normal(-500, 300)
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
            'import_balance_mw': round(import_balance, 1),
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
        return pd.read_feather(CACHE_FILE)
    except Exception:
        return None

def save_cache(df):
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        df.to_feather(CACHE_FILE)
    except Exception:
        pass

def get_generation_mix(days=7):
    from collectors.manual_generation import get_generation_as_dataframe
    today = datetime.now().strftime('%Y-%m-%d')
    manual_df = get_generation_as_dataframe(today)
    if manual_df is not None and len(manual_df) >= 20:
        return to_aggregated(manual_df)

    cached = load_cache()
    if cached is not None:
        latest = pd.to_datetime(cached['datetime']).max()
        if datetime.now() - latest < timedelta(hours=6):
            return to_aggregated(cached)

    rows = fetch_entsoe_data(days)
    if rows and len(rows) > 0:
        df = pd.DataFrame(rows)
        df['hour'] = pd.to_datetime(df['datetime']).dt.hour
        df['date'] = pd.to_datetime(df['datetime']).dt.strftime('%Y-%m-%d')
        pivoted = df.pivot_table(
            index=['datetime', 'date', 'hour'],
            columns='fuel_code',
            values='generation_mw',
            aggfunc='sum'
        ).reset_index().fillna(0)
        pivoted.columns.name = None
        col_map = {}
        total = 0
        for c in pivoted.columns:
            if c in ['datetime', 'date', 'hour']:
                continue
            type_name = 'unknown'
            for tname, codes in GENERATION_TYPES.items():
                if c in codes:
                    type_name = tname
                    break
            col_map[c] = f'{type_name}_mw'
            total += pivoted[c]
        pivoted.rename(columns=col_map, inplace=True)
        for tname in GENERATION_TYPES:
            col = f'{tname}_mw'
            if col not in pivoted.columns:
                pivoted[col] = 0
        type_cols = [f'{t}_mw' for t in GENERATION_TYPES]
        pivoted['total_gen_mw'] = pivoted[type_cols].sum(axis=1)
        pivoted['import_balance_mw'] = 0
        pivoted = pivoted[['datetime', 'date', 'hour'] + type_cols + ['total_gen_mw', 'import_balance_mw']]
        save_cache(pivoted)
        return to_aggregated(pivoted)

    real_sw = fetch_real_solar_wind(days)
    synth = generate_sample_mix(days)
    if real_sw is not None and len(real_sw) > 0:
        synth = synth.merge(
            real_sw[['datetime', 'solar_mw', 'wind_mw']],
            on='datetime', how='left', suffixes=('_syn', '')
        )
        synth['solar_mw'] = synth['solar_mw'].fillna(synth['solar_mw_syn'])
        synth['wind_mw'] = synth['wind_mw'].fillna(synth['wind_mw_syn'])
        synth.drop(columns=['solar_mw_syn', 'wind_mw_syn'], errors='ignore', inplace=True)
        synth['total_gen_mw'] = synth[['nuclear_mw', 'thermal_mw', 'hydro_mw', 'solar_mw', 'wind_mw', 'other_res_mw']].sum(axis=1)
        print(f"[GENMIX] Real СЕС/ВЕС from Open-Meteo integrated ({len(real_sw)} hours)")
    save_cache(synth)
    return to_aggregated(synth)

def get_generation_stats(days=7):
    df = get_generation_mix(days)
    if df is None or len(df) == 0:
        return {}
    latest = df.sort_values('datetime').tail(1).iloc[0]
    recent_24h = df.tail(24)
    if 'source' in df.columns:
        src_val = str(df['source'].iloc[0]).lower()
        if 'ukrenergo' in src_val:
            source_label = 'ukrenergo'
        elif 'entsoe' in src_val:
            source_label = 'entsoe'
        else:
            source_label = 'sample'
    else:
        source_label = 'ukrenergo'
    stats = {
        'updated': datetime.now().isoformat(),
        'latest': {
            'date': latest['date'],
            'hour': int(latest['hour']) + 1,
            'nuclear_mw': round(float(latest['nuclear_mw']), 0) if 'nuclear_mw' in latest else 0,
            'thermal_mw': round(float(latest['thermal_mw']), 0) if 'thermal_mw' in latest else 0,
            'hydro_mw': round(float(latest['hydro_mw']), 0) if 'hydro_mw' in latest else 0,
            'solar_mw': round(float(latest['solar_mw']), 0) if 'solar_mw' in latest else 0,
            'wind_mw': round(float(latest['wind_mw']), 0) if 'wind_mw' in latest else 0,
            'other_res_mw': round(float(latest['other_res_mw']), 0) if 'other_res_mw' in latest else 0,
            'total_gen_mw': round(float(latest['total_gen_mw']), 0) if 'total_gen_mw' in latest else 0,
            'import_balance_mw': round(float(latest['import_balance_mw']), 0) if 'import_balance_mw' in latest else 0,
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
        'source': source_label,
    }
    return stats

def get_generation_timeseries(days=7):
    df = get_generation_mix(days)
    if df is None or len(df) == 0:
        return []
    records = df.tail(24 * days).copy()
    records['hour'] = records['hour'] + 1
    return records.to_dict('records')
