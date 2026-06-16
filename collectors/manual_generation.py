import os
import json
import pandas as pd
from datetime import datetime

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
MANUAL_GEN_FILE = os.path.join(DATA_DIR, 'manual_generation.json')


def _ensure_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def _load():
    if os.path.exists(MANUAL_GEN_FILE):
        try:
            with open(MANUAL_GEN_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save(data):
    _ensure_dir()
    tmp = MANUAL_GEN_FILE + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, MANUAL_GEN_FILE)


def save_generation(date_str, rows):
    data = _load()
    data[date_str] = {
        'updated': datetime.now().isoformat(),
        'hours': rows
    }
    _save(data)


def get_generation(date_str):
    data = _load()
    entry = data.get(date_str)
    if entry is None:
        return None
    return entry.get('hours', [])


def get_generation_range(start_date, end_date):
    data = _load()
    rows = []
    for date_str, entry in sorted(data.items()):
        if start_date <= date_str <= end_date:
            for h in entry.get('hours', []):
                rows.append({**h, 'date': date_str})
    return rows


def get_generation_as_dataframe(date_str):
    hours = get_generation(date_str)
    if not hours:
        return None
    df = pd.DataFrame(hours)
    if len(df) == 0:
        return None
    df['date'] = date_str
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['hour'].astype(str) + ':00:00')
    total_cols = ['nuclear_mw', 'thermal_mw', 'hydro_mw', 'solar_mw', 'wind_mw']
    for c in total_cols:
        if c not in df.columns:
            df[c] = 0
    df['other_res_mw'] = 0
    df['total_gen_mw'] = df[total_cols].sum(axis=1)
    df['import_balance_mw'] = 0
    df['res_share'] = (df['solar_mw'] + df['wind_mw'] + df['other_res_mw']) / df['total_gen_mw'].clip(lower=1)
    df['nuclear_share'] = df['nuclear_mw'] / df['total_gen_mw'].clip(lower=1)
    df['thermal_share'] = df['thermal_mw'] / df['total_gen_mw'].clip(lower=1)
    df['hydro_share'] = df['hydro_mw'] / df['total_gen_mw'].clip(lower=1)
    df['solar_share'] = df['solar_mw'] / df['total_gen_mw'].clip(lower=1)
    df['wind_share'] = df['wind_mw'] / df['total_gen_mw'].clip(lower=1)
    df['source'] = 'manual'
    return df


def get_available_dates():
    data = _load()
    return sorted(data.keys())
