import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import re
import calendar
import io
import time
import random

CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
OREE_CACHE = os.path.join(CACHE_DIR, 'oree_prices.feather')
IDM_CACHE = os.path.join(CACHE_DIR, 'idm_prices.feather')

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
}

def fetch_month_prices(year, month, market='DAM', zone='IPS'):
    url = 'https://www.oree.com.ua/index.php/pricectr/data_view'
    data = {
        'date': f'{month:02d}.{year}',
        'market': market,
        'zone': zone
    }
    for attempt in range(2):
        try:
            resp = requests.post(url, data=data, headers=HEADERS, timeout=15)
            resp.raise_for_status()
            result = resp.json()
            html_content = result.get('content', '')
            if html_content and html_content.strip().startswith('<'):
                tables = pd.read_html(io.StringIO(html_content))
                if tables:
                    df = tables[0]
                    date_col = None
                    for col in df.columns:
                        if 'дат' in str(col).lower():
                            date_col = col
                            break
                    if date_col is None:
                        date_col = df.columns[0]
                    df.rename(columns={date_col: 'date'}, inplace=True)
                    hour_cols = [c for c in df.columns if c != 'date']
                    melted = df.melt(id_vars=['date'], var_name='hour', value_name='price')
                    melted['hour'] = melted['hour'].astype(str).str.extract(r'(\d+)')
                    melted['hour'] = pd.to_numeric(melted['hour'], errors='coerce')
                    melted['price'] = pd.to_numeric(melted['price'], errors='coerce')
                    melted.dropna(subset=['hour', 'price'], inplace=True)
                    melted['hour'] = melted['hour'].astype(int)
                    melted['hour'] = melted['hour'] - 1
                    melted['date'] = melted['date'].astype(str).str.strip()
                    if '.' in str(melted['date'].iloc[0]):
                        melted['datetime'] = pd.to_datetime(melted['date'] + ' ' + melted['hour'].astype(str) + ':00:00', dayfirst=True, errors='coerce')
                    else:
                        melted['datetime'] = pd.to_datetime(melted['date'] + ' ' + melted['hour'].astype(str) + ':00:00', errors='coerce')
                    melted.dropna(subset=['datetime'], inplace=True)
                    melted['date'] = pd.to_datetime(melted['datetime']).dt.strftime('%Y-%m-%d')
                    return melted[['datetime', 'date', 'hour', 'price']]
        except Exception:
            if attempt == 0:
                time.sleep(random.uniform(1, 2))
            else:
                pass
    return None

def generate_sample_prices(num_days=30):
    np.random.seed(42)
    rows = []
    today = datetime.now()
    for i in range(num_days):
        d = today - timedelta(days=num_days - 1 - i)
        base = 1200 + np.random.normal(0, 300)
        for h in range(24):
            hour_factor = 1.0
            if 7 <= h <= 10:
                hour_factor = 1.4
            elif 17 <= h <= 21:
                hour_factor = 1.5
            elif h >= 23 or h <= 5:
                hour_factor = 0.6
            if d.weekday() >= 5:
                hour_factor *= 0.85
            noise = np.random.normal(0, base * 0.12)
            price = max(round(base * hour_factor + noise, 2), 0.01)
            rows.append({
                'datetime': d.replace(hour=h, minute=0, second=0, microsecond=0),
                'date': d.strftime('%Y-%m-%d'),
                'hour': h,
                'price': price
            })
    return pd.DataFrame(rows)

def get_sample_idm_prices(days=30):
    sample = generate_sample_prices(days)
    sample['price'] = sample['price'] * np.random.uniform(0.9, 1.1, len(sample))
    return sample

def fetch_month_idm_prices(year, month, zone='IPS'):
    return fetch_month_prices(year, month, market='IDM', zone=zone)

def scrape_idm_prices(date_from=None, date_to=None):
    if date_from is None:
        date_from = '2025-01-01'
    if date_to is None:
        date_to = datetime.now().strftime('%Y-%m-%d')
    fd = datetime.strptime(date_from, '%Y-%m-%d')
    td = datetime.strptime(date_to, '%Y-%m-%d')
    all_data = []
    for year in range(fd.year, td.year + 1):
        start_month = fd.month if year == fd.year else 1
        end_month = td.month if year == td.year else 12
        for month in range(start_month, end_month + 1):
            df = fetch_month_idm_prices(year, month)
            if df is not None and len(df) > 0:
                all_data.append(df)
    if not all_data:
        return None
    combined = pd.concat(all_data, ignore_index=True)
    combined = combined.drop_duplicates(subset=['datetime']).sort_values('datetime')
    combined = combined[
        (combined['datetime'] >= pd.to_datetime(date_from)) &
        (combined['datetime'] <= pd.to_datetime(date_to) + timedelta(days=1))
    ]
    combined.reset_index(drop=True, inplace=True)
    return combined if len(combined) > 0 else None

def update_idm_prices():
    df = scrape_idm_prices()
    if df is not None and len(df) > 0:
        existing = None
        if os.path.exists(IDM_CACHE):
            try:
                existing = pd.read_feather(IDM_CACHE)
            except Exception:
                pass
        if existing is not None:
            combined = pd.concat([existing, df]).drop_duplicates(subset=['datetime']).sort_values('datetime')
        else:
            combined = df
        combined.reset_index(drop=True, inplace=True)
        try:
            combined.to_feather(IDM_CACHE)
        except Exception:
            pass
        return combined
    cached = None
    if os.path.exists(IDM_CACHE):
        try:
            cached = pd.read_feather(IDM_CACHE)
        except Exception:
            pass
    if cached is not None and len(cached) > 0:
        return cached
    return get_sample_idm_prices(30)

def get_latest_idm_prices(days=7):
    df = update_idm_prices()
    if df is not None:
        cutoff = datetime.now() - timedelta(days=days)
        recent = df[pd.to_datetime(df['datetime']) >= cutoff]
        return recent if len(recent) > 0 else df.tail(24 * days)
    return None

def scrape_oree_prices(date_from=None, date_to=None):
    if date_from is None:
        date_from = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
    if date_to is None:
        date_to = datetime.now().strftime('%Y-%m-%d')

    fd = datetime.strptime(date_from, '%Y-%m-%d')
    td = datetime.strptime(date_to, '%Y-%m-%d')

    all_data = []
    for year in range(fd.year, td.year + 1):
        start_month = fd.month if year == fd.year else 1
        end_month = td.month if year == td.year else 12
        for month in range(start_month, end_month + 1):
            df = fetch_month_prices(year, month)
            if df is not None and len(df) > 0:
                all_data.append(df)
            time.sleep(random.uniform(0.5, 1.5))

    if not all_data:
        return None

    combined = pd.concat(all_data, ignore_index=True)
    combined = combined.drop_duplicates(subset=['datetime']).sort_values('datetime')
    combined = combined[
        (combined['datetime'] >= pd.to_datetime(date_from)) &
        (combined['datetime'] <= pd.to_datetime(date_to) + timedelta(days=1))
    ]
    combined.reset_index(drop=True, inplace=True)
    return combined if len(combined) > 0 else None

def get_cached_oree_prices():
    if os.path.exists(OREE_CACHE):
        try:
            return pd.read_feather(OREE_CACHE)
        except Exception:
            pass
    return None

def get_sample_prices(days=30):
    sample = generate_sample_prices(days)
    try:
        sample.to_feather(OREE_CACHE)
    except Exception:
        pass
    return sample

def update_oree_prices():
    """Fetch OREE prices from 2025-12-01 onwards, filling gaps in cache."""
    existing = get_cached_oree_prices()
    existing_months = set()
    if existing is not None and len(existing) > 0:
        existing_months = set(pd.to_datetime(existing['datetime']).dt.to_period('M'))

    today = datetime.now()
    target_months = set()
    d = pd.Timestamp('2025-01-01')
    end = pd.Timestamp(today.year, today.month, 1)
    while d <= end:
        target_months.add(d.to_period('M'))
        d += pd.DateOffset(months=1)

    # If the last cached day is behind today, re-fetch the current month
    # (handles gaps where OREE has data we haven't cached yet)
    if existing is not None and len(existing) > 0:
        last_dt = pd.to_datetime(existing['datetime']).max()
        last_date = last_dt.date()
        today_date = today.date()
        yesterday_date = today_date - timedelta(days=1)
        if last_date < yesterday_date:
            current_month = pd.Timestamp(today.year, today.month, 1).to_period('M')
            existing_months.discard(current_month)

    missing = sorted(target_months - existing_months)
    if not missing and existing is not None:
        return existing

    all_data = []
    if existing is not None and len(existing) > 0:
        all_data.append(existing)
    for period in missing:
        df = fetch_month_prices(period.year, period.month)
        if df is not None and len(df) > 0:
            all_data.append(df)
        time.sleep(random.uniform(0.5, 1.5))

    if not all_data:
        cached = get_cached_oree_prices()
        if cached is not None and len(cached) > 0:
            return cached
        return get_sample_prices(30)

    combined = pd.concat(all_data, ignore_index=True)
    combined = combined.drop_duplicates(subset=['datetime']).sort_values('datetime')
    combined.reset_index(drop=True, inplace=True)
    try:
        combined.to_feather(OREE_CACHE)
    except Exception:
        pass
    return combined

def get_latest_oree_prices(days=7):
    df = update_oree_prices()
    if df is not None:
        cutoff = datetime.now() - timedelta(days=days)
        recent = df[pd.to_datetime(df['datetime']) >= cutoff]
        return recent if len(recent) > 0 else df.tail(24 * days)
    return None

def get_last_7days_prices():
    data = get_latest_oree_prices(7)
    if data is not None:
        return data[['date', 'hour', 'price']].to_dict('records')
    return []
