import os
import json
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
PROFILES_PATH = os.path.join(DATA_DIR, 'seasonal_profiles.json')


def build_seasonal_profiles(oree_prices):
    df = oree_prices.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['hour'] = df['datetime'].dt.hour
    df['dow'] = df['datetime'].dt.dayofweek
    df['month'] = df['datetime'].dt.month

    hourly = df.groupby('hour')['price'].agg(['mean', 'std']).to_dict('index')
    weekly = df.groupby('dow')['price'].agg(['mean', 'std']).to_dict('index')
    monthly = df.groupby('month')['price'].agg(['mean', 'std']).to_dict('index')

    overall_mean = df['price'].mean()

    profiles = {
        'hourly': {str(k): {'mean': round(v['mean'], 1), 'std': round(v['std'], 1)} for k, v in hourly.items()},
        'weekly': {str(k): {'mean': round(v['mean'], 1), 'std': round(v['std'], 1)} for k, v in weekly.items()},
        'monthly': {str(k): {'mean': round(v['mean'], 1), 'std': round(v['std'], 1)} for k, v in monthly.items()},
        'overall_mean': round(overall_mean, 1),
    }

    _ensure_dir()
    with open(PROFILES_PATH, 'w') as f:
        json.dump(profiles, f, indent=2)
    print(f"[SEASONAL] Profiles saved: {len(hourly)} hours, {len(weekly)} days, {len(monthly)} months")
    return profiles


def load_profiles():
    if os.path.exists(PROFILES_PATH):
        with open(PROFILES_PATH) as f:
            return json.load(f)
    return None


def compute_seasonal_features(df):
    profiles = load_profiles()
    if profiles is None:
        df['hourly_seasonal'] = 0
        df['weekly_seasonal'] = 0
        df['monthly_trend'] = 0
        df['price_residual'] = 0
        return df

    hourly_means = {int(k): v['mean'] for k, v in profiles['hourly'].items()}
    weekly_means = {int(k): v['mean'] for k, v in profiles['weekly'].items()}
    monthly_means = {int(k): v['mean'] for k, v in profiles['monthly'].items()}
    overall = profiles['overall_mean']

    df['hourly_seasonal'] = df['hour'].map(hourly_means).fillna(overall)
    if 'dayofweek' in df.columns:
        df['weekly_seasonal'] = df['dayofweek'].map(weekly_means).fillna(overall)
    else:
        df['weekly_seasonal'] = overall
    if 'month' in df.columns:
        df['monthly_trend'] = df['month'].map(monthly_means).fillna(overall)
    else:
        df['monthly_trend'] = overall

    baseline = (df['hourly_seasonal'] + df['weekly_seasonal'] + df['monthly_trend']) / 3
    if 'price' in df.columns:
        df['price_residual'] = df['price'] - baseline
    else:
        df['price_residual'] = 0

    return df


def _ensure_dir():
    os.makedirs(DATA_DIR, exist_ok=True)
