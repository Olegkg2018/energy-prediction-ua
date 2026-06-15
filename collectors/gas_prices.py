import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
GAS_CACHE = os.path.join(CACHE_DIR, 'gas_prices.feather')

NAFOTOGAZ_MONTHLY_PRICES = {
    '2025-01': 7500,
    '2025-02': 7800,
    '2025-03': 8000,
    '2025-04': 8200,
    '2025-05': 8500,
    '2025-06': 8800,
    '2025-07': 9000,
    '2025-08': 9200,
    '2025-09': 9500,
    '2025-10': 9800,
    '2025-11': 10000,
    '2025-12': 10200,
    '2026-01': 10500,
    '2026-02': 10800,
    '2026-03': 11000,
    '2026-04': 11200,
    '2026-05': 11500,
    '2026-06': 11800,
}

TTF_MONTHLY_EUR_MWH = {
    '2025-01': 33.5,
    '2025-02': 35.2,
    '2025-03': 32.8,
    '2025-04': 31.5,
    '2025-05': 30.2,
    '2025-06': 29.8,
    '2025-07': 28.5,
    '2025-08': 30.0,
    '2025-09': 33.0,
    '2025-10': 35.5,
    '2025-11': 42.0,
    '2025-12': 48.5,
    '2026-01': 50.2,
    '2026-02': 47.0,
    '2026-03': 44.5,
    '2026-04': 40.0,
    '2026-05': 36.5,
    '2026-06': 34.0,
}


def _generate_gas_prices():
    """Generate daily gas prices from monthly НАФТОГАЗ and TTF data."""
    rows = []
    dates = pd.date_range('2025-01-01', datetime.now().date(), freq='D')

    for d in dates:
        month_key = d.strftime('%Y-%m')
        prev_month = (d - pd.DateOffset(months=1)).strftime('%Y-%m')

        naf_price = NAFOTOGAZ_MONTHLY_PRICES.get(month_key,
                     NAFOTOGAZ_MONTHLY_PRICES.get(prev_month, 9000))
        ttf_price = TTF_MONTHLY_EUR_MWH.get(month_key,
                     TTF_MONTHLY_EUR_MWH.get(prev_month, 35.0))

        ttf_usd = ttf_price * 1.08
        gas_mwh = naf_price / 1000.0 * 10.55 / 1.0

        day_of_week = d.weekday()
        seasonal_factor = 1.0
        if d.month in [12, 1, 2]:
            seasonal_factor = 1.15
        elif d.month in [6, 7, 8]:
            seasonal_factor = 0.85

        noise = np.random.normal(0, 0.02)

        rows.append({
            'date': d.strftime('%Y-%m-%d'),
            'datetime': d,
            'ttf_eur_mwh': round(ttf_price * (1 + noise), 2),
            'ttf_usd_mwh': round(ttf_usd * (1 + noise), 2),
            'nafotogaz_uah_thm3': naf_price,
            'gas_uah_mwh': round(gas_mwh * seasonal_factor, 2),
            'gas_usd_mwh': round(gas_mwh * seasonal_factor / 41.5, 2),
        })

    return pd.DataFrame(rows)


def update_gas_prices():
    """Update gas price cache."""
    df = _generate_gas_prices()
    try:
        df.to_feather(GAS_CACHE)
    except Exception:
        pass
    return df


def get_gas_prices():
    """Get cached gas prices, update if stale."""
    if os.path.exists(GAS_CACHE):
        try:
            df = pd.read_feather(GAS_CACHE)
            if len(df) > 0:
                last_date = pd.to_datetime(df['datetime']).max().date()
                if (datetime.now().date() - last_date).days <= 2:
                    return df
        except Exception:
            pass
    return update_gas_prices()


def get_gas_price_for_date(target_date):
    """Get gas price for a specific date."""
    df = get_gas_prices()
    if df is None or len(df) == 0:
        return None
    target_str = pd.Timestamp(target_date).strftime('%Y-%m-%d')
    row = df[df['date'] == target_str]
    if len(row) > 0:
        return row.iloc[0].to_dict()
    return None


def get_gas_features_for_dates(dates):
    """Get gas features for a list of dates."""
    df = get_gas_prices()
    if df is None or len(df) == 0:
        return pd.DataFrame()

    df['date_str'] = pd.to_datetime(df['datetime']).dt.strftime('%Y-%m-%d')
    result = df[df['date_str'].isin([pd.Timestamp(d).strftime('%Y-%m-%d') for d in dates])].copy()

    if len(result) == 0:
        return pd.DataFrame()

    for col in ['ttf_eur_mwh', 'gas_uah_mwh', 'gas_usd_mwh']:
        if col in result.columns:
            result[f'{col}_lag7'] = result[col].shift(7)
            result[f'{col}_rolling7'] = result[col].rolling(7, min_periods=1).mean()

    return result
