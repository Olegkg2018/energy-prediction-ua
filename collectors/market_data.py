import os
import requests
import pandas as pd
from datetime import datetime, timedelta

CACHE_FILE = os.path.join(os.path.dirname(__file__), '..', 'data', 'market_data.feather')
CACHE_TTL_HOURS = 6

YAHOO_TICKERS = {
    'TTF=F': 'ttf_gas_usd',
    'BZ=F': 'brent_oil_usd',
    'EURUAH=X': 'eur_uah',
    'USDUAH=X': 'usd_uah',
}

HEADERS = {'User-Agent': 'Mozilla/5.0'}


def _fetch_yahoo_history(ticker, days=30):
    try:
        url = f'https://query1.finance.yahoo.com/v8/finance/chart/{ticker}'
        params = {'range': f'{days}d', 'interval': '1d'}
        resp = requests.get(url, params=params, timeout=15, headers=HEADERS)
        if resp.status_code != 200:
            return None
        data = resp.json().get('chart', {}).get('result', [{}])[0]
        timestamps = data.get('timestamp', [])
        closes = data.get('indicators', {}).get('quote', [{}])[0].get('close', [])
        if not timestamps or not closes:
            return None
        rows = []
        for ts, close in zip(timestamps, closes):
            if close is not None:
                rows.append({
                    'date': datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d'),
                    'value': close,
                })
        return pd.DataFrame(rows) if rows else None
    except Exception as e:
        print(f"[MARKET] Yahoo {ticker} error: {e}")
        return None


def _load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            df = pd.read_feather(CACHE_FILE)
            if len(df) > 0:
                latest = pd.to_datetime(df['fetch_time']).max()
                if datetime.now() - latest < timedelta(hours=CACHE_TTL_HOURS):
                    return df
        except Exception:
            pass
    return None


def _save_cache(df):
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    df.to_feather(CACHE_FILE)


def fetch_market_data(days=30):
    cached = _load_cache()
    if cached is not None:
        return cached

    print("[MARKET] Fetching market data from Yahoo Finance...")
    all_dfs = {}

    for ticker, col_name in YAHOO_TICKERS.items():
        df = _fetch_yahoo_history(ticker, days=days)
        if df is not None and len(df) > 0:
            df = df.rename(columns={'value': col_name})
            all_dfs[col_name] = df
            print(f"[MARKET] {ticker} ({col_name}): {len(df)} rows")
        else:
            print(f"[MARKET] {ticker} ({col_name}): no data")

    if not all_dfs:
        return None

    result = None
    for col_name, df in all_dfs.items():
        if result is None:
            result = df
        else:
            result = result.merge(df, on='date', how='outer')

    if result is None:
        return None

    result = result.sort_values('date').reset_index(drop=True)
    result = result.ffill()
    result['fetch_time'] = datetime.now().isoformat()

    if 'ttf_gas_usd' in result.columns:
        result['ttf_gas_uah'] = result['ttf_gas_usd'] * result.get('eur_uah', 42.0)
    if 'ttf_gas_usd' in result.columns:
        result['ttf_change_1d'] = result['ttf_gas_usd'].pct_change(1)
        result['ttf_change_7d'] = result['ttf_gas_usd'].pct_change(7)
    if 'brent_oil_usd' in result.columns:
        result['oil_change_7d'] = result['brent_oil_usd'].pct_change(7)

    _save_cache(result)
    return result


def get_market_features(date_str):
    df = fetch_market_data(days=90)
    if df is None or len(df) == 0:
        return {}

    row = df[df['date'] <= date_str].tail(1)
    if len(row) == 0:
        return {}

    row = row.iloc[0]
    features = {}
    for col in ['ttf_gas_usd', 'ttf_gas_uah', 'brent_oil_usd', 'eur_uah', 'usd_uah',
                 'ttf_change_1d', 'ttf_change_7d', 'oil_change_7d']:
        if col in row.index:
            val = row[col]
            features[col] = float(val) if pd.notna(val) else 0.0
    return features


def get_current_prices():
    features = get_market_features(datetime.now().strftime('%Y-%m-%d'))
    return features
