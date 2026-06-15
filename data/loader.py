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
HOLIDAYS_2026 = [
    '2026-01-01', '2026-01-07', '2026-03-08', '2026-04-12',
    '2026-05-01', '2026-05-04', '2026-05-05', '2026-06-07',
    '2026-06-28', '2026-08-24', '2026-10-12', '2026-12-25'
]
ALL_HOLIDAYS = set(HOLIDAYS_2025 + HOLIDAYS_2026)

def load_oree_prices():
    if not os.path.exists(OREE_CACHE):
        return None
    try:
        df = pd.read_feather(OREE_CACHE)
        df['hour'] = pd.to_numeric(df['hour'], errors='coerce').astype(int)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['date'] = pd.to_datetime(df['datetime']).dt.strftime('%Y-%m-%d')
        return df[['datetime', 'date', 'hour', 'price']]
    except Exception:
        return None

def load_csv_prices():
    if not os.path.exists(PRICES_FILE):
        return None
    try:
        df = pd.read_csv(PRICES_FILE)
        hour_cols = [c for c in df.columns if c != 'date']
        melted = df.melt(id_vars=['date'], var_name='hour', value_name='price')
        melted['hour'] = pd.to_numeric(melted['hour'], errors='coerce')
        melted['price'] = pd.to_numeric(melted['price'], errors='coerce')
        melted.dropna(subset=['hour', 'price'], inplace=True)
        melted['hour'] = melted['hour'].astype(int)
        melted['date'] = melted['date'].astype(str).str.strip()
        if '.' in str(melted['date'].iloc[0]):
            melted['datetime'] = pd.to_datetime(melted['date'] + ' ' + melted['hour'].astype(str) + ':00:00', dayfirst=True, errors='coerce')
        else:
            melted['datetime'] = pd.to_datetime(melted['date'] + ' ' + melted['hour'].astype(str) + ':00:00', errors='coerce')
        melted.dropna(subset=['datetime'], inplace=True)
        melted['date'] = pd.to_datetime(melted['datetime']).dt.strftime('%Y-%m-%d')
        return melted[['datetime', 'date', 'hour', 'price']]
    except Exception:
        return None

def generate_synthetic_weather_for_range(start_date, end_date):
    np.random.seed(42)
    rows = []
    ts = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(hours=1)
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
        humidity = int(np.clip(np.random.normal(70 - 15 * np.sin(np.pi * (hour - 6) / 12), 15), 30, 95))
        wind_base = 4 + 2 * np.sin(np.pi * (month - 1) / 6)
        wind_speed = round(abs(np.random.normal(wind_base, 2)), 1)
        rows.append({
            'datetime': ts,
            'date': ts.strftime('%Y-%m-%d'),
            'hour': hour,
            'temperature': temp,
            'humidity': humidity,
            'clouds': clouds,
            'solar_radiation': round(solar, 1),
            'wind_speed': wind_speed,
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
        year_solar_mult = 1.0 + max(0, year - 2021) * 0.6
        year_solar_base_mw = 2500 + max(0, year - 2022) * 1200
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
                solar_base_hour = year_solar_base_mw if is_summer else year_solar_base_mw * 0.6
                solar = np.random.normal(solar_base_hour * year_solar_mult, 500) * max(0, hour_factor)
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
    df['is_holiday'] = dt.dt.strftime('%Y-%m-%d').isin(ALL_HOLIDAYS).astype(int)
    df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
    df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)
    df['sin_dayofyear'] = np.sin(2 * np.pi * dt.dt.dayofyear / 365)
    df['cos_dayofyear'] = np.cos(2 * np.pi * dt.dt.dayofyear / 365)
    df['sin_hour_of_week'] = np.sin(2 * np.pi * (df['dayofweek'] * 24 + df['hour']) / 168)
    df['cos_hour_of_week'] = np.cos(2 * np.pi * (df['dayofweek'] * 24 + df['hour']) / 168)

    df['is_month_start'] = dt.dt.is_month_start.astype(int)
    df['is_month_end'] = dt.dt.is_month_end.astype(int)
    df['is_quarter_start'] = dt.dt.is_quarter_start.astype(int)
    df['is_quarter_end'] = dt.dt.is_quarter_end.astype(int)
    df['day_of_month'] = dt.dt.day
    df['week_of_year'] = dt.dt.isocalendar().week.astype(int)
    df['is_week_before_holiday'] = 0
    for h in ALL_HOLIDAYS:
        hdate = pd.Timestamp(h)
        mask = (dt >= hdate - pd.Timedelta(days=7)) & (dt < hdate)
        df.loc[mask, 'is_week_before_holiday'] = 1

    df['season'] = df['month'].map({12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1,
                                     6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3})
    df['is_heating_season'] = df['month'].isin([10, 11, 12, 1, 2, 3]).astype(int)
    df['is_cooling_season'] = df['month'].isin([5, 6, 7, 8, 9]).astype(int)
    if 'temperature' in df.columns:
        df['temperature_squared'] = df['temperature'] ** 2
    else:
        df['temperature_squared'] = 0
    df['days_since_epoch'] = _days_since_epoch(dt)
    # Demand proxy features
    if 'temperature' in df.columns:
        df['demand_proxy'] = df['temperature'] * (1.2 - 0.4 * df['is_weekend']) * (1 + 0.3 * np.cos(2 * np.pi * (df['hour'] - 19) / 24))
        df['cooling_demand'] = np.maximum(df['temperature'] - 22, 0) * (1 - 0.3 * df['is_weekend'])
        df['heating_demand'] = np.maximum(15 - df['temperature'], 0) * (1 + 0.2 * df['is_weekend'])
    else:
        df['demand_proxy'] = 0
        df['cooling_demand'] = 0
        df['heating_demand'] = 0
    if 'solar_radiation' not in df.columns:
        df['solar_radiation'] = 0
    if 'humidity' not in df.columns:
        df['humidity'] = 50
    if 'is_solar_dip_hour' not in df.columns:
        df['is_solar_dip_hour'] = ((df['hour'] >= 10) & (df['hour'] <= 15)).astype(int)
    df['solar_x_hour'] = df.get('solar_radiation', pd.Series(0, index=df.index)) * df['hour']
    df['wind_x_hour'] = df.get('wind_index', pd.Series(0, index=df.index)) * df['hour']
    for col in ['solar_index', 'wind_index', 'renewable_index',
                'nuclear_share', 'thermal_share', 'hydro_share',
                'solar_share', 'wind_share', 'res_share',
                'total_gen_mw']:
        if col not in df.columns:
            df[col] = 0
    return df

def get_combined_dataset(use_csv=False):
    global _cache
    if _cache is not None:
        return _cache
    if os.path.exists(CACHE_FILE):
        try:
            _cache = pd.read_feather(CACHE_FILE)
            return _cache
        except Exception:
            pass

    # Prefer real OREE data; fall back to synthetic CSV
    oree_prices = load_oree_prices()

    if oree_prices is not None and len(oree_prices) >= 1000:
        prices = oree_prices.copy()
        prices['source'] = 'real'
    else:
        csv_prices = load_csv_prices() if use_csv else None
        if csv_prices is not None:
            prices = csv_prices.copy()
            prices['source'] = 'synthetic'
        else:
            from collectors.oree import get_sample_prices
            prices = get_sample_prices(365)
            prices['source'] = 'synthetic'

    price_min_date = prices['date'].min()
    price_max_date = prices['date'].max()
    years = set(pd.to_datetime(prices['datetime']).dt.year)
    min_year = min(years)
    max_year = max(years)

    # Try real historical weather from Open-Meteo first
    try:
        from collectors.historical_weather import get_historical_weather, compute_renewable_indices_from_weather
        real_weather = get_historical_weather(price_min_date, price_max_date)
        if real_weather is not None and len(real_weather) > 1000:
            print(f'[LOADER] Using REAL weather data: {len(real_weather)} rows')
            weather = real_weather
            # Compute renewable indices from real weather
            real_renewable = compute_renewable_indices_from_weather(real_weather)
            use_real_renewable = True
        else:
            print('[LOADER] Real weather unavailable, falling back to synthetic')
            weather = generate_synthetic_weather_for_range(price_min_date, price_max_date)
            use_real_renewable = False
    except Exception as e:
        print(f'[LOADER] Weather import error: {e}, using synthetic')
        weather = generate_synthetic_weather_for_range(price_min_date, price_max_date)
        use_real_renewable = False

    merged = pd.merge(prices, weather, on=['datetime', 'date', 'hour'], how='left')
    merged['temperature'] = merged['temperature'].interpolate(limit_direction='both')
    if 'humidity' in merged.columns:
        merged['humidity'] = merged['humidity'].interpolate(limit_direction='both')
    if 'clouds' not in merged.columns:
        merged['clouds'] = 50

    oree_years = set(pd.to_datetime(prices['datetime']).dt.year)
    if use_real_renewable:
        merged = pd.merge(merged, real_renewable[['datetime', 'solar_index', 'wind_index', 'renewable_index']],
                          on='datetime', how='left')
    else:
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
    merged['solar_irradiance'] = merged['solar_share'] * merged.get('solar_radiation', 0)
    merged['solar_intensity'] = merged['solar_share'] * merged['sin_hour'].clip(lower=0)
    merged['is_solar_dip_hour'] = ((merged['hour'] >= 10) & (merged['hour'] <= 15)).astype(int)

    # Gas prices (TTF + НАФТОГАЗ)
    try:
        from collectors.gas_prices import get_gas_prices
        gas_df = get_gas_prices()
        if gas_df is not None and len(gas_df) > 0:
            gas_df['date'] = pd.to_datetime(gas_df['datetime']).dt.strftime('%Y-%m-%d')
            gas_cols = ['ttf_eur_mwh', 'ttf_usd_mwh', 'nafotogaz_uah_thm3', 'gas_uah_mwh', 'gas_usd_mwh']
            merged = pd.merge(merged, gas_df[['date'] + gas_cols], on='date', how='left')
            for c in gas_cols:
                if c in merged.columns:
                    merged[c] = merged[c].ffill().bfill()
                    merged[f'{c}_lag7'] = merged[c].shift(7).ffill().fillna(merged[c].mean())
                    merged[f'{c}_rolling7'] = merged[c].rolling(7, min_periods=1).mean()
            print(f'[LOADER] Gas prices integrated: {len(gas_df)} days')
    except Exception as e:
        print(f'[LOADER] Gas prices not available: {e}')

    # ВДР-РДН spread
    try:
        idm_path = os.path.join(os.path.dirname(DATA_DIR), 'data', 'idm_prices.feather')
        if os.path.exists(idm_path):
            idm = pd.read_feather(idm_path)
            idm['datetime'] = pd.to_datetime(idm['datetime'])
            idm = idm.rename(columns={'price': 'idm_price'})
            oree_for_spread = merged[['datetime', 'price']].copy()
            oree_for_spread = oree_for_spread.rename(columns={'price': 'dam_price'})
            spread_data = pd.merge(oree_for_spread, idm[['datetime', 'idm_price']], on='datetime', how='inner')
            spread_data['vrd_rdn_spread'] = spread_data['idm_price'] - spread_data['dam_price']
            spread_data['vrd_rdn_ratio'] = spread_data['idm_price'] / spread_data['dam_price'].clip(lower=1)
            merged = pd.merge(merged, spread_data[['datetime', 'vrd_rdn_spread', 'vrd_rdn_ratio']],
                              on='datetime', how='left')
            merged['vrd_rdn_spread'] = merged['vrd_rdn_spread'].fillna(0)
            merged['vrd_rdn_ratio'] = merged['vrd_rdn_ratio'].fillna(1.0)
            print(f'[LOADER] ВДР-РДН spread integrated: {len(spread_data)} rows')
    except Exception as e:
        print(f'[LOADER] ВДР-РДН spread not available: {e}')
    # Only apply synthetic solar dip scaling to non-real data
    syn_mask = merged['source'] == 'synthetic'
    merged.loc[syn_mask, 'price'] = \
        merged['price'] * (1 - merged['solar_share'] * 1.5).clip(lower=0.03)
    merged['price'] = merged['price'].clip(lower=0.01)
    merged.sort_values('datetime', inplace=True)
    merged.reset_index(drop=True, inplace=True)
    
    # Price lag features using actual datetime differences
    ref = merged[['datetime', 'price']].copy()
    for lag_hours, col_name in [(24, 'price_lag_24h'), (168, 'price_lag_168h')]:
        ref_lag = ref.copy()
        ref_lag['datetime'] = ref_lag['datetime'] + pd.Timedelta(hours=lag_hours)
        ref_lag.rename(columns={'price': col_name}, inplace=True)
        merged = pd.merge(merged, ref_lag[['datetime', col_name]], on='datetime', how='left')
        merged[col_name] = merged[col_name].fillna(merged['price'].mean())

    merged['price_rolling_mean_24h'] = merged['price'].rolling(24, min_periods=1).mean()
    merged['price_rolling_std_24h'] = merged['price'].rolling(24, min_periods=1).std().fillna(0)
    merged['price_delta_1h'] = merged['price'].diff(1).fillna(0)
    ref_yesterday = ref.copy()
    ref_yesterday['datetime'] = ref_yesterday['datetime'] + pd.Timedelta(hours=24)
    ref_yesterday.rename(columns={'price': 'price_vs_yesterday'}, inplace=True)
    merged = pd.merge(merged, ref_yesterday[['datetime', 'price_vs_yesterday']], on='datetime', how='left')
    merged['price_vs_yesterday'] = merged['price_vs_yesterday'].fillna(merged['price'])
    merged['price_vs_yesterday'] = merged['price'] - merged['price_vs_yesterday']

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
