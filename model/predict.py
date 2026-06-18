import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from model.train import load_model, predict_hourly, load_metrics, load_quantile_models
from collectors.generation_mix import get_generation_mix

_FEATURE_CACHE = None
_TRADE_BIAS_CACHE = {}
_DAILY_RESIDUAL_CACHE = {}

def _get_real_weather_for_target(target_date):
    """Try to get real weather forecast for target date from OpenWeather API."""
    try:
        from collectors.weather import get_forecast
        forecast = get_forecast()
        if forecast is not None and len(forecast) > 0:
            forecast['date'] = pd.to_datetime(forecast['date']).dt.strftime('%Y-%m-%d')
            target_weather = forecast[forecast['date'] == target_date]
            if len(target_weather) >= 20:
                return target_weather.sort_values('hour').reset_index(drop=True)
    except Exception:
        pass
    return None


def _get_real_renewable_for_target(target_date):
    """Try to get real renewable index forecast for target date."""
    try:
        from collectors.renewable_index import get_renewable_forecast
        forecast = get_renewable_forecast(days=3)
        if forecast is not None and len(forecast) > 0:
            target_ren = [r for r in forecast if r.get('date') == target_date]
            if len(target_ren) >= 20:
                return pd.DataFrame(target_ren).sort_values('hour').reset_index(drop=True)
    except Exception:
        pass
    return None


def _compute_hour_specific_price_features(oree_lags, pred_date):
    """Compute all price features from historical patterns (no look-ahead, no current price)."""
    result = {}
    default_cols = [
        'price_rolling_mean_24h', 'price_rolling_std_24h',
        'price_rolling_min_24h', 'price_rolling_max_24h',
        'price_rolling_median_24h',
        'price_rolling_mean_48h', 'price_rolling_std_48h',
        'price_rolling_min_48h', 'price_rolling_max_48h',
        'price_rolling_mean_168h', 'price_rolling_std_168h',
        'price_rolling_min_168h', 'price_rolling_max_168h',
        'price_rolling_skew_168h', 'price_rolling_kurt_168h',
        'price_range_48h', 'price_range_168h',
        'price_ewm_12h', 'price_ewm_48h',
        'price_delta_1h', 'price_delta_3h', 'price_delta_6h', 'price_delta_24h',
        'price_vs_yesterday', 'price_vs_last_week',
        'price_same_hour_yesterday', 'price_yoy_ratio',
        'price_ema_6', 'price_ema_12', 'price_ema_24', 'price_ema_diff', 'price_tema',
        'price_bb_pctb_24', 'price_bb_pctb_48',
        'price_momentum_24', 'price_momentum_48',
        'price_roc_12', 'price_roc_24',
    ]
    for c in default_cols:
        result[c] = [0] * 24

    if len(oree_lags) == 0:
        return result

    oree_sorted = oree_lags.sort_values('datetime').copy()
    oree_sorted['hour'] = oree_sorted['datetime'].dt.hour
    oree_sorted['dow'] = oree_sorted['datetime'].dt.dayofweek
    price_mean = oree_sorted['price'].mean()

    last_date = oree_lags['datetime'].max()
    last_date_data = oree_sorted[oree_sorted['datetime'].dt.date == last_date.date()]
    yesterday_lookup = {}
    last_date_is_weekend = last_date.dayofweek >= 5
    for _, row in last_date_data.iterrows():
        yesterday_lookup[int(row['datetime'].hour)] = float(row['price'])

    target_dt = pd.Timestamp(pred_date)
    target_is_weekend = target_dt.dayofweek >= 5
    day_type_changed = last_date_is_weekend != target_is_weekend

    same_dow_lookup = {}
    if day_type_changed:
        target_dow = target_dt.dayofweek
        same_dow_dates = oree_sorted[oree_sorted['dow'] == target_dow]['datetime'].dt.date.unique()
        if len(same_dow_dates) > 0:
            last_same_dow = same_dow_dates[-1]
            same_dow_data = oree_sorted[oree_sorted['datetime'].dt.date == last_same_dow]
            for _, row in same_dow_data.iterrows():
                same_dow_lookup[int(row['datetime'].hour)] = float(row['price'])

    hour_prices = {}
    for h in range(24):
        hour_prices[h] = oree_sorted[oree_sorted['hour'] == h].sort_values('datetime')['price'].values

    for h in range(24):
        hp = hour_prices[h]
        nh = len(hp)
        if day_type_changed and h in same_dow_lookup:
            yest = same_dow_lookup[h]
        else:
            yest = yesterday_lookup.get(h, price_mean)
        result['price_same_hour_yesterday'][h] = yest
        if nh == 0:
            continue
        last = float(hp[-1])

        for window, prefix in [(7, '24h'), (14, '48h'), (90, '168h')]:
            tail = hp[-min(window, nh):]
            result[f'price_rolling_mean_{prefix}'][h] = float(np.mean(tail))
            result[f'price_rolling_std_{prefix}'][h] = float(np.std(tail, ddof=1)) if len(tail) > 1 else 0
            result[f'price_rolling_min_{prefix}'][h] = float(np.min(tail))
            result[f'price_rolling_max_{prefix}'][h] = float(np.max(tail))

        if nh >= 3:
            result['price_rolling_median_24h'][h] = float(np.median(hp[-min(7, nh):]))
        if nh >= 10:
            t168 = hp[-min(90, nh):]
            result['price_rolling_skew_168h'][h] = float(pd.Series(t168).skew())
            result['price_rolling_kurt_168h'][h] = float(pd.Series(t168).kurtosis())

        result['price_range_48h'][h] = result['price_rolling_max_48h'][h] - result['price_rolling_min_48h'][h]
        result['price_range_168h'][h] = result['price_rolling_max_168h'][h] - result['price_rolling_min_168h'][h]

        if nh >= 2:
            result['price_ewm_12h'][h] = float(pd.Series(hp).ewm(span=min(3, nh), adjust=False).mean().iloc[-1])
            result['price_ewm_48h'][h] = float(pd.Series(hp).ewm(span=min(7, nh), adjust=False).mean().iloc[-1])

        if nh >= 2:
            result['price_delta_1h'][h] = float(hp[-1] - hp[-2])
        if nh >= 3:
            result['price_delta_3h'][h] = float(hp[-1] - hp[-3])
        if nh >= 7:
            result['price_delta_6h'][h] = float(hp[-1] - hp[-6])
        if nh >= 8:
            result['price_delta_24h'][h] = float(hp[-1] - hp[-7])

        if nh >= 3:
            result['price_vs_yesterday'][h] = float(hp[-1] - hp[-2])
            result['price_momentum_24'][h] = float(hp[-1] - hp[-2])
        if nh >= 8:
            result['price_vs_last_week'][h] = float(hp[-1] - hp[-7])
            result['price_momentum_48'][h] = float(hp[-2] - hp[-7])

        for span, col in [(3, 'price_ema_6'), (5, 'price_ema_12'), (7, 'price_ema_24')]:
            result[col][h] = float(pd.Series(hp).ewm(span=min(span, nh), adjust=False).mean().iloc[-1])
        result['price_ema_diff'][h] = result['price_ema_6'][h] - result['price_ema_24'][h]
        result['price_tema'][h] = 3 * result['price_ema_6'][h] - 3 * result['price_ema_12'][h] + result['price_ema_6'][h]

        for bb_window, suffix in [(7, '24'), (14, '48')]:
            tail_bb = hp[-min(bb_window, nh):]
            bb_ma = np.mean(tail_bb)
            bb_std = np.std(tail_bb, ddof=1) if len(tail_bb) > 1 else 0
            bb_upper = bb_ma + 2 * bb_std
            bb_lower = bb_ma - 2 * bb_std
            result[f'price_bb_pctb_{suffix}'][h] = float((last - bb_lower) / max(bb_upper - bb_lower, 1))

        if nh >= 3:
            result['price_roc_12'][h] = float((hp[-1] / max(hp[-2], 1)) - 1)
        if nh >= 8:
            result['price_roc_24'][h] = float((hp[-1] / max(hp[-7], 1)) - 1)
            result['price_yoy_ratio'][h] = float(hp[-1] / max(hp[-7], 1))

    return result


def prepare_prediction_features(target_date):
    """Generate features using real weather/renewable data when available."""
    from data.loader import (generate_synthetic_weather_for_range,
                              generate_synthetic_renewable_for_range,
                              generate_synthetic_genmix_for_range,
                              build_features)
    global _FEATURE_CACHE

    pred_date = pd.Timestamp(target_date)

    # --- Weather: try real forecast first, fall back to synthetic ---
    real_weather = _get_real_weather_for_target(target_date)
    if real_weather is not None:
        df = real_weather.copy()
        if 'datetime' not in df.columns:
            df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['hour'].astype(str) + ':00:00')
        df['datetime'] = pd.to_datetime(df['datetime'])
        if 'solar_radiation' not in df.columns:
            df['solar_radiation'] = 0.0
    else:
        np.random.seed(42)
        full_start = '2025-12-01'
        weather = generate_synthetic_weather_for_range(full_start, target_date)
        weather = weather[pd.to_datetime(weather['date']) == pred_date].copy()
        weather = weather.sort_values('hour').reset_index(drop=True)
        df = weather.copy()
        df['datetime'] = pd.to_datetime(df['datetime'])

    # --- Renewable indices: try real forecast first, fall back to synthetic ---
    real_ren = _get_real_renewable_for_target(target_date)
    ren_cols = ['solar_index', 'wind_index', 'renewable_index']
    if real_ren is not None and len(real_ren) > 0:
        for c in ren_cols:
            if c in df.columns:
                del df[c]
        if 'datetime' not in real_ren.columns:
            real_ren['datetime'] = pd.to_datetime(real_ren['date'] + ' ' + real_ren['hour'].astype(str) + ':00:00')
        df = pd.merge(df, real_ren[['datetime'] + ren_cols], on='datetime', how='left')
    else:
        np.random.seed(42)
        syn_renewable = generate_synthetic_renewable_for_range(2025, 2026)
        syn_renewable = syn_renewable[pd.to_datetime(syn_renewable['date']) == pred_date]
        for c in ren_cols:
            if c in df.columns:
                del df[c]
        df = pd.merge(df, syn_renewable[['datetime'] + ren_cols], on='datetime', how='left')

    for c in ren_cols:
        if c not in df.columns:
            df[c] = 0
        df[c] = df[c].fillna(0)

    # --- Generation mix (real ENTSO-E if available, else synthetic) ---
    genmix_cols = ['nuclear_share', 'thermal_share', 'hydro_share',
                   'solar_share', 'wind_share', 'res_share', 'total_gen_mw']
    genmix = get_generation_mix(days=14)
    genmix_ok = False
    if genmix is not None and len(genmix) > 0:
        genmix['date_str'] = pd.to_datetime(genmix['datetime']).dt.strftime('%Y-%m-%d')
        if target_date in genmix['date_str'].values:
            gm = genmix[genmix['date_str'] == target_date].copy()
            gm['datetime'] = pd.to_datetime(gm['date'].astype(str) + ' ' + gm['hour'].astype(int).astype(str) + ':00:00')
            for c in genmix_cols:
                if c in df.columns:
                    del df[c]
            df = pd.merge(df, gm[['datetime'] + genmix_cols], on='datetime', how='left')
            genmix_ok = not df[genmix_cols].isna().any().any()

    if not genmix_ok:
        np.random.seed(99)
        syn_genmix = generate_synthetic_genmix_for_range(2025, 2026)
        syn_genmix = syn_genmix[pd.to_datetime(syn_genmix['date']) == pred_date]
        for c in genmix_cols:
            if c in df.columns:
                del df[c]
        df = pd.merge(df, syn_genmix[['datetime'] + genmix_cols], on='datetime', how='left')

    for c in genmix_cols:
        if c not in df.columns:
            df[c] = 0
        df[c] = df[c].fillna(0)

    # Build time features
    df = build_features(df)
    df['solar_irradiance'] = df['solar_share'] * df['solar_radiation']
    df['solar_intensity'] = df['solar_share'] * df['sin_hour'].clip(lower=0)
    df['is_solar_dip_hour'] = ((df['hour'] >= 10) & (df['hour'] <= 15)).astype(int)
    df['solar_x_hour'] = df['solar_radiation'] * df['hour']
    df['wind_x_hour'] = df['wind_index'] * df['hour']

    # Weather anomaly features — compare forecast to historical average
    try:
        oree_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'oree_prices.feather')
        hist_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'historical_weather.feather')
        if os.path.exists(hist_path):
            hist = pd.read_feather(hist_path)
            for col, window in [('solar_radiation', 168), ('wind_speed', 168), ('temperature', 168)]:
                if col in hist.columns and col in df.columns:
                    hist['hour'] = pd.to_datetime(hist['datetime']).dt.hour
                    avg_by_hour = hist.groupby('hour')[col].mean()
                    std_by_hour = hist.groupby('hour')[col].std().clip(lower=1)
                    for idx in df.index:
                        h = int(df.loc[idx, 'hour'])
                        if h in avg_by_hour.index:
                            mean_val = avg_by_hour[h]
                            std_val = std_by_hour[h]
                            actual_val = df.loc[idx, col]
                            df.loc[idx, f'{col}_anomaly'] = (actual_val - mean_val) / std_val
                            df.loc[idx, f'{col}_vs_avg'] = actual_val - mean_val
                        else:
                            df.loc[idx, f'{col}_anomaly'] = 0
                            df.loc[idx, f'{col}_vs_avg'] = 0
    except Exception:
        for col in ['solar_radiation', 'wind_speed', 'temperature']:
            df[f'{col}_anomaly'] = 0
            df[f'{col}_vs_avg'] = 0

    df['rad_x_wind'] = df.get('solar_radiation', 0) * df.get('wind_speed', 0)
    df['renewable_boost'] = df.get('solar_radiation', 0) * df.get('solar_share', 0) + df.get('wind_speed', 0) * df.get('wind_share', 0) * 100

    # --- Gas prices for target date ---
    try:
        from collectors.gas_prices import get_gas_price_for_date
        gas = get_gas_price_for_date(target_date)
        if gas:
            for c in ['ttf_eur_mwh', 'gas_uah_mwh']:
                df[c] = gas.get(c, 0)
        else:
            df['ttf_eur_mwh'] = 35.0
            df['gas_uah_mwh'] = 9.0
    except Exception:
        df['ttf_eur_mwh'] = 35.0
        df['gas_uah_mwh'] = 9.0

    # --- ВДР-РДН spread (use last known) ---
    try:
        idm_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'idm_prices.feather')
        if os.path.exists(idm_path):
            idm = pd.read_feather(idm_path)
            idm['datetime'] = pd.to_datetime(idm['datetime'])
            oree_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'oree_prices.feather')
            if os.path.exists(oree_path):
                oree = pd.read_feather(oree_path)
                oree['datetime'] = pd.to_datetime(oree['datetime'])
                spread = pd.merge(
                    oree[['datetime', 'price']].rename(columns={'price': 'dam'}),
                    idm[['datetime', 'price']].rename(columns={'price': 'vdr'}),
                    on='datetime', how='inner'
                )
                spread['vrd_rdn_spread'] = spread['vdr'] - spread['dam']
                spread['vrd_rdn_ratio'] = spread['vdr'] / spread['dam'].clip(lower=1)
                last_spread = spread[spread['datetime'] < pred_date]
                if len(last_spread) > 0:
                    df['vrd_rdn_spread'] = float(last_spread['vrd_rdn_spread'].mean())
                    df['vrd_rdn_ratio'] = float(last_spread['vrd_rdn_ratio'].mean())
                else:
                    df['vrd_rdn_spread'] = 0
                    df['vrd_rdn_ratio'] = 1.0
            else:
                df['vrd_rdn_spread'] = 0
                df['vrd_rdn_ratio'] = 1.0
        else:
            df['vrd_rdn_spread'] = 0
            df['vrd_rdn_ratio'] = 1.0
    except Exception:
        df['vrd_rdn_spread'] = 0
        df['vrd_rdn_ratio'] = 1.0

    # --- Price lag features (NO look-ahead: exclude target date) ---
    try:
        oree_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'oree_prices.feather')
        if os.path.exists(oree_path):
            oree_lags = pd.read_feather(oree_path)[['datetime', 'price']].copy()
            oree_lags['datetime'] = pd.to_datetime(oree_lags['datetime'])
            oree_lags = oree_lags.drop_duplicates(subset='datetime')
            oree_lags = oree_lags[oree_lags['datetime'] < pred_date].copy()

            price_mean = oree_lags['price'].mean() if len(oree_lags) > 0 else 0

            # Extended price lags
            lag_refs = {}
            for lag_hours, col_name in [
                (1, 'price_lag_1h'), (2, 'price_lag_2h'), (3, 'price_lag_3h'), (6, 'price_lag_6h'),
                (12, 'price_lag_12h'), (24, 'price_lag_24h'), (48, 'price_lag_48h'),
                (168, 'price_lag_168h'), (336, 'price_lag_336h'), (504, 'price_lag_504h'),
            ]:
                ref = oree_lags.copy()
                ref['datetime'] = ref['datetime'] + pd.Timedelta(hours=lag_hours)
                ref.rename(columns={'price': col_name}, inplace=True)
                lag_refs[col_name] = ref[['datetime', col_name]]

            # Batch merge all lag columns at once
            if lag_refs:
                first_key = list(lag_refs.keys())[0]
                merged_lags = lag_refs[first_key]
                for col_name, ref_df in lag_refs.items():
                    if col_name != first_key:
                        merged_lags = pd.merge(merged_lags, ref_df, on='datetime', how='outer')
                df = pd.merge(df, merged_lags, on='datetime', how='left')
                for col_name in lag_refs:
                    df[col_name] = df[col_name].fillna(price_mean)

            # If day type changed (weekday↔weekend), override 24h lag with same DOW from last week
            target_dt_check = pd.Timestamp(pred_date)
            last_dow = oree_lags['datetime'].max().dayofweek
            target_dow = target_dt_check.dayofweek
            if (last_dow >= 5) != (target_dow >= 5) and 'price_lag_24h' in df.columns:
                same_dow_lags = oree_lags.copy()
                same_dow_lags['dow'] = same_dow_lags['datetime'].dt.dayofweek
                same_dow_data = same_dow_lags[same_dow_lags['dow'] == target_dow].sort_values('datetime')
                if len(same_dow_data) >= 24:
                    same_dow_ref = same_dow_data.tail(24).copy()
                    same_dow_ref['datetime'] = same_dow_ref['datetime'] + pd.Timedelta(days=7)
                    same_dow_ref = same_dow_ref.rename(columns={'price': 'price_lag_24h'})
                    df = df.drop(columns=['price_lag_24h'], errors='ignore')
                    df = pd.merge(df, same_dow_ref[['datetime', 'price_lag_24h']], on='datetime', how='left')
                    df['price_lag_24h'] = df['price_lag_24h'].fillna(price_mean)

            # Hour-specific rolling/technical features
            hour_features = _compute_hour_specific_price_features(oree_lags, pred_date)
            df = df.sort_values('hour').reset_index(drop=True)
            feature_df = pd.DataFrame(hour_features, index=df.index)
            df = pd.concat([df, feature_df], axis=1)

            # Gas momentum features
            if 'gas_uah_mwh' in df.columns:
                df['gas_momentum_7d'] = 0
                df['gas_rolling_std_7d'] = 0
            if 'ttf_eur_mwh' in df.columns:
                df['spark_spread'] = df.get('price_lag_24h', 0) - df['ttf_eur_mwh'] * 0.4
                df['spark_spread_lag7'] = df.get('spark_spread', 0)

            # MA terms (ARIMA moving average)
            if 'price_lag_1h' in df.columns:
                df['price_ma_24h'] = df['price_same_hour_yesterday']
                df['price_ma_168h'] = df['price_same_hour_yesterday']

            # Seasonal features from profiles
            try:
                from model.seasonality import load_profiles, compute_seasonal_features
                target_dt_s = pd.Timestamp(pred_date)
                df['dayofweek'] = target_dt_s.dayofweek
                df['month'] = target_dt_s.month
                df = compute_seasonal_features(df)
            except Exception:
                df['hourly_seasonal'] = 0
                df['weekly_seasonal'] = 0
                df['price_residual'] = 0

            # price_residual is computed at prediction time using lagged data
            # (was 0 during prediction = data leakage from training)
        else:
            for c in ['price_lag_2h', 'price_lag_3h', 'price_lag_6h', 'price_lag_12h',
                       'price_lag_24h', 'price_lag_48h', 'price_lag_168h', 'price_lag_336h', 'price_lag_504h']:
                df[c] = 0
    except Exception as e:
        import traceback
        traceback.print_exc()
        for c in ['price_lag_24h', 'price_lag_168h', 'price_rolling_mean_24h', 'price_rolling_std_24h',
                   'price_delta_1h', 'price_vs_yesterday']:
            df[c] = 0

    return df



def _get_quantile_predictions(features, midday_mask):
    import json as _json
    qm = load_quantile_models()
    if qm is None or not midday_mask.any():
        return None, None, None
    _qc_path = os.path.join(os.path.dirname(__file__), 'quantile_config.json')
    if os.path.exists(_qc_path):
        with open(_qc_path) as _qf:
            quantile_features = _json.load(_qf).get('feature_cols', [])
    else:
        quantile_features = list(features.columns)
    available_q = [c for c in quantile_features if c in features.columns]
    midday_features = features[midday_mask]
    p10 = qm.get(0.10).predict(midday_features[available_q].values) if 0.10 in qm else None
    p50 = qm.get(0.50).predict(midday_features[available_q].values) if 0.50 in qm else None
    p90 = qm.get(0.90).predict(midday_features[available_q].values) if 0.90 in qm else None
    return p10, p50, p90


def _compute_hourly_trade_bias(model, target_date):
    """Estimate per-hour residual bias from recent history before target_date."""
    cache_key = str(target_date)
    if cache_key in _TRADE_BIAS_CACHE:
        return _TRADE_BIAS_CACHE[cache_key]

    oree_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'oree_prices.feather')
    if not os.path.exists(oree_path):
        _TRADE_BIAS_CACHE[cache_key] = {}
        return {}

    try:
        oree = pd.read_feather(oree_path)[['datetime', 'date', 'hour', 'price']].copy()
        oree['date'] = pd.to_datetime(oree['date']).dt.normalize()
        target_dt = pd.Timestamp(target_date).normalize()

        solar_hours = list(range(10, 17))
        evening_hours = [19, 20, 21, 22, 23]

        # Longer lookback for solar dip, shorter for volatile evening peaks.
        solar_dates = sorted([d for d in oree['date'].unique() if d < target_dt])[-30:]
        evening_dates = sorted([d for d in oree['date'].unique() if d < target_dt])[-7:]

        def day_residual_map(day_ts):
            day_key = pd.Timestamp(day_ts).strftime('%Y-%m-%d')
            if day_key in _DAILY_RESIDUAL_CACHE:
                return _DAILY_RESIDUAL_CACHE[day_key]

            features = prepare_prediction_features(day_key)
            if features is None or len(features) == 0:
                _DAILY_RESIDUAL_CACHE[day_key] = {}
                return {}

            preds = predict_hourly(model, features)
            if preds is None or len(preds) == 0:
                _DAILY_RESIDUAL_CACHE[day_key] = {}
                return {}

            pred_df = pd.DataFrame({
                'hour': features['hour'].astype(int).values,
                'pred': np.asarray(preds, dtype=float),
            })
            actual_df = oree[oree['date'] == pd.Timestamp(day_ts)][['hour', 'price']].rename(columns={'price': 'actual'})
            merged = pd.merge(actual_df, pred_df, on='hour', how='inner')
            if merged.empty:
                _DAILY_RESIDUAL_CACHE[day_key] = {}
                return {}

            merged['residual'] = merged['actual'] - merged['pred']
            res_map = {int(r['hour']): float(r['residual']) for _, r in merged[['hour', 'residual']].iterrows()}
            _DAILY_RESIDUAL_CACHE[day_key] = res_map
            return res_map

        solar_res = {h: [] for h in solar_hours}
        for d in solar_dates:
            res_map = day_residual_map(d)
            for h in solar_hours:
                if h in res_map:
                    solar_res[h].append(res_map[h])

        evening_res = {h: [] for h in evening_hours}
        for d in evening_dates:
            res_map = day_residual_map(d)
            for h in evening_hours:
                if h in res_map:
                    evening_res[h].append(res_map[h])

        bias = {}
        for h in solar_hours:
            vals = solar_res.get(h, [])
            if len(vals) >= 5:
                # Stronger cap for solar dip hours where overprediction can be extreme.
                bias[h] = float(np.clip(np.median(vals), -2500, 2500))

        for h in evening_hours:
            vals = evening_res.get(h, [])
            if len(vals) >= 5:
                # Evening residuals are less stable; use tighter clipping.
                bias[h] = float(np.clip(np.median(vals), -1200, 1200))

        _TRADE_BIAS_CACHE[cache_key] = bias
        return bias
    except Exception:
        _TRADE_BIAS_CACHE[cache_key] = {}
        return {}


def _apply_trade_bias_correction(features, predictions, hourly_bias):
    if predictions is None or len(predictions) == 0 or not hourly_bias:
        return predictions

    corrected = np.asarray(predictions, dtype=float).copy()
    hours = features['hour'].astype(int).values
    for i, h in enumerate(hours):
        if h in hourly_bias:
            corrected[i] += hourly_bias[h]
    return corrected

def _build_results(features, predictions, p10_all=None, p50_all=None, p90_all=None, midday_mask=None):
    results = []
    q_idx = 0
    for idx, (_, row) in enumerate(features.iterrows()):
        hour = int(row['hour'])
        price = float(predictions[idx]) if predictions is not None and idx < len(predictions) else 0
        entry = {
            'hour': f"{(hour % 24) + 1:02d}:00",
            'hour_num': hour,
            'price': max(price, 0.01),
            'temperature': round(float(row.get('temperature', 15) if pd.notna(row.get('temperature', 15)) else 15), 1),
            'humidity': int(row.get('humidity', 50) if pd.notna(row.get('humidity', 50)) else 50),
            'clouds': int(row.get('clouds', 50) if pd.notna(row.get('clouds', 50)) else 50),
            'wind_speed': round(float(row.get('wind_speed', 0) if pd.notna(row.get('wind_speed', 0)) else 0), 1),
            'solar_radiation': round(float(row.get('solar_radiation', 0) if pd.notna(row.get('solar_radiation', 0)) else 0), 1),
        }
        if midday_mask is not None and midday_mask.iloc[idx] and p50_all is not None:
            entry['price_p10'] = round(float(p10_all[q_idx]), 2) if p10_all is not None else None
            entry['price_p50'] = round(float(p50_all[q_idx]), 2) if p50_all is not None else None
            entry['price_p90'] = round(float(p90_all[q_idx]), 2) if p90_all is not None else None
            q_idx += 1
        results.append(entry)
    results.sort(key=lambda x: x['hour_num'])
    return results

def predict_next_day_prices(target_date=None):
    if target_date is None:
        target_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')

    try:
        model = load_model()
        if model is None:
            print("[PREDICT] Model not loaded")
            return None

        features = prepare_prediction_features(target_date)
        if features is None:
            print(f"[PREDICT] Features not available for {target_date}")
            return None

        predictions = predict_hourly(model, features)
        if predictions is None:
            print("[PREDICT] predict_hourly returned None")
            return None

        hourly_bias = _compute_hourly_trade_bias(model, target_date)
        predictions = _apply_trade_bias_correction(features, predictions, hourly_bias)
        midday_mask = features['hour'].between(9, 23)
        p10, p50, p90 = _get_quantile_predictions(features, midday_mask)

        return _build_results(features, predictions, p10, p50, p90, midday_mask)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[PREDICT] Error for {target_date}: {e}")
        return None

def predict_with_dates(dates):
    model = load_model()
    if model is None:
        return None

    all_predictions = {}
    for target_date in dates:
        features = prepare_prediction_features(target_date)
        if features is None:
            all_predictions[target_date] = []
            continue
        preds = predict_hourly(model, features)
        hourly_bias = _compute_hourly_trade_bias(model, target_date)
        preds = _apply_trade_bias_correction(features, preds, hourly_bias)
        midday_mask = features['hour'].between(9, 23)
        p10, p50, p90 = _get_quantile_predictions(features, midday_mask)
        all_predictions[target_date] = _build_results(features, preds, p10, p50, p90, midday_mask)
    return all_predictions

def get_model_stats():
    metrics = load_metrics()
    if metrics is None:
        return {'status': 'not_trained'}
    result = {
        'status': 'trained',
        'mae': metrics.get('mae', 0),
        'rmse': metrics.get('rmse', 0),
        'r2': metrics.get('r2', 0),
        'n_train': metrics.get('n_train', 0),
        'n_test': metrics.get('n_test', 0),
        'has_lstm': metrics.get('has_lstm', False),
        'ensemble_weights': metrics.get('ensemble_weights', {}),
    }
    ensemble_path = os.path.join(os.path.dirname(__file__), 'ensemble_config.json')
    if os.path.exists(ensemble_path):
        import json
        with open(ensemble_path) as f:
            result['ensemble'] = json.load(f)
    return result
