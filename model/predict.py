import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from model.train import load_model, predict_hourly, load_metrics, load_quantile_models
from collectors.generation_mix import get_generation_mix

_FEATURE_CACHE = None

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
    price_mean = oree_sorted['price'].mean()

    last_date = oree_lags['datetime'].max()
    yesterday = last_date - pd.Timedelta(days=1)
    yesterday_data = oree_sorted[oree_sorted['datetime'].dt.date == yesterday.date()]
    yesterday_lookup = {}
    for _, row in yesterday_data.iterrows():
        yesterday_lookup[int(row['datetime'].hour)] = float(row['price'])

    hour_prices = {}
    for h in range(24):
        hour_prices[h] = oree_sorted[oree_sorted['hour'] == h].sort_values('datetime')['price'].values

    for h in range(24):
        hp = hour_prices[h]
        nh = len(hp)
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
                (2, 'price_lag_2h'), (3, 'price_lag_3h'), (6, 'price_lag_6h'),
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

    model = load_model()
    if model is None:
        return None

    features = prepare_prediction_features(target_date)
    if features is None:
        return None

    predictions = predict_hourly(model, features)
    midday_mask = features['hour'].between(9, 19)
    p10, p50, p90 = _get_quantile_predictions(features, midday_mask)

    return _build_results(features, predictions, p10, p50, p90, midday_mask)

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
        midday_mask = features['hour'].between(9, 19)
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
