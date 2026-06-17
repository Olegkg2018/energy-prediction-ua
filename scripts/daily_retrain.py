#!/usr/bin/env python3
"""Daily retrain script — runs after market close to update model with latest data."""
import os
import sys
import json
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def run_daily_retrain():
    print(f"[{datetime.now().isoformat()}] Starting daily retrain...")

    from collectors.oree import update_oree_prices
    try:
        update_oree_prices()
        print("[OK] OREE prices updated")
    except Exception as e:
        print(f"[WARN] OREE update failed: {e}")

    from data.loader import clear_cache, get_combined_dataset
    clear_cache()
    print("[OK] Cache cleared")

    df = get_combined_dataset()
    print(f"[OK] Dataset loaded: {len(df)} rows")

    from model.train import train_model, train_quantile_models
    model, metrics = train_model(df, force=True)
    print(f"[OK] Model trained: MAE={metrics['mae']:.2f}, R2={metrics['r2']:.4f}")

    try:
        train_quantile_models(df, force=True)
        print("[OK] Quantile models trained")
    except Exception as e:
        print(f"[WARN] Quantile training failed: {e}")

    from model.seasonality import build_seasonal_profiles
    from collectors.oree import get_cached_oree_prices
    oree = get_cached_oree_prices()
    if oree is not None:
        build_seasonal_profiles(oree)
        print("[OK] Seasonal profiles updated")

    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'mae': metrics['mae'],
        'r2': metrics['r2'],
        'n_train': metrics['n_train'],
    }
    log_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'retrain_log.jsonl')
    with open(log_path, 'a') as f:
        f.write(json.dumps(log_entry) + '\n')
    print(f"[OK] Retrain log saved")
    print(f"[DONE] MAE={metrics['mae']:.2f}")


if __name__ == '__main__':
    run_daily_retrain()
