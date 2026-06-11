import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Load real April data for calibration
df = pd.read_feather('/home/oleg/k2/energy_prediction/data/oree_prices.feather')
df['datetime'] = pd.to_datetime(df['datetime'])
apr = df[df['datetime'].dt.month == 4]

# Extract hourly patterns from April
hourly_patterns = {}
for h in range(24):
    hdata = apr[apr['datetime'].dt.hour == h]
    hourly_patterns[h] = {
        'mean': hdata['price'].mean(),
        'std': hdata['price'].std(),
        'min': hdata['price'].min(),
        'max': hdata['price'].max()
    }

# Seasonal multipliers for Jan-Mar vs April
# Winter has higher base prices, less solar generation
seasonal_mult = {
    1: 1.3,  # Jan - winter peak
    2: 1.25, # Feb - winter
    3: 1.1   # Mar - spring transition
}

# Solar dip reduction in winter (less solar generation)
solar_dip_factor = {
    1: 0.3,  # Jan - minimal solar
    2: 0.4,  # Feb
    3: 0.6   # Mar - more solar
}

np.random.seed(42)
rows = []

for month in [1, 2, 3]:
    days_in_month = [31, 28, 31][month - 1]
    for day in range(1, days_in_month + 1):
        base_date = datetime(2026, month, day)
        is_weekend = base_date.weekday() >= 5
        
        for h in range(24):
            dt = base_date.replace(hour=h)
            
            # Get April pattern for this hour
            pat = hourly_patterns[h]
            base_price = pat['mean']
            base_std = pat['std']
            
            # Apply seasonal multiplier
            base_price *= seasonal_mult[month]
            
            # Weekend effect (lower prices)
            if is_weekend:
                base_price *= 0.85
            
            # Winter solar dip adjustment (less dip in winter)
            if 10 <= h <= 15:
                dip_strength = 1 - (solar_dip_factor[month] * 0.7)
                base_price *= dip_strength
            
            # Add noise
            noise = np.random.normal(0, base_std * 0.3)
            price = max(base_price + noise, 1)
            
            # Occasional price spikes (like real data)
            if np.random.random() < 0.02:
                price = np.random.uniform(10000, 15000)
            
            # Occasional near-zero prices (surplus)
            if np.random.random() < 0.01 and 10 <= h <= 15:
                price = np.random.uniform(1, 50)
            
            rows.append({
                'datetime': dt,
                'date': dt.strftime('%Y-%m-%d'),
                'hour': h,
                'price': round(price, 2)
            })

# Create DataFrame
synth_df = pd.DataFrame(rows)
print(f'Generated {len(synth_df)} rows for Jan-Mar 2026')
print(f'Date range: {synth_df["datetime"].min()} to {synth_df["datetime"].max()}')

# Verify stats
print('\nJan-Mar 2026 stats:')
print(f'  Mean: {synth_df["price"].mean():.0f}, Median: {synth_df["price"].median():.0f}')

# Load existing OREE data
existing = pd.read_feather('/home/oleg/k2/energy_prediction/data/oree_prices.feather')
existing['datetime'] = pd.to_datetime(existing['datetime'])

# Combine and save
combined = pd.concat([synth_df, existing], ignore_index=True)
combined = combined.sort_values('datetime').reset_index(drop=True)

# Save
combined.to_feather('/home/oleg/k2/energy_prediction/data/oree_prices.feather')
print(f'\nCombined total: {len(combined)} rows')
print(f'Date range: {combined["datetime"].min()} to {combined["datetime"].max()}')
print(f'Months: {combined["datetime"].dt.month.value_counts().sort_index().to_dict()}')
