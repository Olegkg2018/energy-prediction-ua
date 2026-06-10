import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
PRICES_FILE = os.path.join(DATA_DIR, 'prices_2025.csv')
WEATHER_FILE = os.path.join(DATA_DIR, 'weather_2025.csv')

LAT, LON = 49.53, 30.40

def generate_price_for_day(date, base_price=1500):
    np.random.seed(hash(str(date)) % (2**31))
    dow = date.weekday()
    month = date.month
    hours = list(range(24))
    prices = []

    is_weekend = dow >= 5
    is_holiday = month == 1 and date.day in [1, 7] or month == 3 and date.day == 8 or month == 5 and date.day in [1, 5, 6, 9]

    for h in hours:
        base = base_price
        if month in [1, 12]:
            base += 400
        elif month in [6, 7, 8]:
            base -= 200

        if 7 <= h <= 10:
            base *= 1.4
        elif 17 <= h <= 21:
            base *= 1.5
        elif h >= 23 or h <= 5:
            base *= 0.6

        if is_weekend or is_holiday:
            base *= 0.75

        if 10 <= h <= 15 and month in [4, 5, 6, 7, 8, 9]:
            solar_dip = np.random.uniform(0.08, 0.35)
            base *= solar_dip

        noise = np.random.normal(0, base * 0.15)
        price = base + noise

        if np.random.random() < 0.05 and 10 <= h <= 15 and month in [5, 6, 7]:
            price = np.random.uniform(1, 100)

        if np.random.random() < 0.01:
            price = np.random.uniform(3000, 6000)

        prices.append(max(round(price, 2), 0.01))
    return prices

def generate_temperature_for_day(date):
    np.random.seed(hash(str(date)) % (2**31))
    month = date.month
    hours = list(range(24))

    if month in [12, 1, 2]:
        base_temp = np.random.uniform(-10, 5)
    elif month in [3, 4, 5]:
        base_temp = np.random.uniform(5, 20)
    elif month in [6, 7, 8]:
        base_temp = np.random.uniform(18, 32)
    else:
        base_temp = np.random.uniform(5, 18)

    temps = []
    for h in hours:
        daily_var = 6 * np.sin(np.pi * (h - 6) / 12)
        noise = np.random.normal(0, 1.5)
        temps.append(round(base_temp + daily_var + noise, 1))
    return temps

def generate_full_year():
    os.makedirs(DATA_DIR, exist_ok=True)

    print("[DATA] Генерація цін на 2025...")
    start = datetime(2025, 1, 1)
    rows = []
    for i in range(365):
        d = start + timedelta(days=i)
        prices = generate_price_for_day(d)
        row = {'date': d.strftime('%d.%m.%Y')}
        for h_idx, price in enumerate(prices):
            row[f'{h_idx:02d}'] = price
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(PRICES_FILE, index=False)
    print(f"[DATA] Створено {PRICES_FILE}: {len(df)} днів")

    print("[DATA] Генерація температури на 2025...")
    start = datetime(2025, 1, 1)
    rows = []
    for i in range(365):
        d = start + timedelta(days=i)
        temps = generate_temperature_for_day(d)
        for h_idx, temp in enumerate(temps):
            rows.append({
                'date': d.strftime('%d.%m.%Y'),
                'hour': h_idx,
                'temperature': temp
            })
    df = pd.DataFrame(rows)
    df.to_csv(WEATHER_FILE, index=False)
    print(f"[DATA] Створено {WEATHER_FILE}: {len(df)} записів")

    print("[DATA] Генерація завершена")

if __name__ == '__main__':
    generate_full_year()
