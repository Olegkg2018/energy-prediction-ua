import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from flask import Flask, render_template, jsonify, request

CURRENT_DIR = os.path.dirname(__file__)
sys.path.insert(0, CURRENT_DIR)

# Load .env file if present (without python-dotenv)
_env_path = os.path.join(CURRENT_DIR, '.env')
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith('#') and '=' in _line:
                _k, _v = _line.split('=', 1)
                os.environ.setdefault(_k.strip(), _v.strip().strip("'\""))

from data.loader import get_combined_dataset, get_data_statistics, clear_cache
from model.train import train_model, load_model
from model.predict import predict_next_day_prices, get_model_stats, predict_with_dates
from collectors.oree import get_last_7days_prices, update_oree_prices, get_latest_idm_prices
from collectors.weather import get_forecast, get_forecast_for_dates
from collectors.renewable_index import get_renewable_indices, get_zone_details, get_renewable_forecast
from collectors.generation_mix import get_generation_stats, get_generation_timeseries
from optimizer.scheduler import create_optimizer, SimpleBatteryOptimizer

app = Flask(__name__)
app.config['SECRET_KEY'] = 'energy-ua-prediction-secret'

FACTORS_FILE = os.path.join(CURRENT_DIR, 'data', 'manual_factors.json')

def load_manual_factors():
    if os.path.exists(FACTORS_FILE):
        try:
            with open(FACTORS_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return {
        'nuclear_outage': False,
        'missile_risk': 0,
        'nuclear_capacity_pct': 100,
        'notes': ''
    }

def save_manual_factors(data):
    os.makedirs(os.path.dirname(FACTORS_FILE), exist_ok=True)
    with open(FACTORS_FILE, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

@app.route('/')
def index():
    stats = get_data_statistics()
    model_info = get_model_stats()
    return render_template('index.html',
                         stats=stats,
                         model_info=model_info,
                         now=datetime.now())

@app.route('/api/stats')
def api_stats():
    return jsonify(get_data_statistics())

@app.route('/api/model_info')
def api_model_info():
    return jsonify(get_model_stats())

@app.route('/train', methods=['POST'])
def train():
    force = request.json.get('force', True) if request.is_json else True
    try:
        clear_cache()
        data = get_combined_dataset()
        model, metrics = train_model(data, force=force)
        if model is None:
            return jsonify({'success': False, 'error': 'Недостатньо даних для навчання'})
        return jsonify({'success': True, 'metrics': metrics})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/about')
def about_page():
    model_info = get_model_stats()
    return render_template('about.html', model_info=model_info)

@app.route('/predict')
def predict_page():
    today = datetime.now().strftime('%Y-%m-%d')
    tomorrow = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    day_after = (datetime.now() + timedelta(days=2)).strftime('%Y-%m-%d')
    return render_template('prediction.html',
                         today=today,
                         tomorrow=tomorrow,
                         day_after=day_after)

@app.route('/api/predict')
def api_predict():
    target_date = request.args.get('date', (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d'))
    results = predict_next_day_prices(target_date=target_date)
    if results is None:
        return jsonify({'success': False, 'error': 'Модель не навчена або немає прогнозу погоди'})
    return jsonify({'success': True, 'predictions': results, 'date': target_date})

@app.route('/api/predict_multi')
def api_predict_multi():
    dates_str = request.args.get('dates', '')
    days = request.args.get('days', 0, type=int)
    if days > 0:
        dates = [(datetime.now() + timedelta(days=i + 1)).strftime('%Y-%m-%d') for i in range(days)]
    elif dates_str:
        dates = dates_str.split(',')
    else:
        return jsonify({'success': False, 'error': 'Provide dates or days'})
    results = predict_with_dates(dates)
    if results is None:
        return jsonify({'success': False, 'error': 'Model not trained'})
    return jsonify({'success': True, 'predictions': results})

@app.route('/api/forward_curve')
def api_forward_curve():
    days = request.args.get('days', 7, type=int)
    if days < 1 or days > 90:
        return jsonify({'success': False, 'error': 'days must be 1-90'})
    dates = [(datetime.now() + timedelta(days=i + 1)).strftime('%Y-%m-%d') for i in range(days)]
    results = predict_with_dates(dates)
    if results is None:
        return jsonify({'success': False, 'error': 'Model not trained'})

    flat = []
    for date, preds in results.items():
        for p in preds:
            flat.append({'date': date, 'hour': p['hour'], 'price': p['price'],
                         'temperature': p.get('temperature'),
                         'humidity': p.get('humidity'),
                         'clouds': p.get('clouds'),
                         'wind_speed': p.get('wind_speed'),
                         'solar_radiation': p.get('solar_radiation')})

    daily_avg = {}
    for date, preds in results.items():
        prices = [p['price'] for p in preds]
        daily_avg[date] = round(sum(prices) / len(prices), 2) if prices else 0

    return jsonify({
        'success': True,
        'curve': flat,
        'daily_avg': daily_avg,
        'days': days,
        'min_price': round(min(p['price'] for p in flat), 2) if flat else 0,
        'max_price': round(max(p['price'] for p in flat), 2) if flat else 0,
        'avg_price': round(sum(p['price'] for p in flat) / len(flat), 2) if flat else 0,
    })

@app.route('/api/optimize')
def api_optimize():
    target_date = request.args.get('date', (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d'))
    capacity = request.args.get('capacity', 4, type=float)
    power = request.args.get('power', 1, type=float)
    initial_soc = request.args.get('initial_soc', 0, type=float)

    predictions = predict_next_day_prices(target_date=target_date)
    if predictions is None:
        return jsonify({'success': False, 'error': 'Немає прогнозу цін'})
    prices = [p['price'] for p in sorted(predictions, key=lambda x: x['hour_num'])]

    manual = load_manual_factors()
    factors = {
        'nuclear_outage': manual.get('nuclear_outage', False),
        'missile_risk': manual.get('missile_risk', 0),
    }

    opt_kwargs = {
        'capacity': capacity,
        'power': power,
        'initial_soc': initial_soc,
    }
    optimizer = create_optimizer(use_lp=True, **opt_kwargs)
    result, status = optimizer.optimize(prices, factors=factors)
    if result is None:
        opt_simple = SimpleBatteryOptimizer(**opt_kwargs)
        result, status = opt_simple.optimize(prices, factors=factors)
    result['date'] = target_date
    result['prices'] = prices
    result['factors'] = factors
    return jsonify({'success': True, 'optimization': result})

@app.route('/api/forecast_weather')
def api_forecast_weather():
    forecast = get_forecast_for_dates(days_ahead=3)
    return jsonify({'success': True, 'forecast': forecast})

@app.route('/api/oree_prices')
def api_oree_prices():
    try:
        prices = update_oree_prices()
        if prices is not None:
            out = prices.copy()
            out['hour'] = out['hour'] + 1
            return jsonify({
                'success': True,
                'prices': out[['date', 'hour', 'price']].to_dict('records')
            })
    except Exception as e:
        pass
    return jsonify({'success': False, 'error': 'Не вдалось отримати дані з OREE'})

@app.route('/api/idm_prices')
def api_idm_prices():
    try:
        prices = get_latest_idm_prices(days=7)
        if prices is not None:
            p = prices.copy()
            p['hour'] = p['hour'] + 1
            return jsonify({
                'success': True,
                'prices': p[['date', 'hour', 'price']].to_dict('records')
            })
    except Exception:
        pass
    return jsonify({'success': False, 'error': 'ВДР дані недоступні'})

@app.route('/api/live_prices')
def api_live_prices():
    try:
        prices = update_oree_prices()
        if prices is not None:
            latest = prices.tail(48).copy()
            latest['hour'] = latest['hour'] + 1
            return jsonify({
                'success': True,
                'prices': latest[['date', 'hour', 'price']].to_dict('records')
            })
    except Exception:
        pass
    return jsonify({'success': False, 'error': 'No data available'})

@app.route('/api/factors')
def api_factors():
    target_date = request.args.get('date', (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d'))
    predictions = predict_next_day_prices(target_date=target_date)
    if predictions is None:
        return jsonify({'success': False, 'error': 'Немає прогнозу'})

    prices = sorted(predictions, key=lambda x: x['hour_num'])
    avg_price = np.mean([p['price'] for p in prices])
    min_price = min(prices, key=lambda x: x['price'])
    max_price = max(prices, key=lambda x: x['price'])
    surplus_hours = [p for p in prices if p['price'] < min(avg_price * 0.3, 200)]
    deficit_hours = [p for p in prices if p['price'] > max(avg_price * 1.5, 3000)]
    cheap_hours = [p for p in prices if p['price'] < min(avg_price * 0.15, 50)]
    volatility = np.std([p['price'] for p in prices])

    recommended = 'discharge' if avg_price > 2000 else 'charge'

    sorted_asc = sorted(prices, key=lambda x: x['price'])
    best_charge = [{'hour': p['hour'], 'price': round(p['price'], 2)} for p in sorted_asc[:5]]
    sorted_desc = sorted(prices, key=lambda x: x['price'], reverse=True)
    best_discharge = [{'hour': p['hour'], 'price': round(p['price'], 2)} for p in sorted_desc[:5]]

    factors = {
        'avg_price': round(avg_price, 2),
        'min_price': min_price,
        'max_price': max_price,
        'surplus_hours': surplus_hours,
        'deficit_hours': deficit_hours,
        'cheap_hours': cheap_hours,
        'volatility': round(volatility, 2),
        'recommended': recommended,
        'recommended_desc': ('Продавайте (розряд) — середня ціна висока' if recommended == 'discharge'
                            else 'Купляйте (заряд) — середня ціна низька'),
        'best_charge_hours': best_charge,
        'best_discharge_hours': best_discharge,
        'num_surplus_hours': len(surplus_hours),
        'num_deficit_hours': len(deficit_hours),
        'has_surplus': len(surplus_hours) > 3,
        'has_deficit': len(deficit_hours) > 0,
        'has_10uah': any(p['price'] < 20 for p in prices),
        'analysis': []
    }

    now = datetime.now()
    month = now.month
    hour = now.hour

    is_winter = month in [1, 2, 12]
    is_summer = month in [6, 7, 8]
    is_transition = month in [3, 4, 5, 9, 10, 11]

    manual = load_manual_factors()
    if manual.get('nuclear_outage'):
        factors['analysis'].append({
            'factor': 'Відключення АЕС',
            'value': 'Аварійне відключення або ремонт',
            'impact': 'Дефіцит базової генерації, ріст цін',
            'effect': 'Вищі ціни на всі години (до +30%)'
        })
    if manual.get('missile_risk', 0) > 0:
        risk_level = 'Високий' if manual['missile_risk'] > 0.7 else 'Середній' if manual['missile_risk'] > 0.3 else 'Низький'
        factors['analysis'].append({
            'factor': 'Ризик прильотів',
            'value': f'{risk_level} ({manual["missile_risk"]*100:.0f}%)',
            'impact': 'Нічні пікові ціни, зростання невизначеності',
            'effect': 'Розряд батареї вночі, премія за ризик'
        })
    factors['manual_factors'] = manual

    factors['analysis'].append({
        'factor': 'Сезон',
        'value': 'Зима' if is_winter else 'Літо' if is_summer else 'Перехідний період',
        'impact': 'Високе споживання' if is_winter else 'Низьке споживання' if is_summer else 'Середнє споживання',
        'effect': 'Вищі ціни' if is_winter else 'Нижчі ціни вдень' if is_summer else 'Помірні ціни'
    })

    if len(cheap_hours) > 0:
        factor_note = 'Профіцит від СЕС'
        if is_winter:
            factor_note = 'Нічний профіцит від ВЕС'
        factors['analysis'].append({
            'factor': 'Профіцит генерації',
            'value': f"{len(cheap_hours)} год з ціною < 50 грн",
            'impact': factor_note,
            'effect': 'Заряджати батарею в ці години'
        })

    if len(deficit_hours) > 0:
        factors['analysis'].append({
            'factor': 'Дефіцит потужності',
            'value': f"{len(deficit_hours)} год з ціною > 2000 грн",
            'impact': 'Можливі відключення або пікове споживання',
            'effect': 'Розряджати батарею в ці години'
        })

    factors['analysis'].append({
        'factor': 'Волатильність',
        'value': f"{round(volatility, 0)} грн",
        'impact': 'Висока' if volatility > 500 else 'Середня' if volatility > 200 else 'Низька',
        'effect': 'Більше можливостей для арбітражу' if volatility > 500 else 'Помірний арбітраж'
    })

    solar_hours = sum(1 for p in prices if 6 <= p['hour_num'] <= 18)
    if solar_hours > 10:
        midday_prices = [p['price'] for p in prices if 10 <= p['hour_num'] <= 15]
        if midday_prices and np.mean(midday_prices) < avg_price * 0.7:
            factors['solar_dip'] = round(float(np.mean(midday_prices)), 2)
            factors['analysis'].append({
                'factor': 'Сонячна генерація',
                'value': f"Зниження цін вдень до {factors['solar_dip']} грн",
                'impact': 'СЕС (сонячні станції) знижують ціну вдень',
                'effect': 'Заряд в обід, розряд ввечері'
            })

    if factors.get('has_10uah'):
        factors['analysis'].append({
            'factor': 'Ціна 10 грн!',
            'value': f"Години: {', '.join(h['hour'] for h in cheap_hours[:6])}",
            'impact': 'Надлишок генерації, високі ВДЕ, низьке споживання',
            'effect': 'Максимально зарядити батарею'
        })

    try:
        idm_prices = get_latest_idm_prices(days=1)
        if idm_prices is not None and len(idm_prices) > 0:
            latest_idm = idm_prices.tail(24)
            idm_avg = latest_idm['price'].mean()
            idm_spread = idm_avg - avg_price
            factors['idm_avg_price'] = round(float(idm_avg), 2)
            factors['idm_spread'] = round(float(idm_spread), 2)
            factors['analysis'].append({
                'factor': 'Спред ВДР-РДН',
                'value': f"{idm_spread:+.0f} грн (ВДР: {idm_avg:.0f} грн)",
                'impact': 'Позитивний спред = дефіцит, негативний = профіцит',
                'effect': 'Заряд по ВДР при негативному спреді, продаж на РДН'
            })
    except Exception:
        pass

    return jsonify({'success': True, 'factors': factors, 'date': target_date})

@app.route('/api/factors/manual', methods=['GET', 'POST'])
def api_manual_factors():
    if request.method == 'POST':
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'})
        current = load_manual_factors()
        current.update({
            'nuclear_outage': data.get('nuclear_outage', current.get('nuclear_outage', False)),
            'missile_risk': max(0, min(1, float(data.get('missile_risk', current.get('missile_risk', 0))))),
            'nuclear_capacity_pct': max(0, min(100, int(data.get('nuclear_capacity_pct', current.get('nuclear_capacity_pct', 100))))),
            'notes': data.get('notes', current.get('notes', ''))
        })
        save_manual_factors(current)
        return jsonify({'success': True, 'factors': current})
    return jsonify({'success': True, 'factors': load_manual_factors()})

@app.route('/api/position_value')
def api_position_value():
    buy_price = request.args.get('buy', 0, type=float)
    sell_price = request.args.get('sell', 0, type=float)
    volume = request.args.get('volume', 1, type=float)
    days = request.args.get('days', 1, type=int)
    if days < 1 or days > 30:
        return jsonify({'success': False, 'error': 'days must be 1-30'})

    flat = []
    for i in range(days):
        date = (datetime.now() + timedelta(days=i + 1)).strftime('%Y-%m-%d')
        preds = predict_next_day_prices(target_date=date)
        if preds is None:
            continue
        for p in preds:
            flat.append({'date': date, 'hour': p['hour'], 'hour_num': p['hour_num'], 'price': p['price']})

    if not flat:
        return jsonify({'success': False, 'error': 'No predictions available'})

    total_volume = volume * 24 * days
    if buy_price > 0:
        buy_cost = buy_price * total_volume
        market_value = sum(f['price'] for f in flat)
        buy_pnl = round(market_value - buy_cost, 2)
        buy_pnl_pct = round((market_value / buy_cost - 1) * 100, 2) if buy_cost > 0 else 0
    else:
        buy_pnl = 0
        buy_pnl_pct = 0

    if sell_price > 0:
        sell_revenue = sell_price * total_volume
        market_cost = sum(f['price'] for f in flat)
        sell_pnl = round(sell_revenue - market_cost, 2)
        sell_pnl_pct = round((sell_revenue / market_cost - 1) * 100, 2) if market_cost > 0 else 0
    else:
        sell_pnl = 0
        sell_pnl_pct = 0

    return jsonify({
        'success': True,
        'position': {
            'buy_price_uah': buy_price,
            'sell_price_uah': sell_price,
            'volume_mwh': total_volume,
            'days': days,
        },
        'pnl': {
            'buy_pnl_uah': buy_pnl,
            'buy_pnl_pct': buy_pnl_pct,
            'sell_pnl_uah': sell_pnl,
            'sell_pnl_pct': sell_pnl_pct,
            'total_pnl_uah': round(buy_pnl + sell_pnl, 2),
        },
        'avg_forward_price': round(sum(f['price'] for f in flat) / len(flat), 2) if flat else 0,
        'flat': flat,
    })

@app.route('/api/historical_surplus')
def api_historical_surplus():
    try:
        data = get_combined_dataset()
        surplus = data[data['price'] < 100]
        if len(surplus) == 0:
            return jsonify({'success': False, 'error': 'No surplus data'})
        by_hour = surplus.groupby('hour')['price'].agg(['count', 'mean', 'min']).reset_index()
        result = []
        for _, r in by_hour.iterrows():
            result.append({
                'hour': f"{int(r['hour']) + 1:02d}:00",
                'count': int(r['count']),
                'avg_price': round(float(r['mean']), 2),
                'min_price': round(float(r['min']), 2)
            })
        return jsonify({'success': True, 'surplus_by_hour': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/refresh_data')
def api_refresh_data():
    clear_cache()
    try:
        update_oree_prices()
        get_latest_idm_prices()
        get_forecast()
    except Exception:
        pass
    return jsonify({'success': True, 'message': 'Дані оновлено'})

@app.route('/api/generation_mix')
def api_generation_mix():
    try:
        stats = get_generation_stats(days=7)
        return jsonify({'success': True, 'stats': stats})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/generation_timeseries')
def api_generation_timeseries():
    try:
        data = get_generation_timeseries(days=7)
        return jsonify({'success': True, 'data': data})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/renewable_index')
def api_renewable_index():
    try:
        forecast = get_renewable_forecast(days=2)
        zones = get_zone_details()
        return jsonify({'success': True, 'forecast': forecast, 'zones': zones})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    print("=" * 60)
    print("Енергетичний прогнозатор - Україна")
    print("Прогнозування цін на електроенергію для юр. осіб")
    print("=" * 60)
    try:
        data = get_combined_dataset()
        model, metrics = train_model(data)
        if model:
            print(f"[OK] Модель завантажена. R²={metrics.get('r2', '?'):.3f}")
        else:
            print("[!] Модель не навчена. Зайдіть на /train")
    except Exception as e:
        print(f"[!] Помилка ініціалізації: {e}")
    print(f"\nСервер запущено: http://127.0.0.1:5000")
    print(f"Локації погоди ВДЕ: Південь, Запоріжжя, Дніпро, Одеса, Карпати")
    print(f"Генерація ОЕС: {'ENTSO-E' if os.environ.get('ENTSOE_API_KEY') else 'Семпловані дані'}")
    app.run(host='0.0.0.0', port=5000, debug=True)
