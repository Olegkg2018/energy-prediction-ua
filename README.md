# Energy Prediction UA

Прогнозування ціни електроенергії в Україні для юридичних осіб.  
XGBoost модель + PuLP оптимізація батареї + Flask веб-інтерфейс.

## Можливості

- **Прогноз цін РДН** на добу наперед (XGBoost, 23 фічі)
- **Оптимізація батареї** 1 МВт / 4 МВт·год (PuLP, до 2 циклів/добу для весняно-осінньої duck curve)
- **Реальні дані OREE** — РДН та ВДР (внутрішньодобовий ринок), 24:00 → наступний день
- **Стан ОЕС** — генерація по типах через ENTSO-E Transparency (АЕС, ТЕС, ГЕС, СЕС, ВЕС)
- **Індекс ВДЕ** — погода в 5 зонах генерації (Південь, Запоріжжя, Дніпро, Одеса, Карпати), сонячний та вітровий індекс
- **Прогноз погоди** — OpenWeather OneCall 3.0 (кешування на 3 год), інтерполяція 3→1 год для прогнозу
- **Згладжування цін** — ковзне середнє ±2 год для усунення зубців у прогнозі
- **Ручні фактори** — відключення АЕС, ризик прильотів (зберігаються в JSON)
- **Дашборд** — статистика, графіки, аналіз факторів, порівняння прогнозу з реальними цінами OREE
- **Перетренування моделі** — через `/train` (POST) з очищенням кешу

## Фічі моделі (23 ознаки)

- Базові: `hour`, `dayofweek`, `month`, `day`, `is_weekend`, `is_holiday`, `sin_hour`, `cos_hour`, `sin_month`, `cos_month`, `temperature`, `solar_radiation`
- ВДЕ: `solar_index`, `wind_index`, `renewable_index`
- ОЕС: `nuclear_share`, `thermal_share`, `hydro_share`, `solar_share`, `wind_share`, `res_share`, `total_gen_mw`
- Тренд: `days_since_epoch`

## Швидкий старт

### Через Docker (рекомендовано)

```bash
git clone https://github.com/Olegkg2018/energy-prediction-ua.git
cd energy-prediction-ua
cp .env.example .env
# nano .env — вставити API ключі (не обов'язково, без них — синтетичні дані)
docker compose up -d
```

Відкрити http://localhost:5000

### Без Docker

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python collectors/data_generator.py
python app.py
```

## API ключі (опціонально)

| Ключ | Де взяти | Навіщо |
|------|----------|--------|
| `OPENWEATHER_API_KEY` | https://openweathermap.org/api | Реальна погода (5 зон ВДЕ) |
| `ENTSOE_API_KEY` | https://transparency.entsoe.eu | Реальна генерація ОЕС |

Без ключів працює на синтетичних даних.

## API Endpoints

| Endpoint | Опис |
|----------|------|
| `/` | Дашборд |
| `/predict` | Прогноз + оптимізація |
| `/about` | Інформація про проект |
| `/api/stats` | Статистика моделі (MAE, R², ознаки) |
| `/api/model_info` | Інформація про модель |
| `/api/predict?date=YYYY-MM-DD` | Прогноз цін на добу (24 год) |
| `/api/predict_multi?days=3` | Прогноз на декілька діб |
| `/api/optimize?date=YYYY-MM-DD&capacity=4&power=1` | Оптимізація батареї (PuLP, до 2 циклів) |
| `/api/factors?date=YYYY-MM-DD` | Аналіз факторів впливу на ціну |
| `/api/factors/manual` | GET/POST ручних факторів (nuclear_outage, missile_risk) |
| `/api/generation_mix` | Поточна генерація ОЕС по типах |
| `/api/generation_timeseries` | Часовий ряд генерації за 7 днів |
| `/api/renewable_index` | Індекс ВДЕ по 5 зонах + прогноз |
| `/api/oree_prices` | Історичні ціни РДН (60 днів) |
| `/api/idm_prices` | Ціни ВДР (60 днів) |
| `/api/live_prices` | Останні 48 год OREE |
| `/api/forecast_weather` | Прогноз погоди на 3 дні |
| `/api/historical_surplus` | Історичний надлишок генерації |
| `/api/refresh_data` | Примусове оновлення всіх кешів |
| `/train` (POST) | Перетренувати модель |

## Архітектура

```
energy-prediction-ua/
├── app.py                     # Flask сервер (20+ REST роутів)
├── collectors/
│   ├── oree.py                # Парсинг OREE (РДН + ВДР), 24:00→наст. день
│   ├── weather.py             # OpenWeather OneCall 3.0 + кеш + інтерполяція
│   ├── renewable_index.py     # Індекси ВДЕ (5 зон + прогноз)
│   └── generation_mix.py      # ENTSO-E / Ukrenergo генерація ОЕС
├── model/
│   ├── train.py               # XGBoost навчання (23 фічі)
│   ├── predict.py             # Прогноз на добу + згладжування ±2 год
│   └── model.pkl              # Натренована модель
├── optimizer/
│   └── scheduler.py           # PuLP батарея (до 2 циклів/добу)
├── data/
│   ├── loader.py              # Завантаження + feature engineering
│   ├── prices_2025.csv        # Історичні ціни (статика)
│   ├── oree_prices.feather    # OREE кеш
│   ├── cache.feather          # Об'єднаний датасет
│   └── forecast_cache.json    # Погодний кеш
├── templates/                 # Jinja2 + Plotly (дашборд, прогноз, about)
├── static/                    # CSS/JS
├── Dockerfile
└── docker-compose.yml
```

## Зони моніторингу ВДЕ

| Зона | Координати | Фактор СЕС | Фактор ВЕС |
|------|-----------|-----------|-----------|
| Південь (Миколаїв) | 47.0, 32.0 | 1.3 | 1.2 |
| Запоріжжя | 47.8, 35.0 | 1.2 | 0.8 |
| Дніпро | 48.5, 35.0 | 1.1 | 0.9 |
| Одеса | 46.5, 30.5 | 1.3 | 1.0 |
| Карпати | 49.0, 24.0 | 0.7 | 1.4 |

## Ліцензія

MIT
