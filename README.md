# Energy Prediction UA

Прогнозування ціни електроенергії в Україні для юридичних осіб.  
XGBoost модель + PuLP оптимізація батареї + Flask веб-інтерфейс.

## Можливості

- **Прогноз цін РДН** на добу наперед (XGBoost, 21 фіча)
- **Оптимізація батареї** 1 МВт / 4 МВт·год (PuLP лінійне програмування)
- **Реальні дані OREE** — РДН та ВДР (внутрішньодобовий ринок)
- **Стан ОЕС** — генерація по типах через ENTSO-E Transparency (АЕС, ТЕС, ГЕС, СЕС, ВЕС)
- **Індекс ВДЕ** — погода в 5 зонах генерації (Південь, Запоріжжя, Дніпро, Одеса, Карпати), сонячний та вітровий індекс
- **Ручні фактори** — відключення АЕС, ризик прильотів
- **Дашборд** — статистика, графіки, аналіз факторів

## Фічі моделі (21 ознака)

- Базові: `hour`, `dayofweek`, `month`, `day`, `is_weekend`, `is_holiday`, `sin_hour`, `cos_hour`, `sin_month`, `cos_month`, `temperature`
- ВДЕ: `solar_index`, `wind_index`, `renewable_index`
- ОЕС: `nuclear_share`, `thermal_share`, `hydro_share`, `solar_share`, `wind_share`, `res_share`, `total_gen_mw`

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
| `/api/predict?date=YYYY-MM-DD` | Прогноз цін |
| `/api/optimize?date=YYYY-MM-DD` | Оптимізація батареї |
| `/api/factors?date=YYYY-MM-DD` | Аналіз факторів |
| `/api/generation_mix` | Стан ОЕС |
| `/api/renewable_index` | Індекс ВДЕ по зонах |
| `/api/oree_prices` | Ціни РДН |
| `/api/idm_prices` | Ціни ВДР |

## Архітектура

```
energy-prediction-ua/
├── app.py                     # Flask сервер
├── collectors/
│   ├── oree.py                # Парсинг OREE (РДН + ВДР)
│   ├── weather.py             # OpenWeather API
│   ├── renewable_index.py     # Індекси ВДЕ (5 зон)
│   └── generation_mix.py      # ENTSO-E генерація ОЕС
├── model/
│   ├── train.py               # XGBoost навчання
│   └── predict.py             # Прогноз
├── optimizer/
│   └── scheduler.py           # PuLP батарея
├── data/
│   └── loader.py              # Завантаження + feature engineering
├── templates/                 # Jinja2 + Plotly
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
