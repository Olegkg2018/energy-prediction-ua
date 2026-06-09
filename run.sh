#!/bin/bash
# Energy Prediction UA - запуск веб-програми
# Прогнозування цін на електроенергію для юр. осіб

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

# Активувати віртуальне середовище
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Згенерувати дані, якщо їх немає
if [ ! -f "data/prices_2025.csv" ] || [ ! -f "data/weather_2025.csv" ]; then
    echo ">>> Генерація тестових даних за 2025..."
    python collectors/data_generator.py
fi

# Встановити API ключі (опціонально)
# export OPENWEATHER_API_KEY="your_key_here"
# export ENTSOE_API_KEY="your_entsoe_token_here"

echo "============================================================"
echo "  Енергетичний прогнозатор - Україна"
echo "  Прогнозування цін на електроенергію для юр. осіб"
echo "  Адреса: http://127.0.0.1:5000"
echo "  Локація погоди: 49.53, 30.40 (Київська обл.)"
echo "============================================================"
echo ""

python app.py
