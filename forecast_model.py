import pandas as pd
from prophet import Prophet
import json
from collections import defaultdict

def forecast_top_products(csv_path, days_to_forecast=30):
    # 1. Чтение CSV
    df = pd.read_csv(csv_path, parse_dates=['ds'])

    # 2. Группировка всех продуктов по ID
    grouped = df.groupby('product_id')

    results = []
    product_totals = {}

    # 3. Прогноз по каждому продукту
    for product_id, group in grouped:
        group = group[['ds', 'y']].copy()
        group = group.sort_values('ds')

        if len(group) < 10:
            continue  # недостаточно данных

        m = Prophet()
        m.fit(group.rename(columns={'ds': 'ds', 'y': 'y'}))

        future = m.make_future_dataframe(periods=days_to_forecast)
        forecast = m.predict(future)

        total_forecasted = forecast['yhat'][-days_to_forecast:].sum()
        product_totals[product_id] = total_forecasted

        # Найдём день с пиком
        top_days = forecast[['ds', 'yhat']].sort_values('yhat', ascending=False).head(3)

        results.append({
            'product_id': int(product_id),
            'forecast_sum': float(total_forecasted),
            'top_days': top_days.to_dict(orient='records'),
            'full_forecast': forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(days_to_forecast).to_dict(orient='records')
        })

    # 4. Топ 5 продуктов по продажам
    top_5 = sorted(results, key=lambda x: x['forecast_sum'], reverse=True)[:5]

    # 5. Аналитика для OpenAI
    analysis_text = generate_text_summary(top_5)

    return {
        'top_5': top_5,
        'text_summary': analysis_text
    }

def generate_text_summary(top_5):
    summary = "Прогнозируемые ТОП-5 продуктов по объёму продаж на следующие дни:\n\n"
    for i, item in enumerate(top_5, 1):
        summary += f"{i}. Продукт ID: {item['product_id']}, ожидаемый объем: {round(item['forecast_sum'], 2)}\n"
        for day in item['top_days']:
            summary += f"   📅 Пик: {day['ds'].strftime('%Y-%m-%d')} — {round(day['yhat'], 2)} продаж\n"
    return summary