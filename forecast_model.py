import pandas as pd
from prophet import Prophet
import json
from collections import defaultdict

def forecast_top_products(csv_path, days_to_forecast=30):
    # 1. Read CSV
    df = pd.read_csv(csv_path, parse_dates=['ds'])

    # 2. Group all products by ID
    grouped = df.groupby('product_id')

    results = []
    product_totals = {}

    # 3. Forecast for each product
    for product_id, group in grouped:
        group = group[['ds', 'y']].copy()
        group = group.sort_values('ds')

        if len(group) < 10:
            continue  # insufficient data

        m = Prophet()
        m.fit(group.rename(columns={'ds': 'ds', 'y': 'y'}))

        future = m.make_future_dataframe(periods=days_to_forecast)
        forecast = m.predict(future)

        total_forecasted = forecast['yhat'][-days_to_forecast:].sum()
        product_totals[product_id] = total_forecasted

        results.append({
            'product_id': int(product_id),
            'forecasted_sales': float(total_forecasted),
            'peak_day': forecast.loc[forecast['yhat'].idxmax(), 'ds'].strftime('%Y-%m-%d'),
            'peak_value': float(forecast['yhat'].max())
        })

    # 4. Get top 5 products by sales
    top_5 = sorted(results, key=lambda x: x['forecasted_sales'], reverse=True)[:5]

    # 5. Return JSON with top 5 products
    return json.dumps({
        'top_products': top_5
    }, indent=2)