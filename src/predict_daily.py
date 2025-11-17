# src/predict_daily.py
import pandas as pd
import joblib
import xgboost as xgb
import json
import numpy as np
import openmeteo_requests
import requests_cache
from retry_requests import retry
from datetime import timedelta

def fetch_combined_data():
    # Setup Open-Meteo
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    # ดึงข้อมูล: ย้อนหลัง 7 วัน (Observed) + ล่วงหน้า 7 วัน (Forecast)
    # รวมเป็น 14-15 วันต่อเนื่องกัน เพื่อให้การสร้าง Lag Feature ไม่ขาดตอน
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 15.1589961,
        "longitude": 100.1805096,
        "daily": ["precipitation_sum", "soil_moisture_0_to_100cm_mean", "temperature_2m_mean"],
        "timezone": "Asia/Bangkok",
        "past_days": 10,    # เผื่อไว้คำนวณ Lag
        "forecast_days": 7  # ทำนายล่วงหน้า 7 วัน
    }
    
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]
    
    daily = response.Daily()
    
    daily_data = {
        "date": pd.date_range(
            start=pd.to_datetime(daily.Time(), unit="s", utc=True),
            end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=daily.Interval()),
            inclusive="left"
        ),
        "precipitation_sum": daily.Variables(0).ValuesAsNumpy(),
        "soil_moisture_0_to_100cm_mean": daily.Variables(1).ValuesAsNumpy(),
        "temperature_2m_mean": daily.Variables(2).ValuesAsNumpy()
    }
    return pd.DataFrame(daily_data)

def run_prediction():
    # 1. Load Model
    model = xgb.XGBClassifier()
    model.load_model("models/xgboost_model.json")
    scaler = joblib.load("models/scaler.pkl")
    
    # 2. Fetch Data (Past + Future)
    df = fetch_combined_data()
    
    # 3. Create Lag Features (Shift ข้อมูลใน DataFrame เดียวกัน)
    # ค่า Lag ของ "วันพรุ่งนี้" จะไปดึงมาจาก "Observed Data" ของ 5-7 วันที่แล้ว
    # ค่า Lag ของ "อีก 7 วันข้างหน้า" จะไปดึงมาจาก "Forecast Data" ของ 2 วันข้างหน้า
    df['precip_lag_5'] = df['precipitation_sum'].shift(5)
    df['precip_lag_6'] = df['precipitation_sum'].shift(6)
    df['precip_lag_7'] = df['precipitation_sum'].shift(7)
    
    # 4. Filter เอาเฉพาะ "อนาคต 7 วัน" (วันนี้ + 1 ถึง วันนี้ + 7)
    # ใช้ pd.Timestamp.now() เพื่อหาเส้นแบ่งเวลาปัจจุบัน
    today = pd.Timestamp.now(tz='Asia/Bangkok').normalize()
    future_df = df[df['date'] >= today].head(7).copy()
    
    # Fill NaN (เผื่อ Lag วันแรกๆ ขาดหายไป)
    future_df = future_df.fillna(0)
    
    print(f"Predicting for dates: {future_df['date'].dt.date.values}")

    # 5. Prepare & Predict
    features = ['precipitation_sum', 'soil_moisture_0_to_100cm_mean', 'temperature_2m_mean',
                'precip_lag_5', 'precip_lag_6', 'precip_lag_7']
    
    X = future_df[features]
    X_scaled = scaler.transform(X)
    
    predictions = model.predict(X_scaled)
    probs = model.predict_proba(X_scaled)[:, 1]
    
    # 6. Output JSON (List of Objects)
    results = []
    for date, pred, prob in zip(future_df['date'], predictions, probs):
        results.append({
            "date": date.strftime('%Y-%m-%d'),
            "is_flood": int(pred),
            "flood_probability": float(prob) * 100,
            "risk_level": "High" if prob > 0.5 else "Low"
        })
    
    with open("prediction_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("7-day prediction saved to prediction_results.json")

if __name__ == "__main__":
    run_prediction()