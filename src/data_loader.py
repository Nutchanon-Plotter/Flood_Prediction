# src/data_loader.py
import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry

def fetch_data():
    # Setup Open-Meteo API client (จากโค้ดเดิมของคุณ)
    cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)

    # 1. Fetch Flood Data
    url_flood = "https://flood-api.open-meteo.com/v1/flood"
    params_flood = {
        "latitude": 15.1589961,
        "longitude": 100.1805096,
        "daily": ["river_discharge"],
        "timezone": "Asia/Bangkok",
        "start_date": "2000-01-01",
        "end_date": "2024-12-31",
    }
    res_flood = openmeteo.weather_api(url_flood, params=params_flood)[0]
    daily_flood = res_flood.Daily()
    
    df_flood = pd.DataFrame({
        "date": pd.date_range(
            start=pd.to_datetime(daily_flood.Time(), unit="s", utc=True),
            end=pd.to_datetime(daily_flood.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=daily_flood.Interval()),
            inclusive="left"
        ),
        "river_discharge": daily_flood.Variables(0).ValuesAsNumpy()
    })

    # 2. Fetch Weather Data (ใช้เฉพาะตัวแปรที่โมเดลคุณเลือกใช้ใน Feature Importance)
    url_weather = "https://archive-api.open-meteo.com/v1/archive"
    params_weather = {
        "latitude": 15.1589961,
        "longitude": 100.1805096,
        "start_date": "2000-01-01",
        "end_date": "2024-12-31",
        "daily": ["precipitation_sum", "soil_moisture_0_to_100cm_mean", "temperature_2m_mean"],
        "timezone": "Asia/Bangkok",
    }
    res_weather = openmeteo.weather_api(url_weather, params=params_weather)[0]
    daily_weather = res_weather.Daily()
    
    df_weather = pd.DataFrame({
        "date": pd.date_range(
            start=pd.to_datetime(daily_weather.Time(), unit="s", utc=True),
            end=pd.to_datetime(daily_weather.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=daily_weather.Interval()),
            inclusive="left"
        ),
        "precipitation_sum": daily_weather.Variables(0).ValuesAsNumpy(),
        "soil_moisture_0_to_100cm_mean": daily_weather.Variables(1).ValuesAsNumpy(),
        "temperature_2m_mean": daily_weather.Variables(2).ValuesAsNumpy()
    })

    # Merge Data
    df = pd.merge(df_weather, df_flood, on='date')
    df['date'] = pd.to_datetime(df['date'])
    
    # Save raw data
    df.to_csv("data/raw/raw_data.csv", index=False)
    print("Data fetched and saved to data/raw/raw_data.csv")

if __name__ == "__main__":
    fetch_data()