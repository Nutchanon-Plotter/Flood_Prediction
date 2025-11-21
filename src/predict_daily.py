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
import datetime as dt
from sklearn.preprocessing import StandardScaler
import os
import warnings
warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------------------------------------

# ใช้พิกัดที่ขยับเข้าฝั่งเล็กน้อย (เพื่อป้องกันข้อมูล Soil Moisture เป็น NaN)
LOCATIONS = [
    {"name": "ChaoPhraya_Dam", "lat": 15.159213709346405, "lon": 100.17985248529882}, 
    {"name": "NakhonSawan_Muang_Upstream", "lat": 15.700409309316225, "lon": 100.14120663110944},
]

# ตัวแปร Weather ทั้งหมด
WEATHER_VARS = ["temperature_2m_min", "apparent_temperature_mean", "apparent_temperature_max", "apparent_temperature_min", "temperature_2m_max", "temperature_2m_mean", "daylight_duration", "sunshine_duration", "precipitation_sum", "rain_sum", "snowfall_sum", "precipitation_hours", "wind_speed_10m_max", "wind_direction_10m_dominant", "shortwave_radiation_sum", "et0_fao_evapotranspiration", "soil_moisture_0_to_100cm_mean", "soil_moisture_0_to_7cm_mean", "soil_moisture_28_to_100cm_mean", "soil_moisture_7_to_28cm_mean", "soil_temperature_0_to_100cm_mean", "soil_temperature_0_to_7cm_mean", "soil_temperature_28_to_100cm_mean", "soil_temperature_7_to_28cm_mean", "wet_bulb_temperature_2m_mean", "wet_bulb_temperature_2m_max", "wet_bulb_temperature_2m_min", "vapour_pressure_deficit_max", "surface_pressure_mean", "surface_pressure_max", "surface_pressure_min", "winddirection_10m_dominant", "wind_gusts_10m_max", "wind_gusts_10m_mean", "wind_speed_10m_mean", "wind_gusts_10m_min", "wind_speed_10m_min", "pressure_msl_min", "pressure_msl_max", "pressure_msl_mean", "snowfall_water_equivalent_sum", "relative_humidity_2m_max", "relative_humidity_2m_min", "relative_humidity_2m_mean", "et0_fao_evapotranspiration_sum", "dew_point_2m_min", "dew_point_2m_max", "dew_point_2m_mean", "cloud_cover_mean", "cloud_cover_max", "cloud_cover_min"]
FLOOD_VARS = ["river_discharge"]

# --------------------------------------------------------------------------------
# 2. DATA LOADER (Forecast Mode)
# --------------------------------------------------------------------------------

class ForecastLoader:
    def __init__(self, timezone="Asia/Bangkok"):
        self.timezone = timezone
        cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        self.client = openmeteo_requests.Client(session=retry_session)
        self.URLS = {
            "weather": "https://api.open-meteo.com/v1/forecast",
            "flood": "https://flood-api.open-meteo.com/v1/flood",
        }

    def _fetch_api(self, url, lat, lon, variables, past_days, forecast_days):
        full_params = {
            "latitude": lat, "longitude": lon, "daily": variables,
            "timezone": self.timezone, "past_days": past_days, "forecast_days": forecast_days
        }
        try:
            responses = self.client.weather_api(url, params=full_params)
            daily = responses[0].Daily()
            data = {"date": pd.date_range(
                start=pd.to_datetime(daily.Time(), unit="s", utc=True),
                end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=daily.Interval()),
                inclusive="left"
            )}
            for i, var in enumerate(variables):
                data[var] = daily.Variables(i).ValuesAsNumpy()
            df = pd.DataFrame(data)
            df['date'] = df['date'].dt.tz_convert(self.timezone).dt.normalize()
            return df
        except Exception as e:
            return pd.DataFrame()

    def get_forecast_data_multi(self, locations):
        all_data_frames = []
        PAST = 35
        FORECAST = 7
        
        for loc in locations:
            df_w = self._fetch_api(self.URLS["weather"], loc['lat'], loc['lon'], WEATHER_VARS, PAST, FORECAST)
            df_f = self._fetch_api(self.URLS["flood"], loc['lat'], loc['lon'], FLOOD_VARS, PAST, FORECAST)
            if not df_w.empty and not df_f.empty:
                merged = pd.merge(df_w, df_f, on="date", how="inner")
                merged['location'] = loc['name']
                all_data_frames.append(merged)
        
        if all_data_frames:
            return pd.concat(all_data_frames, ignore_index=True)
        return pd.DataFrame()

# --------------------------------------------------------------------------------
# 3. PREPROCESS (Feature Engineering)
# --------------------------------------------------------------------------------

def run_preprocess(df):
    if df.empty: return df
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])
    
    df = df.set_index('date').sort_index()
    
    # Fill NaN for soil columns if any (Important for inference stability)
    soil_cols = [c for c in df.columns if 'soil' in c]
    if soil_cols:
        df[soil_cols] = df[soil_cols].fillna(method='ffill').fillna(method='bfill')

    df_target = df[df['location'] == 'ChaoPhraya_Dam'].copy()
    df_upstream = df[df['location'] == 'NakhonSawan_Muang_Upstream'].copy()

    # Lag Features
    lag_features = ['river_discharge']
    lag_days = [1, 2]
    df_upstream_predictors = df_upstream[lag_features].rename(columns={'river_discharge': 'C2_discharge'})
    for lag in lag_days:
        lagged = df_upstream_predictors.shift(periods=lag).rename(columns={'C2_discharge': f'C2_discharge_lag{lag}'})
        df_target = df_target.merge(lagged, left_index=True, right_index=True, how='left')

    # Rolling & Temporal Features
    df_target['precip_rolling_7d'] = df_target['precipitation_sum'].rolling(window=7).sum() 
    df_target['precip_rolling_15d'] = df_target['precipitation_sum'].rolling(window=15).sum()
    df_target['soil_moisture_rolling_7d_avg'] = df_target['soil_moisture_0_to_100cm_mean'].rolling(window=7).mean()
    df_target['month'] = df_target.index.month

    df_target = df_target.drop(columns=['location'])
    return df_target

# --------------------------------------------------------------------------------
# 4. MAIN EXECUTION (Final Logic)
# --------------------------------------------------------------------------------
if __name__ == '__main__':
    # 1. Load Data
    print("--- 1. Loading Data ---")
    loader = ForecastLoader()
    df_raw = loader.get_forecast_data_multi(LOCATIONS)
    
    # 2. Preprocess
    print("--- 2. Preprocessing ---")
    df_processed = run_preprocess(df_raw)

    # 3. Filter Forecast Period
    today = pd.to_datetime(dt.date.today()).tz_localize('Asia/Bangkok').normalize()
    df_forecast = df_processed[df_processed.index >= today].copy()
    df_forecast = df_forecast.iloc[:7]

    if not df_forecast.empty:
        start_date_str = df_forecast.index.min().strftime('%Y-%m-%d')
        end_date_str = df_forecast.index.max().strftime('%Y-%m-%d')

        # ------------------------------------------------------------------
        # 4. Load Scaler & Align Features (แก้ปัญหา NaN และ Col หาย)
        # ------------------------------------------------------------------
        SCALER_PATH = "models/scaler.pkl" 

        if os.path.exists(SCALER_PATH):
            print(f"✅ Loading scaler from {SCALER_PATH}")
            scaler = joblib.load(SCALER_PATH)
            
            # 1. ดึงรายชื่อ Feature ที่ Scaler ต้องการ
            expected_features = scaler.feature_names_in_
            
            # 2. [Magic Fix] จัดเรียงคอลัมน์ + เติม 0
            # - fill_value=0 : ถ้าคอลัมน์ไหนขาด (เช่น Soil) ให้เติม 0
            df_aligned = df_forecast.reindex(columns=expected_features, fill_value=0)
            
            # 3. [Extra Safety] เติม 0 อีกครั้งเผื่อมี NaN หลงเหลือในคอลัมน์ที่มีอยู่แล้ว
            df_aligned = df_aligned.fillna(0)

            # 4. Transform
            X_scaled = scaler.transform(df_aligned)
            
            # 5. สร้าง DataFrame ผลลัพธ์
            df_result = pd.DataFrame(X_scaled, columns=expected_features)
            
            # ------------------------------------------------------------------
            # 5. Save
            # ------------------------------------------------------------------
            output_filename = f"data/inference_data/final_inference_data_{start_date_str}_{end_date_str}.csv"
            
            import os
            os.makedirs('data/inference_data', exist_ok=True)

            df_result.to_csv(output_filename, index=False)
            
            print(f"\n--- ✅ Process Complete ---")
            print(f"Data Range: {start_date_str} to {end_date_str}")
            print(f"Saved to: {output_filename}")
            
            # แสดงตัวอย่างข้อมูล 5 แถวแรก (จะเห็นว่าไม่มี ,, ว่างๆ แล้ว)
            print(df_result.head())

        else:
            print(f"❌ Error: Scaler not found at {SCALER_PATH}")
            exit()

    else:
        print("Error: No forecast data available.")

# --------------------------------------------------------------------------------
# def run_prediction():
#     # 1. Load Model
#     model = xgb.XGBClassifier()
#     model.load_model("models/xgboost_model.json")
#     scaler = joblib.load("models/scaler.pkl")
    
#     # 2. Fetch Data (Past + Future)
#     df = fetch_combined_data()
    
#     # 3. Create Lag Features (Shift ข้อมูลใน DataFrame เดียวกัน)
#     # ค่า Lag ของ "วันพรุ่งนี้" จะไปดึงมาจาก "Observed Data" ของ 5-7 วันที่แล้ว
#     # ค่า Lag ของ "อีก 7 วันข้างหน้า" จะไปดึงมาจาก "Forecast Data" ของ 2 วันข้างหน้า
#     df['precip_lag_5'] = df['precipitation_sum'].shift(5)
#     df['precip_lag_6'] = df['precipitation_sum'].shift(6)
#     df['precip_lag_7'] = df['precipitation_sum'].shift(7)
    
#     # 4. Filter เอาเฉพาะ "อนาคต 7 วัน" (วันนี้ + 1 ถึง วันนี้ + 7)
#     # ใช้ pd.Timestamp.now() เพื่อหาเส้นแบ่งเวลาปัจจุบัน
#     today = pd.Timestamp.now(tz='Asia/Bangkok').normalize()
#     future_df = df[df['date'] >= today].head(7).copy()
    
#     # Fill NaN (เผื่อ Lag วันแรกๆ ขาดหายไป)
#     future_df = future_df.fillna(0)
    
#     print(f"Predicting for dates: {future_df['date'].dt.date.values}")

#     # 5. Prepare & Predict
#     features = ['precipitation_sum', 'soil_moisture_0_to_100cm_mean', 'temperature_2m_mean',
#                 'precip_lag_5', 'precip_lag_6', 'precip_lag_7']
    
#     X = future_df[features]
#     X_scaled = scaler.transform(X)
    
#     predictions = model.predict(X_scaled)
#     probs = model.predict_proba(X_scaled)[:, 1]
    
#     # 6. Output JSON (List of Objects)
#     results = []
#     for date, pred, prob in zip(future_df['date'], predictions, probs):
#         results.append({
#             "date": date.strftime('%Y-%m-%d'),
#             "is_flood": int(pred),
#             "flood_probability": float(prob) * 100,
#             "risk_level": "High" if prob > 0.5 else "Low"
#         })
    
#     with open("prediction_results.json", "w") as f:
#         json.dump(results, f, indent=4)
#     print("7-day prediction saved to prediction_results.json")

# if __name__ == "__main__":
#     run_prediction()