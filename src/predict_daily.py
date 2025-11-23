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
import sys
import warnings

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------------------------------------

LOCATIONS = [
    {"name": "ChaoPhraya_Dam", "lat": 15.159213709346405, "lon": 100.17985248529882}, 
    {"name": "NakhonSawan_Muang_Upstream", "lat": 15.700409309316225, "lon": 100.14120663110944},
]

WEATHER_VARS = ["temperature_2m_min", "apparent_temperature_mean", "apparent_temperature_max", "apparent_temperature_min", "temperature_2m_max", "temperature_2m_mean", "daylight_duration", "sunshine_duration", "precipitation_sum", "rain_sum", "snowfall_sum", "precipitation_hours", "wind_speed_10m_max", "wind_direction_10m_dominant", "shortwave_radiation_sum", "et0_fao_evapotranspiration", "soil_moisture_0_to_100cm_mean", "soil_moisture_0_to_7cm_mean", "soil_moisture_28_to_100cm_mean", "soil_moisture_7_to_28cm_mean", "soil_temperature_0_to_100cm_mean", "soil_temperature_0_to_7cm_mean", "soil_temperature_28_to_100cm_mean", "soil_temperature_7_to_28cm_mean", "wet_bulb_temperature_2m_mean", "wet_bulb_temperature_2m_max", "wet_bulb_temperature_2m_min", "vapour_pressure_deficit_max", "surface_pressure_mean", "surface_pressure_max", "surface_pressure_min", "winddirection_10m_dominant", "wind_gusts_10m_max", "wind_gusts_10m_mean", "wind_speed_10m_mean", "wind_gusts_10m_min", "wind_speed_10m_min", "pressure_msl_min", "pressure_msl_max", "pressure_msl_mean", "snowfall_water_equivalent_sum", "relative_humidity_2m_max", "relative_humidity_2m_min", "relative_humidity_2m_mean", "et0_fao_evapotranspiration_sum", "dew_point_2m_min", "dew_point_2m_max", "dew_point_2m_mean", "cloud_cover_mean", "cloud_cover_max", "cloud_cover_min"]
FLOOD_VARS = ["river_discharge"]

# --------------------------------------------------------------------------------
# 2. DATA LOADER
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
# 3. PREPROCESS
# --------------------------------------------------------------------------------

def run_preprocess(df):
    if df.empty: return df
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])
    
    df = df.set_index('date').sort_index()
    
    soil_cols = [c for c in df.columns if 'soil' in c]
    if soil_cols:
        df[soil_cols] = df[soil_cols].fillna(method='ffill').fillna(method='bfill')

    df_target = df[df['location'] == 'ChaoPhraya_Dam'].copy()
    df_upstream = df[df['location'] == 'NakhonSawan_Muang_Upstream'].copy()

    lag_features = ['river_discharge']
    lag_days = [1, 2]
    df_upstream_predictors = df_upstream[lag_features].rename(columns={'river_discharge': 'C2_discharge'})
    for lag in lag_days:
        lagged = df_upstream_predictors.shift(periods=lag).rename(columns={'C2_discharge': f'C2_discharge_lag{lag}'})
        df_target = df_target.merge(lagged, left_index=True, right_index=True, how='left')

    df_target['precip_rolling_7d'] = df_target['precipitation_sum'].rolling(window=7).sum() 
    df_target['precip_rolling_15d'] = df_target['precipitation_sum'].rolling(window=15).sum()
    df_target['soil_moisture_rolling_7d_avg'] = df_target['soil_moisture_0_to_100cm_mean'].rolling(window=7).mean()
    df_target['month'] = df_target.index.month

    df_target = df_target.drop(columns=['location'])
    return df_target

# --------------------------------------------------------------------------------
# 4. MODEL LOADER
# --------------------------------------------------------------------------------

def load_production_model_verified():
    print("\n--- 🔍 Model Loading Verification ---")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    models_dir = os.path.join(project_root, 'models')
    filename_path = os.path.join(models_dir, "model_filename.txt")
    
    if not os.path.exists(filename_path):
        print(f"❌ Critical Error: Metadata file not found at {filename_path}")
        sys.exit(1)
        
    with open(filename_path, "r") as f:
        target_filename = f.read().strip()
    
    print(f"📋 Target Model File: '{target_filename}'")
    
    model_path = os.path.join(models_dir, target_filename)
    if not os.path.exists(model_path):
        print(f"❌ Critical Error: Model file missing at {model_path}")
        sys.exit(1)
        
    print(f"📂 Attempting to load from: {model_path}")
    try:
        if target_filename.endswith(".json"):
            model = xgb.XGBClassifier()
            model.load_model(model_path)
            model_type = "XGBoost"
        else:
            model = joblib.load(model_path)
            model_type = "Scikit-Learn"
            
        print(f"✅ Success: Model loaded correctly.")
        print(f"🤖 Model Type Identified: {model_type}")
        print("-------------------------------------\n")
        
        return model, models_dir, model_type
        
    except Exception as e:
        print(f"❌ Loading Failed: {e}")
        sys.exit(1)

# --------------------------------------------------------------------------------
# 5. MAIN EXECUTION
# --------------------------------------------------------------------------------
if __name__ == '__main__':
    print("🔮 Starting Daily Prediction Pipeline...")

    # 1. Load Verified Model
    model, models_dir, model_type = load_production_model_verified()
    
    # 2. Load Scaler
    SCALER_PATH = os.path.join(models_dir, "scaler.pkl")
    if not os.path.exists(SCALER_PATH):
        print(f"❌ Error: Scaler not found at {SCALER_PATH}")
        sys.exit(1)
    
    scaler = joblib.load(SCALER_PATH)
    print("✅ Scaler loaded successfully.")

    # 3. Load Data & Preprocess
    print("--- Loading & Preprocessing Data ---")
    loader = ForecastLoader()
    df_raw = loader.get_forecast_data_multi(LOCATIONS)
    df_processed = run_preprocess(df_raw)

    # 4. Filter Next 7 Days
    today = pd.to_datetime(dt.date.today()).tz_localize('Asia/Bangkok').normalize()
    df_forecast = df_processed[df_processed.index >= today].copy()
    df_forecast = df_forecast.iloc[:7]

    if not df_forecast.empty:
        # --- Handle Shape Mismatch ---
        scaler_features = scaler.feature_names_in_
        df_aligned = df_forecast.reindex(columns=scaler_features, fill_value=0).fillna(0)
        X_scaled_all = scaler.transform(df_aligned)
        X_scaled_df = pd.DataFrame(X_scaled_all, columns=scaler_features)
        
        required_features = []
        if model_type == "XGBoost":
            try:
                required_features = model.get_booster().feature_names
            except: pass
        else:
            if hasattr(model, "feature_names_in_"):
                required_features = model.feature_names_in_
        
        if required_features is None or len(required_features) == 0:
            print("⚠️ Warning: Could not detect model features. Using top 10 default.")
            required_features = ['C2_discharge_lag1', 'soil_moisture_0_to_100cm_mean', 'soil_temperature_7_to_28cm_mean', 'soil_moisture_7_to_28cm_mean', 'precip_rolling_15d', 'soil_temperature_0_to_100cm_mean', 'soil_moisture_28_to_100cm_mean', 'daylight_duration', 'wet_bulb_temperature_2m_mean', 'dew_point_2m_mean']

        print(f"🎯 Model expects {len(required_features)} features.")
        X_final = X_scaled_df[required_features]
        
        # 6. Predict
        print("--- Running Inference ---")
        predictions = model.predict(X_final)
        
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_final)
            if probs.shape[1] == 3:
                flood_probs = probs[:, 2] * 100
            else:
                flood_probs = probs.max(axis=1) * 100
        else:
            flood_probs = [0] * len(predictions)

        # 7. Output Results
        results = []
        status_map = {0: "Normal", 1: "Risk", 2: "Flood"}
        
        for date, pred, prob in zip(df_forecast.index, predictions, flood_probs):
            status_text = status_map.get(int(pred), "Unknown")
            results.append({
                "date": date.strftime('%Y-%m-%d'),
                "status_code": int(pred),
                "status_text": status_text,
                "flood_probability": float(prob),
                "risk_level": "High" if pred == 2 else ("Medium" if pred == 1 else "Low")
            })
        
        # --- 8. Save to 'predict_result' Directory (ส่วนที่แก้ไข) ---
        base_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(base_dir)
        
        # สร้างโฟลเดอร์ predict_result ถ้ายังไม่มี
        output_dir = os.path.join(project_root, "predict_result")
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, "prediction_results.json")
        
        with open(output_path, "w") as f:
            json.dump(results, f, indent=4)
            
        print(f"✅ Prediction saved to: {output_path}")
        print(json.dumps(results, indent=2))

    else:
        print("❌ Error: No forecast data available.")
        sys.exit(1)