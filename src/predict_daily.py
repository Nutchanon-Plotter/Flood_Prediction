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

# 57 Features (ตาม Scaler)
EXPECTED_FEATURES = [
    "temperature_2m_min", "apparent_temperature_mean", "apparent_temperature_max", "apparent_temperature_min",
    "temperature_2m_max", "temperature_2m_mean", "daylight_duration", "sunshine_duration",
    "precipitation_sum", "rain_sum", "snowfall_sum", "precipitation_hours",
    "wind_speed_10m_max", "wind_direction_10m_dominant", "shortwave_radiation_sum", "et0_fao_evapotranspiration",
    "soil_moisture_0_to_100cm_mean", "soil_moisture_0_to_7cm_mean", "soil_moisture_28_to_100cm_mean", "soil_moisture_7_to_28cm_mean",
    "soil_temperature_0_to_100cm_mean", "soil_temperature_0_to_7cm_mean", "soil_temperature_28_to_100cm_mean", "soil_temperature_7_to_28cm_mean",
    "wet_bulb_temperature_2m_mean", "wet_bulb_temperature_2m_max", "wet_bulb_temperature_2m_min", "vapour_pressure_deficit_max",
    "surface_pressure_mean", "surface_pressure_max", "surface_pressure_min", "winddirection_10m_dominant",
    "wind_gusts_10m_max", "wind_gusts_10m_mean", "wind_speed_10m_mean", "wind_gusts_10m_min", "wind_speed_10m_min",
    "pressure_msl_min", "pressure_msl_max", "pressure_msl_mean", "snowfall_water_equivalent_sum",
    "relative_humidity_2m_max", "relative_humidity_2m_min", "relative_humidity_2m_mean", "et0_fao_evapotranspiration_sum",
    "dew_point_2m_min", "dew_point_2m_max", "dew_point_2m_mean", "cloud_cover_mean", "cloud_cover_max", "cloud_cover_min",
    "C2_discharge_lag1", "C2_discharge_lag2", "precip_rolling_7d", "precip_rolling_15d",
    "soil_moisture_rolling_7d_avg", "month"
]

WEATHER_VARS = [
    "temperature_2m_min", "apparent_temperature_mean", "apparent_temperature_max", "apparent_temperature_min", 
    "temperature_2m_max", "temperature_2m_mean", "daylight_duration", "sunshine_duration", 
    "precipitation_sum", "rain_sum", "snowfall_sum", "precipitation_hours", 
    "wind_speed_10m_max", "wind_direction_10m_dominant", "shortwave_radiation_sum", "et0_fao_evapotranspiration", 
    "soil_moisture_0_to_100cm_mean", "soil_moisture_0_to_7cm_mean", "soil_moisture_28_to_100cm_mean", "soil_moisture_7_to_28cm_mean", 
    "soil_temperature_0_to_100cm_mean", "soil_temperature_0_to_7cm_mean", "soil_temperature_28_to_100cm_mean", "soil_temperature_7_to_28cm_mean", 
    "wet_bulb_temperature_2m_mean", "wet_bulb_temperature_2m_max", "wet_bulb_temperature_2m_min", "vapour_pressure_deficit_max", 
    "surface_pressure_mean", "surface_pressure_max", "surface_pressure_min", 
    "wind_gusts_10m_max", "wind_gusts_10m_mean", "wind_speed_10m_mean", "wind_gusts_10m_min", "wind_speed_10m_min", 
    "pressure_msl_min", "pressure_msl_max", "pressure_msl_mean", "snowfall_water_equivalent_sum", 
    "relative_humidity_2m_max", "relative_humidity_2m_min", "relative_humidity_2m_mean", "et0_fao_evapotranspiration_sum", 
    "dew_point_2m_min", "dew_point_2m_max", "dew_point_2m_mean", "cloud_cover_mean", "cloud_cover_max", "cloud_cover_min"
]
FLOOD_VARS = ["river_discharge"]

# --------------------------------------------------------------------------------
# 2. DATA LOADER (Optimized: History from File + Future from API)
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

    def _fetch_api(self, url, lat, lon, variables, forecast_days):
        full_params = {
            "latitude": lat, "longitude": lon, "daily": variables,
            "timezone": self.timezone, 
            "past_days": 0, # ไม่ดึงย้อนหลังแล้ว (ใช้ไฟล์ csv แทน)
            "forecast_days": forecast_days
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
            print(f"⚠️ API Error: {e}")
            return pd.DataFrame()

    def get_forecast_data_multi(self, locations):
        all_data_frames = []
        FORECAST = 7
        
        print(f"   📡 Fetching Future {FORECAST} days from API...")
        
        for loc in locations:
            df_w = self._fetch_api(self.URLS["weather"], loc['lat'], loc['lon'], WEATHER_VARS, FORECAST)
            df_f = self._fetch_api(self.URLS["flood"], loc['lat'], loc['lon'], FLOOD_VARS, FORECAST)
            
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
    
    # Fix Columns
    if 'wind_direction_10m_dominant' in df.columns:
        df['winddirection_10m_dominant'] = df['wind_direction_10m_dominant']

    # Fill NaN
    soil_cols = [c for c in df.columns if 'soil' in c]
    if soil_cols:
        df[soil_cols] = df[soil_cols].interpolate(method='time').ffill().bfill()

    # Split Location
    target_name = LOCATIONS[0]['name'] # ChaoPhraya_Dam
    upstream_name = LOCATIONS[1]['name'] # NakhonSawan
    
    df_target = df[df['location'] == target_name].copy()
    df_upstream = df[df['location'] == upstream_name].copy()

    # Lag Features
    lag_features = ['river_discharge']
    lag_days = [1, 2]
    if not df_upstream.empty:
        df_upstream_predictors = df_upstream[lag_features].rename(columns={'river_discharge': 'C2_discharge'})
        for lag in lag_days:
            lagged = df_upstream_predictors.shift(periods=lag).rename(columns={'C2_discharge': f'C2_discharge_lag{lag}'})
            df_target = df_target.merge(lagged, left_index=True, right_index=True, how='left')
    else:
        # Fallback
        for lag in lag_days:
            df_target[f'C2_discharge_lag{lag}'] = 0

    # Rolling Features
    df_target['precip_rolling_7d'] = df_target['precipitation_sum'].rolling(window=7).sum() 
    df_target['precip_rolling_15d'] = df_target['precipitation_sum'].rolling(window=15).sum()
    df_target['soil_moisture_rolling_7d_avg'] = df_target['soil_moisture_0_to_100cm_mean'].rolling(window=7).mean()
    df_target['month'] = df_target.index.month

    if 'location' in df_target.columns: df_target = df_target.drop(columns=['location'])
    
    # Clean NaN created by rolling/lag
    df_target = df_target.ffill().bfill().fillna(0)
    
    return df_target

# --------------------------------------------------------------------------------
# 4. MODEL LOADER
# --------------------------------------------------------------------------------

def load_production_model_verified():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    models_dir = os.path.join(project_root, 'models')
    filename_path = os.path.join(models_dir, "model_filename.txt")
    
    if not os.path.exists(filename_path):
        print(f"❌ Error: Metadata not found. Run training first.")
        sys.exit(1)
        
    with open(filename_path, "r") as f: target_filename = f.read().strip()
    model_path = os.path.join(models_dir, target_filename)
    
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file missing: {model_path}")
        sys.exit(1)
        
    try:
        if target_filename.endswith(".json"):
            model = xgb.XGBClassifier(); model.load_model(model_path); model_type = "XGBoost"
        else:
            model = joblib.load(model_path); model_type = "Scikit-Learn"
        return model, models_dir, model_type
    except Exception as e:
        print(f"❌ Load failed: {e}")
        sys.exit(1)

# --------------------------------------------------------------------------------
# 5. MAIN EXECUTION
# --------------------------------------------------------------------------------
if __name__ == '__main__':
    print("🔮 Starting Optimized Prediction Pipeline...")

    # 1. Load Model
    model, models_dir, model_type = load_production_model_verified()
    scaler_path = os.path.join(models_dir, "scaler.pkl")
    if not os.path.exists(scaler_path):
        print(f"❌ Error: Scaler missing.")
        sys.exit(1)
    scaler = joblib.load(scaler_path)
    print("✅ Model & Scaler loaded.")

    # 2. Load Data (Stitching Strategy)
    # A. Load History form CSV
    print("--- 📂 Loading Historical Data (Local CSV) ---")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    raw_path = os.path.join(project_root, 'data', 'raw', 'raw_data.csv')
    
    if not os.path.exists(raw_path):
        print("❌ Raw data file not found.")
        sys.exit(1)
        
    df_history = pd.read_csv(raw_path)
    df_history['date'] = pd.to_datetime(df_history['date'])
    # เอาแค่ 60 วันล่าสุดพอก็ได้ (ลดภาระการคำนวณ)
    df_history = df_history.sort_values('date').tail(60 * len(LOCATIONS)) 
    print(f"   Loaded {len(df_history)} rows from history.")

    # B. Fetch Future from API
    loader = ForecastLoader()
    df_future = loader.get_forecast_data_multi(LOCATIONS)
    
    if df_future.empty:
        print("❌ API returned no data.")
        sys.exit(1)

    # C. Combine
    df_combined = pd.concat([df_history, df_future], ignore_index=True)
    df_combined = df_combined.drop_duplicates(subset=['date', 'location'], keep='last')
    
    # 3. Preprocess
    print("--- ⚙️ Processing Features ---")
    df_processed = run_preprocess(df_combined)

    # 4. Slice Next 7 Days
    today = pd.to_datetime(dt.date.today()).tz_localize('Asia/Bangkok').normalize()
    df_forecast = df_processed[df_processed.index >= today].copy()
    df_forecast = df_forecast.iloc[:7]

    if not df_forecast.empty:
        # 5. Align & Scale
        df_aligned = df_forecast.reindex(columns=EXPECTED_FEATURES, fill_value=0)
        X_scaled_all = scaler.transform(df_aligned)
        X_scaled_df = pd.DataFrame(X_scaled_all, columns=EXPECTED_FEATURES)
        
        # 6. Predict
        required_features = []
        if model_type == "XGBoost":
            try: required_features = model.get_booster().feature_names
            except: pass
        else:
            if hasattr(model, "feature_names_in_"): required_features = model.feature_names_in_
        
        if not required_features:
            required_features = EXPECTED_FEATURES[:10]

        X_final = X_scaled_df[required_features]
        predictions = model.predict(X_final)
        
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_final)
            if probs.shape[1] == 3: flood_probs = probs[:, 2] * 100
            else: flood_probs = probs.max(axis=1) * 100
        else:
            flood_probs = [0] * len(predictions)

        # 7. Output
        results = []
        status_map = {0: "Normal", 1: "Risk", 2: "Flood"}
        target_loc_name = LOCATIONS[0]['name']
        
        for date, pred, prob in zip(df_forecast.index, predictions, flood_probs):
            results.append({
                "date": date.strftime('%Y-%m-%d'),
                "location": target_loc_name,
                "status_code": int(pred),
                "status_text": status_map.get(int(pred), "Unknown"),
                "flood_probability": float(prob),
                "risk_level": "High" if pred == 2 else ("Medium" if pred == 1 else "Low")
            })
        
        # SAVE
        output_dir_1 = os.path.join(project_root, "predict_result")
        os.makedirs(output_dir_1, exist_ok=True)
        output_path_1 = os.path.join(output_dir_1, "prediction_results.json")
        with open(output_path_1, "w") as f: json.dump(results, f, indent=4)
        
        output_dir_2 = os.path.join(project_root, "web", "frontend", "public")
        os.makedirs(output_dir_2, exist_ok=True)
        output_path_2 = os.path.join(output_dir_2, "prediction_results.json")
        with open(output_path_2, "w") as f: json.dump(results, f, indent=4)

        print(f"✅ Prediction saved to: {output_path_1}")
        print(json.dumps(results, indent=2))

    else:
        print("❌ Error: Forecast slice is empty.")
        sys.exit(1)