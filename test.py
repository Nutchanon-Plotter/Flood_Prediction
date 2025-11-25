import pandas as pd
import os
import numpy as np

# --- 1. SETUP & CONFIG ---
# รายชื่อ 57 Features ที่เราคาดหวัง (Source of Truth)
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

def preprocess_for_debug(df):
    """Logic เดียวกับ monitor.py เพื่อจำลองการสร้าง Feature"""
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()
    
    # Fix Column Names
    if 'wind_direction_10m_dominant' in df.columns:
        df['winddirection_10m_dominant'] = df['wind_direction_10m_dominant']

    df_target = df[df['location'] == 'ChaoPhraya_Dam'].copy()
    df_upstream = df[df['location'] == 'NakhonSawan_Muang_Upstream'].copy()

    # Lag Features
    lag_features = ['river_discharge']
    lag_days = [1, 2]
    
    if not df_upstream.empty:
        df_upstream_predictors = df_upstream[lag_features].rename(columns={'river_discharge': 'C2_discharge'})
        for lag in lag_days:
            lagged = df_upstream_predictors.shift(periods=lag).rename(columns={'C2_discharge': f'C2_discharge_lag{lag}'})
            df_target = df_target.merge(lagged, left_index=True, right_index=True, how='left')
    
    # Rolling
    df_target['precip_rolling_7d'] = df_target['precipitation_sum'].rolling(window=7).sum() 
    df_target['precip_rolling_15d'] = df_target['precipitation_sum'].rolling(window=15).sum()
    df_target['soil_moisture_rolling_7d_avg'] = df_target['soil_moisture_0_to_100cm_mean'].rolling(window=7).mean()
    df_target['month'] = df_target.index.month

    if 'location' in df_target.columns: df_target = df_target.drop(columns=['location'])
    
    return df_target

# --- 2. MAIN CHECK ---
def check_columns():
    print("🕵️‍♂️ CHECKING MISSING COLUMNS...")
    
    # Load Files
    try:
        raw_df = pd.read_csv('data/raw/raw_data.csv')
        print(f"✅ Raw Data Loaded: {raw_df.shape}")
    except FileNotFoundError:
        print("❌ Error: data/raw/raw_data.csv not found.")
        return

    # Process Raw Data to get Current Features
    current_processed = preprocess_for_debug(raw_df)
    current_cols = set(current_processed.columns)
    expected_cols = set(EXPECTED_FEATURES)

    # Find Missing
    missing = expected_cols - current_cols
    extra = current_cols - expected_cols

    print("\n" + "="*40)
    print(f"🚨 MISSING COLUMNS ({len(missing)}):")
    print("="*40)
    if missing:
        for col in sorted(missing):
            print(f" - {col}")
    else:
        print("✅ None! All features are present.")

    print("\n" + "="*40)
    print(f"⚠️ EXTRA COLUMNS (Not in Reference):")
    print("="*40)
    if extra:
        for col in sorted(extra):
            print(f" + {col}")
    else:
        print("Clean.")

if __name__ == "__main__":
    check_columns()