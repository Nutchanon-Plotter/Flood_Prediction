# src/monitor.py
import pandas as pd
import os
import sys
import joblib
import numpy as np
from evidently.test_suite import TestSuite
from evidently.tests import TestNumberOfDriftedColumns
import warnings

# ปิด Warning ที่ไม่จำเป็น
warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------------
# 1. CONFIG: 57 Features (Source of Truth)
# --------------------------------------------------------------------------------
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

# --------------------------------------------------------------------------------
# 2. PREPROCESS
# --------------------------------------------------------------------------------
def preprocess_current_data(df):
    if df.empty: return df
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()
    
    # Fill NaN
    soil_cols = [c for c in df.columns if 'soil' in c]
    if soil_cols:
        df[soil_cols] = df[soil_cols].ffill().bfill()

    # Fix Columns
    if 'wind_direction_10m_dominant' in df.columns:
        df['winddirection_10m_dominant'] = df['wind_direction_10m_dominant']

    df_target = df[df['location'] == 'ChaoPhraya_Dam'].copy()
    df_upstream = df[df['location'] == 'NakhonSawan_Muang_Upstream'].copy()
    
    if df_target.empty: return pd.DataFrame()

    # Lag Features
    lag_features = ['river_discharge']
    lag_days = [1, 2]
    if not df_upstream.empty:
        df_upstream_predictors = df_upstream[lag_features].rename(columns={'river_discharge': 'C2_discharge'})
        for lag in lag_days:
            lagged = df_upstream_predictors.shift(periods=lag).rename(columns={'C2_discharge': f'C2_discharge_lag{lag}'})
            df_target = df_target.merge(lagged, left_index=True, right_index=True, how='left')
    else:
        # ปล่อยว่างไว้ (เดี๋ยวไปเติม Mean ทีหลัง)
        pass 

    # Rolling & Temporal
    df_target['precip_rolling_7d'] = df_target['precipitation_sum'].rolling(window=7).sum() 
    df_target['precip_rolling_15d'] = df_target['precipitation_sum'].rolling(window=15).sum()
    df_target['soil_moisture_rolling_7d_avg'] = df_target['soil_moisture_0_to_100cm_mean'].rolling(window=7).mean()
    df_target['month'] = df_target.index.month

    if 'location' in df_target.columns: df_target = df_target.drop(columns=['location'])
    
    return df_target

# --------------------------------------------------------------------------------
# 3. MONITORING LOGIC
# --------------------------------------------------------------------------------
def monitor_drift():
    print("🔎 Starting Data Drift Monitor (Relaxed Mode)...")
    
    # Setup Paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    ref_path = os.path.join(project_root, 'data', 'preprocess_data', 'X_train_scaled.csv')
    current_path = os.path.join(project_root, 'data', 'raw', 'raw_data.csv')
    scaler_path = os.path.join(project_root, 'models', 'scaler.pkl')
    report_path = os.path.join(project_root, 'models', 'drift_report.html')

    # 1. Load Reference
    if not os.path.exists(ref_path):
        print("⚠️ Reference data missing. Force Retrain.")
        set_github_output(True) 
        return

    try:
        reference_data = pd.read_csv(ref_path)[EXPECTED_FEATURES]
        reference_means = reference_data.mean()
    except KeyError as e:
        print(f"❌ Reference schema mismatch: {e}")
        set_github_output(True)
        return

    # 2. Load Current
    if not os.path.exists(current_path):
        print("⚠️ Current data missing.")
        set_github_output(True)
        return

    raw_data = pd.read_csv(current_path)
    current_processed = preprocess_current_data(raw_data)
    
    if current_processed.empty:
        print("❌ Processed data empty.")
        set_github_output(True)
        return

    # 3. Smart Alignment & Fillna
    current_aligned = current_processed.reindex(columns=EXPECTED_FEATURES)
    current_aligned = current_aligned.fillna(reference_means).fillna(0)
    
    # 4. Scale
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        current_scaled_np = scaler.transform(current_aligned)
        current_final = pd.DataFrame(current_scaled_np, columns=EXPECTED_FEATURES)
    else:
        print("⚠️ Scaler not found.")
        current_final = current_aligned

    current_window = current_final.tail(50)

    # 5. Smart Filter: ตัด Feature ที่นิ่งสนิทออก (Constant Filter)
    valid_features = []
    for col in EXPECTED_FEATURES:
        # ถ้าค่าความแปรปรวน > 0 แสดงว่าข้อมูลมีการเคลื่อนไหวจริง
        if current_window[col].std() > 0.0001:
            valid_features.append(col)
    
    if len(valid_features) == 0:
        print("⚠️ No valid features (Data is static). Skipping check.")
        set_github_output(False)
        return

    ref_valid = reference_data[valid_features]
    curr_valid = current_window[valid_features]

    print(f"   Checking {len(valid_features)}/{len(EXPECTED_FEATURES)} features.")

    # 6. Run Evidently with Relaxed Threshold
    # lt=20 : ยอมให้มี Drift ได้ไม่เกิน 20 Features (จาก 57) ถึงจะแจ้งเตือน
    data_drift_tests = TestSuite(tests=[
        TestNumberOfDriftedColumns(lt=20) 
    ])
    
    data_drift_tests.run(reference_data=ref_valid, current_data=curr_valid)
    data_drift_tests.save_html(report_path)
    print(f"✅ Drift Report saved: {report_path}")
    
    # 7. Evaluate Result
    result = data_drift_tests.as_dict()
    try:
        is_drift = not result["tests"][0]["parameters"]["condition"]["pass"]
        drifted_count = result["tests"][0]["parameters"]["value"]
        print(f"📊 Drifted Features: {drifted_count} (Threshold < 20 to Pass)")
    except Exception as e:
        print(f"⚠️ Error parsing result: {e}")
        is_drift = True
    
    print(f"🚨 FINAL DRIFT STATUS: {is_drift}")
    set_github_output(is_drift)

def set_github_output(is_drift):
    if 'GITHUB_OUTPUT' in os.environ:
        with open(os.environ['GITHUB_OUTPUT'], 'a') as fh:
            fh.write(f"drift_detected={str(is_drift).lower()}\n")
    else:
        print(f"   (Local) drift_detected = {is_drift}")

if __name__ == "__main__":
    monitor_drift()