# src/preprocess.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import joblib

def process_and_split_data():
    # -------------------------------------------------------------------------
    # Setup Paths
    # -------------------------------------------------------------------------
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    input_file_path = os.path.join(project_root, 'data', 'raw', 'raw_data.csv')
    output_dir = os.path.join(project_root, 'data', 'preprocess_data')
    models_dir = os.path.join(project_root, 'models')  # <--- เพิ่ม path สำหรับ models
    
    # สร้างโฟลเดอร์ถ้ายังไม่มี
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    print(f"📂 Loading data from: {input_file_path}")

    # --- 1. Load Data ---
    if not os.path.exists(input_file_path):
        raise FileNotFoundError(f"❌ ไม่พบไฟล์ข้อมูลดิบที่: {input_file_path}")

    df = pd.read_csv(input_file_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')

    # --- 2. Feature Engineering ---
    if 'location' not in df.columns:
        print("⚠️ Warning: 'location' column not found.")
    
    # แยก DataFrame ตาม Location
    df_target = df[df['location'] == 'ChaoPhraya_Dam'].copy()
    df_upstream = df[df['location'] == 'NakhonSawan_Muang_Upstream'].copy()
    
    if df_target.empty or df_upstream.empty:
        print(f"❌ Error: Data for target or upstream location is empty.")
        exit(1)
    
    # Create Lag Features
    lag_features = ['river_discharge']
    lag_days = [1, 2]
    df_upstream_predictors = df_upstream[lag_features].rename(columns={'river_discharge': 'C2_discharge'})

    for lag in lag_days:
        lagged_data = df_upstream_predictors.shift(periods=lag)
        lagged_data = lagged_data.rename(columns={'C2_discharge': f'C2_discharge_lag{lag}'})
        df_target = df_target.merge(lagged_data, left_index=True, right_index=True, how='left')

    # Rolling Features
    df_target['precip_rolling_7d'] = df_target['precipitation_sum'].rolling(window=7).sum()
    df_target['precip_rolling_15d'] = df_target['precipitation_sum'].rolling(window=15).sum()
    df_target['soil_moisture_rolling_7d_avg'] = df_target['soil_moisture_0_to_100cm_mean'].rolling(window=7).mean()
    df_target['month'] = df_target.index.month

    # Create Target (Multiclass)
    threshold_risk = 2100.0
    threshold_flood = 2700.0
    conditions = [
        (df_target['river_discharge'] >= threshold_flood),
        (df_target['river_discharge'] >= threshold_risk) & (df_target['river_discharge'] < threshold_flood)
    ]
    choices = [2, 1]
    df_target['flood_multiclass'] = np.select(conditions, choices, default=0)

    # Clean up
    if 'location' in df_target.columns: df_target = df_target.drop(columns=['location'])
    if 'flood' in df_target.columns: df_target = df_target.drop(columns=['flood'])

    df_final = df_target.dropna().copy()

    print("--- 🏁 Preprocessing Summary ---")
    print(f"Total Columns: {len(df_final.columns)}")
    print(f"Total Rows: {len(df_final)}")

    # ----------------------------------------------------------------------
    # Split Train/Test & Save Artifacts
    # ----------------------------------------------------------------------
    features = [col for col in df_final.columns if col not in ['river_discharge', 'flood_multiclass']]
    X = df_final[features]
    y = df_final['flood_multiclass']

    split_index = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save Scaler (สำคัญมาก!)
    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"\n✅ Scaler saved to: {scaler_path}")

    # Save Data CSVs
    pd.DataFrame(X_train_scaled, columns=features, index=X_train.index).to_csv(os.path.join(output_dir, 'X_train_scaled.csv'))
    pd.DataFrame(X_test_scaled, columns=features, index=X_test.index).to_csv(os.path.join(output_dir, 'X_test_scaled.csv'))
    y_train.to_csv(os.path.join(output_dir, 'y_train.csv'))
    y_test.to_csv(os.path.join(output_dir, 'y_test.csv'))

    print("\n✅ Data artifacts saved successfully.")

if __name__ == "__main__":
    process_and_split_data()