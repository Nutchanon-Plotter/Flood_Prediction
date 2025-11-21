import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os  # <--- 1. เพิ่ม import os

def process_and_split_data():
    # -------------------------------------------------------------------------
    # Setup Paths (จัดการ Path ให้ทำงานได้ทุกที่ ทั้ง Local และ GitHub Actions)
    # -------------------------------------------------------------------------
    # หาตำแหน่งไฟล์ script ปัจจุบัน (เช่น /project/src/preprocess.py)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # ถอยกลับไปหา Project Root (เช่น /project/)
    project_root = os.path.dirname(current_dir)
    
    # กำหนด Path ของไฟล์ Input และ Output Directory
    input_file_path = os.path.join(project_root, 'data', 'raw', 'raw_data.csv')
    output_dir = os.path.join(project_root, 'data', 'preprocess_data')
    
    # สร้างโฟลเดอร์ Output ถ้ายังไม่มี (สำคัญมาก!)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"📂 Loading data from: {input_file_path}")

    # --- 1. Load Data ---
    if not os.path.exists(input_file_path):
        raise FileNotFoundError(f"❌ ไม่พบไฟล์ข้อมูลดิบที่: {input_file_path}")

    df = pd.read_csv(input_file_path)

    # Convert 'date' to datetime and set as index for time-series operations
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')

    # --- 2. Separate DataFrames ---
    # ตรวจสอบว่ามีข้อมูลตาม location ที่ต้องการหรือไม่
    if 'location' not in df.columns:
         # กรณีข้อมูลอาจจะถูก filter มาแล้ว หรือไม่มี column นี้ ให้ข้ามหรือ handle error
         print("⚠️ Warning: 'location' column not found. Assuming data is ready or single location.")
         # ตรงนี้ต้องระวัง ถ้า data raw ไม่มี location logic นี้จะพัง
         # แต่สมมติว่า format ตรงกับ data_loader ที่ให้ไปก่อนหน้า
         pass

    df_target = df[df['location'] == 'ChaoPhraya_Dam'].copy()
    df_upstream = df[df['location'] == 'NakhonSawan_Muang_Upstream'].copy()
    
    if df_target.empty or df_upstream.empty:
        print(f"❌ Error: Data for target or upstream location is empty.")
        print(f"   Target rows: {len(df_target)}, Upstream rows: {len(df_upstream)}")
        # อาจจะ return หรือ exit ถ้าข้อมูลไม่ครบ
    
    # --- 3. Feature Engineering: Time-Lagged (C.2 Discharge) ---
    lag_features = ['river_discharge']
    lag_days = [1, 2]

    df_upstream_predictors = df_upstream[lag_features].rename(columns={'river_discharge': 'C2_discharge'})

    for lag in lag_days:
        lagged_data = df_upstream_predictors.shift(periods=lag)
        lagged_data = lagged_data.rename(columns={'C2_discharge': f'C2_discharge_lag{lag}'})
        # Join โดยใช้ Index (Date)
        df_target = df_target.merge(lagged_data, left_index=True, right_index=True, how='left')

    # --- 4. Feature Engineering: Rolling Window & Temporal ---
    df_target['precip_rolling_7d'] = df_target['precipitation_sum'].rolling(window=7).sum()
    df_target['precip_rolling_15d'] = df_target['precipitation_sum'].rolling(window=15).sum()
    df_target['soil_moisture_rolling_7d_avg'] = df_target['soil_moisture_0_to_100cm_mean'].rolling(window=7).mean()
    df_target['month'] = df_target.index.month

    # --- 5. Create Multiclass Target Variable (y) ---
    threshold_risk = 2100.0   # เกณฑ์สำหรับ medium Risk
    threshold_flood = 2700.0  # เกณฑ์สำหรับ high risk

    conditions = [
        (df_target['river_discharge'] >= threshold_flood),  # Class 2: Flood
        (df_target['river_discharge'] >= threshold_risk) & (df_target['river_discharge'] < threshold_flood) # Class 1: Risk
    ]
    choices = [2, 1]  # 2 = High Risk, 1 = Medium Risk (Default 0 = Normal)

    df_target['flood_multiclass'] = np.select(conditions, choices, default=0)

    # --- 6. Final Cleaning ---
    if 'location' in df_target.columns:
        df_target = df_target.drop(columns=['location'])

    # ลบคอลัมน์ 'flood' (Binary) เดิมทิ้ง
    if 'flood' in df_target.columns:
        df_target = df_target.drop(columns=['flood'])

    # ลบแถวที่มีค่า NaN (จาก Lag/Rolling)
    df_final = df_target.dropna().copy()

    print("--- 🏁 สรุปการสร้าง Features Multiclass ---")
    print(f"จำนวนคอลัมน์รวม: {len(df_final.columns)}")
    print(f"จำนวนแถวข้อมูลที่พร้อมใช้งาน: {len(df_final)}")

    # ----------------------------------------------------------------------
    # Split Train/Test & Save Artifacts
    # ----------------------------------------------------------------------

    # --- 2. Define X (Features) and y (Target) ---
    # X: คอลัมน์ทั้งหมด ยกเว้น 'river_discharge' (ค่าจริงของ target) และ 'flood_multiclass' (label)
    features = [col for col in df_final.columns if col not in ['river_discharge', 'flood_multiclass']]
    X = df_final[features]
    y = df_final['flood_multiclass']

    # --- 3. Time-Series Split (80% Train, 20% Test) ---
    split_ratio = 0.8
    split_index = int(len(X) * split_ratio)

    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    # --- 4. Feature Scaling (StandardScaler) ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert back to DataFrame
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=features, index=X_train.index)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=features, index=X_test.index)

    # --- 5. Save Artifacts (ใช้ Absolute Path) ---
    print(f"\n💾 Saving processed data to: {output_dir}")
    
    X_train_scaled_df.to_csv(os.path.join(output_dir, 'X_train_scaled.csv'))
    X_test_scaled_df.to_csv(os.path.join(output_dir, 'X_test_scaled.csv'))
    y_train.to_csv(os.path.join(output_dir, 'y_train.csv'))
    y_test.to_csv(os.path.join(output_dir, 'y_test.csv'))

    print("\n--- 🏁 Train/Test Split และ Scaling เสร็จสมบูรณ์ ---")
    print(f"Train set: {len(X_train)} แถว")
    print(f"Test set:  {len(X_test)} แถว")

    print("\n📊 Class Distribution ในชุด Train:")
    # แสดงสัดส่วน Multiclass ในชุด Train (ถ้ามี class ครบ)
    # ใช้ map เพื่อความปลอดภัยกรณี class ไม่ครบ
    label_map = {0: 'Normal', 1: 'Risk', 2: 'Flood'}
    dist = y_train.value_counts().sort_index()
    dist.index = dist.index.map(label_map)
    print(dist)

    print("\n✅ Saved 4 CSV files successfully.")

if __name__ == "__main__":
    process_and_split_data()