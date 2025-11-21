import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# --- 1. Load Data ---
df = pd.read_csv('data/raw/raw_data.csv')

# Convert 'date' to datetime and set as index for time-series operations
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')

# --- 2. Separate DataFrames ---
df_target = df[df['location'] == 'ChaoPhraya_Dam'].copy()
df_upstream = df[df['location'] == 'NakhonSawan_Muang_Upstream'].copy()

# --- 3. Feature Engineering: Time-Lagged (C.2 Discharge) ---
lag_features = ['river_discharge']
lag_days = [1, 2]

df_upstream_predictors = df_upstream[lag_features].rename(columns={'river_discharge': 'C2_discharge'})

for lag in lag_days:
    lagged_data = df_upstream_predictors.shift(periods=lag)
    lagged_data = lagged_data.rename(columns={'C2_discharge': f'C2_discharge_lag{lag}'})
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
    (df_target['river_discharge'] >= threshold_flood),  # Class 2: Flood (>= 2700)
    (df_target['river_discharge'] >= threshold_risk) & (df_target['river_discharge'] < threshold_flood) # Class 1: Risk (2100 <= Discharge < 2700)
]
choices = [2, 1]  # 2 = high risk, 1 = medium risk (Default 0 = Normal)

df_target['flood_multiclass'] = np.select(conditions, choices, default=0)

# --- 6. Final Cleaning และ Save ---
df_target = df_target.drop(columns=['location'])

# ลบคอลัมน์ 'flood' (Binary) ที่ไม่ได้ใช้แล้วหากมี
if 'flood' in df_target.columns:
    df_target = df_target.drop(columns=['flood'])

# ลบแถวที่มีค่า NaN (จาก Lag/Rolling)
df_final_all_features_multiclass = df_target.dropna().copy()

# Save the final Multiclass DataFrame
# df_final_all_features_multiclass.to_csv('df_final_all_features_multiclass.csv')

print("--- 🏁 สรุปการสร้าง Features Multiclass ---")
print(f"จำนวนคอลัมน์รวม: {len(df_final_all_features_multiclass.columns)} คอลัมน์")
print(f"จำนวนแถวข้อมูลที่พร้อมใช้งาน: {len(df_final_all_features_multiclass)} แถว")
# print("DataFrame ถูกบันทึกเป็นไฟล์ 'df_final_all_features_multiclass.csv' เรียบร้อยแล้ว")

# ----------------------------------------------------------------------
#split_train_test_multiclass.py
# ----------------------------------------------------------------------

# --- 1. Load Data Artifacts (Multiclass File) ---
# ต้องโหลดไฟล์ multiclass ที่สร้างล่าสุด
# try:
#     df_final = pd.read_csv('df_final_all_features_multiclass.csv', index_col='date', parse_dates=True)
#     print("โหลดไฟล์ Multiclass สำเร็จ")
# except FileNotFoundError:
#     print("Error: ไม่พบไฟล์ 'df_final_all_features_multiclass.csv'")
#     print("โปรดตรวจสอบว่าได้รันโค้ด Preprocessing ขั้นตอนสุดท้ายเพื่อสร้างไฟล์นี้แล้ว")
#     exit()
df_final = df_final_all_features_multiclass.copy()

# --- 2. Define X (Features) and y (Target) ---
# X: คอลัมน์ทั้งหมด ยกเว้น 'river_discharge' และ 'flood_multiclass' (Target ใหม่)
features = [col for col in df_final.columns if col not in ['river_discharge', 'flood_multiclass']]
X = df_final[features]

# y: คอลัมน์ Target Multiclass
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

# --- 5. Save Artifacts (ใช้ชื่อไฟล์ Multiclass) ---
X_train_scaled_df.to_csv('data/preprocess_data/X_train_scaled.csv')
X_test_scaled_df.to_csv('data/preprocess_data/X_test_scaled.csv')
y_train.to_csv('data/preprocess_data/y_train.csv')
y_test.to_csv('data/preprocess_data/y_test.csv')

print("\n--- 🏁 Train/Test Split และ Scaling สำหรับ Multiclass เสร็จสมบูรณ์ ---")
print(f"จำนวนแถวข้อมูลรวม: {len(df_final)} แถว")
print(f"Train set: {len(X_train)} แถว ({X_train.index.min().strftime('%Y-%m-%d')} ถึง {X_train.index.max().strftime('%Y-%m-%d')})")
print(f"Test set:  {len(X_test)} แถว ({X_test.index.min().strftime('%Y-%m-%d')} ถึง {X_test.index.max().strftime('%Y-%m-%d')})")

print("\n📊 Class Distribution ในชุด Train:")
# แสดงสัดส่วน Multiclass ในชุด Train
print(y_train.value_counts().rename({0: 'Normal', 1: 'Risk', 2: 'Flood'}, inplace=False))

print("\nไฟล์ CSV 4 ไฟล์ Multiclass (Artifacts) ถูกบันทึกเรียบร้อยแล้ว")
