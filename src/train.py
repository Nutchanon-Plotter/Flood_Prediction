# src/train.py
import mlflow
import mlflow.xgboost
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
from preprocess import preprocess_data
import pandas as pd
import os
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

def calculate_multiclass_weights():
    # --- 1. Load Data Artifacts (ใช้ชื่อไฟล์ Multiclass ที่ถูกต้อง) ---
    try:
        # โหลด Target Multiclass จากชุด Train
        y_train = pd.read_csv('y_train_multiclass_final.csv', index_col=0).squeeze()
        print("✅ โหลดไฟล์ y_train_multiclass_final.csv สำเร็จ")
    except FileNotFoundError:
        print("❌ Error: ไม่พบไฟล์ y_train_multiclass_final.csv")
        print("โปรดตรวจสอบว่าได้รันโค้ด Train/Test Split สำหรับ Multiclass เสร็จสมบูรณ์แล้ว")
        exit()


    # --- 2. Calculate Multiclass Class Weights (Inverse Frequency) ---
    # ใช้วิธีคำนวณน้ำหนักแบบ Inverse Frequency ซึ่งเป็นมาตรฐานสำหรับ Multiclass Imbalance
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)

    # สร้าง Dictionary เพื่อแสดงผล
    multiclass_weights = dict(zip(classes, weights))

    print("\n--- 📊 Multiclass Class Weights สำหรับการแก้ Imbalance ---")
    print(f"Total Samples (Train): {len(y_train):,}")
    print(f"Counts (0: Normal, 1: Risk, 2: Flood):\n{y_train.value_counts().sort_index()}")
    print("\n🔥 Calculated Class Weights (ยิ่งค่าน้ำหนักสูง โมเดลยิ่งให้ความสำคัญกับ Class นั้น):")

    # แสดงผลลัพธ์
    for cls, weight in multiclass_weights.items():
        label = {0: 'Normal', 1: 'Risk', 2: 'Flood'}[cls]
        print(f"  Class {cls} ({label}): {weight:.4f}")

    return multiclass_weights
    # --- 3. คำแนะนำในการใช้งาน ---
    # print("\n--- 💡 คำแนะนำการใช้งาน ---")
    # print("1. สำหรับ Random Forest หรือ Logistic Regression: ใช้พารามิเตอร์ class_weight='balanced'")
    # print("2. สำหรับ XGBoost: ไม่สามารถใช้ 'scale_pos_weight' ได้โดยตรง คุณต้องสร้างอาร์เรย์ 'sample_weight' จากน้ำหนักเหล่านี้ แล้วส่งเป็นพารามิเตอร์ในการเรียก fit()")

def train():
    # 1. Load Processed Data
    X_train, X_test, y_train, y_test = preprocess_data()

    # 2. Calculate multiclass weights
    multiclass_weights = calculate_multiclass_weights()

    # -------------------------------------------------------
    # [ส่วนที่แก้ไข] ตั้งค่า DagsHub Tracking URI
    # -------------------------------------------------------
    # หมายเหตุ: ชื่อ Repo เป็น Loan_Default แต่เราจะสร้าง Experiment ชื่อ Flood_Prediction ข้างในนั้นครับ
    dagshub_uri = "https://dagshub.com/plotter.natchanon/Loan_Defualt_Prediction.mlflow"
    mlflow.set_tracking_uri(dagshub_uri)
    
    # ตั้งชื่อ Experiment (ให้รู้ว่าเป็นโปรเจกต์น้ำท่วม)
    mlflow.set_experiment("Flood_Prediction_Project")

    # -------------------------------------------------------

    with mlflow.start_run() as run:
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "scale_pos_weight": scale_pos_weight,
            "max_depth": 6,
            "learning_rate": 0.1,
            "use_label_encoder": False
        }
        
        mlflow.log_params(params)

        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        mlflow.log_metric("accuracy", acc)
        print(f"Model Accuracy: {acc:.4f}")
        
        # Log Model ไปเก็บไว้บน DagsHub Artifacts
        mlflow.xgboost.log_model(model, "model")

        # Save Model Locally (สำหรับ Docker/Predict ใช้ใน Step ถัดไป)
        model.save_model("models/xgboost_model.json")
        print("Model saved locally to models/xgboost_model.json")

        # --- Model Promotion Logic (เหมือนเดิม) ---
        if acc > 0.85:
            print("Promoting model to Production...")
            model_uri = f"runs:/{run.info.run_id}/model"
            mv = mlflow.register_model(model_uri, "Flood_Model_Prod")
            
            # ใช้ MlflowClient เพื่อปรับ Stage (ถ้าจำเป็น)
            # client = mlflow.tracking.MlflowClient()
            # client.transition_model_version_stage(name="Flood_Model_Prod", version=mv.version, stage="Production")

if __name__ == "__main__":
    train()