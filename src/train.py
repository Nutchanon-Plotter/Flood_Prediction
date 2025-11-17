# src/train.py
import mlflow
import mlflow.xgboost
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
from preprocess import preprocess_data
import pandas as pd
import os

def train():
    # 1. Load Processed Data
    X_train, X_test, y_train, y_test = preprocess_data()

    # 2. Calculate scale_pos_weight
    class_counts = y_train.value_counts()
    scale_pos_weight = class_counts[0] / class_counts[1]

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