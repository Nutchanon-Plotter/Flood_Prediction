# src/train.py
import mlflow
import mlflow.xgboost
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, f1_score
import pandas as pd
import os
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier
import joblib
from mlflow.tracking import MlflowClient
import shutil

# ... (ฟังก์ชัน calculate_multiclass_weights, feature_selection, Load_processed_data เหมือนเดิม) ...
# ... (ขอข้ามเพื่อความกระชับ ให้ใช้โค้ดเดิมของคุณในส่วนบน) ...

def calculate_multiclass_weights():
    # --- 1. Load Data Artifacts ---
    try:
        # Load Target Multiclass
        y_train = pd.read_csv('data/preprocess_data/y_train.csv', index_col=0).squeeze()
        print("✅ โหลดไฟล์ y_train.csv สำเร็จ")
    except FileNotFoundError:
        print("❌ Error: ไม่พบไฟล์ y_train.csv")
        print("โปรดตรวจสอบว่าได้รันโค้ด Train/Test Split สำหรับ Multiclass เสร็จสมบูรณ์แล้ว")
        exit()

    # --- 2. Calculate Multiclass Class Weights (Inverse Frequency) ---
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)

    multiclass_weights = dict(zip(classes, weights))
    print(f"Calculated Weights: {multiclass_weights}")
    return multiclass_weights

def Load_processed_data():
    # (ใช้โค้ดเดิมของคุณ)
    X_train_scale = pd.read_csv('data/preprocess_data/X_train_scaled.csv', index_col=0)
    X_test_scale = pd.read_csv('data/preprocess_data/X_test_scaled.csv', index_col=0)
    y_train = pd.read_csv('data/preprocess_data/y_train.csv', index_col=0).squeeze()
    y_test = pd.read_csv('data/preprocess_data/y_test.csv', index_col=0).squeeze()
    
    # เลือกเฉพาะ Feature ที่ใช้ (Hardcode ไว้ก่อนเพื่อลดความซับซ้อน หรือเรียก feature_selection() ก็ได้)
    # เพื่อความชัวร์ ผมจะใช้ feature_selection() ตามโค้ดคุณ
    # แต่ถ้าอยากให้เร็ว comment feature_selection ออกแล้วใช้ feature ทั้งหมดก็ได้
    # selected_features = feature_selection() 
    # X_train = X_train_scale[selected_features]
    # X_test = X_test_scale[selected_features]
    
    return X_train_scale, X_test_scale, y_train, y_test

def tune_train_evaluate_mlflow(model, params, model_name, X_train, y_train, X_test, y_test, sample_weights):
    print(f"--- 🧠 Starting Grid Search Tuning for: {model_name} ---")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=params,
        scoring='f1_weighted', 
        cv=cv,
        n_jobs=-1,
        verbose=1
    )

    # 1. Grid Search with Sample Weights
    fit_params = {}
    if model_name == "XGBoost":
        fit_params['sample_weight'] = sample_weights

    grid_search.fit(X_train, y_train, **fit_params)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_cv_score = grid_search.best_score_
    
    # 2. Prediction และ Metrics
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    report = classification_report(y_test, y_pred, output_dict=True)

    print(f"\n✅ Best Parameters: {best_params}")
    print(f"✅ Best F1 Score (Weighted) on Test Set: {f1_weighted:.4f}")
    
    # 3. MLflow Tracking
    with mlflow.start_run(run_name=f"GridSearch_{model_name}", nested=True) as run:
        mlflow.log_params(best_params)
        mlflow.log_metric("cv_f1_weighted", best_cv_score)
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("test_f1_weighted", f1_weighted)
        
        for cls, metrics in report.items():
            if isinstance(metrics, dict) and 'f1-score' in metrics:
                mlflow.log_metric(f"test_{cls}_f1_score", metrics['f1-score'])

        # --- [CRITICAL UPDATE] Log Scaler to MLflow ---
        scaler_path = 'data/preprocess_data/scaler.pkl'
        if os.path.exists(scaler_path):
            print(f"📦 Logging scaler artifact from {scaler_path}")
            # log_artifact จะส่งไฟล์ขึ้นไปเก็บคู่กับ Model บน Server
            mlflow.log_artifact(scaler_path, artifact_path="model")
        else:
            print(f"⚠️ Warning: Scaler file not found at {scaler_path}")

        # 4. Log Model as Artifact
        if model_name == "XGBoost":
            mlflow.xgboost.log_model(best_model, "model")
        else:
            mlflow.sklearn.log_model(best_model, "model")

        # --- Save Model Locally (จุดที่แก้ Error) ---
        # ตรวจสอบว่ามีโฟลเดอร์ models หรือไม่ ถ้าไม่มีให้สร้าง
        os.makedirs("models", exist_ok=True) # <--- เพิ่มบรรทัดนี้

        model_filename = f"models/{model_name}_best_model"
        if model_name == "XGBoost":
            best_model.save_model(f"{model_filename}.json")
        else:
            joblib.dump(best_model, f"{model_filename}.pkl") # <--- บรรทัดนี้ที่เคย Error

        if os.path.exists(scaler_path):
            shutil.copy(scaler_path, "models/scaler.pkl")
            print(f"✅ Scaler copied to models/scaler.pkl")

        print(f"Model logged to MLflow Run ID: {run.info.run_id}")
        
    print("--------------------------------------------------\n")
    return run.info.run_id, f1_weighted, model_name

def train():
    # >>> แก้ไข 1: สร้างโฟลเดอร์ models ทันทีเมื่อเริ่มโปรแกรม <<<
    os.makedirs("models", exist_ok=True)
    print("📁 Created 'models' directory.")

    # 1. Load Processed Data
    X_train, X_test, y_train, y_test = Load_processed_data()
    X_train_values = X_train.values
    X_test_values = X_test.values
    y_train_flat = y_train.values.ravel()
    y_test_flat = y_test.values.ravel()

    # 2. Calculate multiclass weights
    multiclass_weights = calculate_multiclass_weights()
    sample_weights_array = y_train.map(multiclass_weights).values

    # 3. กำหนด Hyperparameter (เหมือนเดิม)
    lr_params = { 'C': [0.1, 1, 10], 'solver': ['lbfgs'] }
    lr_model_base = LogisticRegression(multi_class='multinomial', random_state=42, max_iter=1000, class_weight='balanced')
    rf_params = { 'n_estimators': [100], 'max_depth': [5, 10], 'min_samples_split': [2] } # ลดจำนวนลงเพื่อให้รันเร็วขึ้น
    rf_model_base = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced')
    num_classes = len(np.unique(y_train))
    xgb_params = { 'n_estimators': [100], 'max_depth': [3], 'learning_rate': [0.1], 'reg_lambda': [1] }
    xgb_model_base = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42, 
                                  objective='multi:softprob', num_class=num_classes)

    # 4. MLflow Setup
    dagshub_uri = "https://dagshub.com/plotter.natchanon/Flood_Prediction.mlflow"
    mlflow.set_tracking_uri(dagshub_uri)
    mlflow.set_experiment("Flood_Prediction_Project")
    
    print("\n=======================================================")
    print("🚀 เริ่มการทดลอง Grid Search และ MLflow Tracking")
    print("=======================================================\n")
    
    results = []

    with mlflow.start_run(run_name="Summary_and_Promotion") as summary_run:
        
        # A. Logistic Regression
        lr_run_id, lr_f1, lr_name = tune_train_evaluate_mlflow(
            lr_model_base, lr_params, "Logistic Regression", 
            X_train_values, y_train_flat, X_test_values, y_test_flat, sample_weights=None 
        )
        results.append({"run_id": lr_run_id, "f1_weighted": lr_f1, "name": lr_name})

        # B. Random Forest
        rf_run_id, rf_f1, rf_name = tune_train_evaluate_mlflow(
            rf_model_base, rf_params, "Random Forest", 
            X_train_values, y_train_flat, X_test_values, y_test_flat, sample_weights=None 
        )
        results.append({"run_id": rf_run_id, "f1_weighted": rf_f1, "name": rf_name})

        # C. XGBoost
        xgb_run_id, xgb_f1, xgb_name = tune_train_evaluate_mlflow(
            xgb_model_base, xgb_params, "XGBoost", 
            X_train_values, y_train_flat, X_test_values, y_test_flat, sample_weights=sample_weights_array
        )
        results.append({"run_id": xgb_run_id, "f1_weighted": xgb_f1, "name": xgb_name})
        
        # 6. Promotion Logic
        best_model_result = max(results, key=lambda x: x['f1_weighted'])
        best_f1 = best_model_result['f1_weighted']
        best_name = best_model_result['name']
        best_run_id = best_model_result['run_id']
        
        print(f"\n🏆 Best Model: {best_name} (F1: {best_f1:.4f})")
        
        PROMOTION_THRESHOLD = 0.80
        if best_f1 >= PROMOTION_THRESHOLD:
            model_uri = f"runs:/{best_run_id}/model"
            mlflow.log_metric("best_f1_weighted_overall", best_f1)
            mlflow.set_tag("best_model_name", best_name)
            
            print(f"✅ Promoting {best_name} to Production...")
            mv = mlflow.register_model(model_uri, "Flood_Model_Prod")
            
            client = MlflowClient()
            client.transition_model_version_stage(
                name="Flood_Model_Prod", 
                version=mv.version, 
                stage="Production",
                archive_existing_versions=True
            )
            print(f"🎉 Promoted Version {mv.version}!")
        else:
            print(f"❌ Skipping promotion (Threshold {PROMOTION_THRESHOLD}).")

    print("\n✅ Completed.")

if __name__ == "__main__":
    train()