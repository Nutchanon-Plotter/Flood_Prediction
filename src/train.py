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

def calculate_multiclass_weights():
    # --- 1. Load Data Artifacts ---
    try:
        # Load Target Multiclass
        y_train = pd.read_csv('../data/preprocess_data/y_train.csv', index_col=0).squeeze()
        print("✅ โหลดไฟล์ y_train.csv สำเร็จ")
    except FileNotFoundError:
        print("❌ Error: ไม่พบไฟล์ y_train.csv")
        print("โปรดตรวจสอบว่าได้รันโค้ด Train/Test Split สำหรับ Multiclass เสร็จสมบูรณ์แล้ว")
        exit()


    # --- 2. Calculate Multiclass Class Weights (Inverse Frequency) ---
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)

    # Create Dictionary
    multiclass_weights = dict(zip(classes, weights))

    print("\n--- 📊 Multiclass Class Weights สำหรับการแก้ Imbalance ---")
    print(f"Total Samples (Train): {len(y_train):,}")
    print(f"Counts (0: Normal, 1: Risk, 2: Flood):\n{y_train.value_counts().sort_index()}")
    print("\n🔥 Calculated Class Weights (ยิ่งค่าน้ำหนักสูง โมเดลยิ่งให้ความสำคัญกับ Class นั้น):")

    for cls, weight in multiclass_weights.items():
        label = {0: 'Normal', 1: 'Risk', 2: 'Flood'}[cls]
        print(f"  Class {cls} ({label}): {weight:.4f}")

    return multiclass_weights
   
def feature_selection():
    y_train = pd.read_csv('../data/preprocess_data/y_train.csv', index_col=0).squeeze()
    X_train_scaled_df = pd.read_csv('../data/preprocess_data/X_train_scaled.csv', index_col=0)
    multiclass_weights = calculate_multiclass_weights()
    sample_weights = y_train.map(multiclass_weights)

    fs_model = xgb.XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )
    fs_model.fit(X_train_scaled_df, y_train, sample_weight=sample_weights)

    importances = fs_model.feature_importances_
    importance_df = pd.DataFrame({
        "Feature": X_train_scaled_df.columns,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)
    print(importance_df)

    K = 10
    selected_features = importance_df.head(K)["Feature"].tolist()
    print("\nSelected features:", selected_features)

    return selected_features

def Load_processed_data():
    selected_features = feature_selection()
    X_train_scale = pd.read_csv('../data/preprocess_data/X_train_scaled.csv', index_col=0)
    X_test_scale = pd.read_csv('../data/preprocess_data/X_test_scaled.csv', index_col=0)
    y_train = pd.read_csv('../data/preprocess_data/y_train.csv', index_col=0).squeeze()
    y_test = pd.read_csv('../data/preprocess_data/y_test.csv', index_col=0).squeeze()
    X_train = X_train_scale[selected_features]
    X_test = X_test_scale[selected_features]
    return X_train, X_test, y_train, y_test

def tune_train_evaluate_mlflow(model, params, model_name, X_train, y_train, X_test, y_test, sample_weights):
    """
    Use Grid Search, train model, Log result and return Metrics and Run ID
    """
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
        
        # Log Hyperparameters
        mlflow.log_params(best_params)
        
        # Log Metrics
        mlflow.log_metric("cv_f1_weighted", best_cv_score)
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("test_f1_weighted", f1_weighted)
        
        for cls, metrics in report.items():
            if isinstance(metrics, dict):
                mlflow.log_metric(f"test_{cls}_f1_score", metrics['f1_score'])

        # 4. Log Model as Artifact
        if model_name == "XGBoost":
            mlflow.xgboost.log_model(best_model, "model")
        else:
            mlflow.sklearn.log_model(best_model, "model")

        # Save Model Locally 
        model_filename = f"models/{model_name}_best_model"
        if model_name == "XGBoost":
             best_model.save_model(f"{model_filename}.json")
        else:
             joblib.dump(best_model, f"{model_filename}.pkl")

        print(f"Model logged to MLflow Run ID: {run.info.run_id}")
        
    print("--------------------------------------------------\n")
    # return Run ID and Metric F1 
    return run.info.run_id, f1_weighted, model_name

def train():
    # 1. Load Processed Data
    X_train, X_test, y_train, y_test = Load_processed_data()
    X_train_values = X_train.values
    X_test_values = X_test.values
    y_train_flat = y_train.values.ravel()
    y_test_flat = y_test.values.ravel()

    # 2. Calculate multiclass weights
    multiclass_weights = calculate_multiclass_weights()
    sample_weights_array = y_train.map(multiclass_weights).values

    # 3. กำหนด Hyperparameter
    lr_params = { 'C': [0.1, 1, 10], 'solver': ['lbfgs'] }
    lr_model_base = LogisticRegression(multi_class='multinomial', random_state=42, max_iter=1000, class_weight='balanced')
    rf_params = { 'n_estimators': [100, 200], 'max_depth': [5, 10, None], 'min_samples_split': [2, 5] }
    rf_model_base = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced')
    num_classes = len(np.unique(y_train))
    xgb_params = { 'n_estimators': [100, 200], 'max_depth': [3, 5], 'learning_rate': [0.01, 0.1], 'reg_lambda': [0.1, 1] }
    xgb_model_base = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42, 
                                  objective='multi:softprob', num_class=num_classes)

    # 4. MLflow Setup
    dagshub_uri = "https://dagshub.com/plotter.natchanon/Loan_Defualt_Prediction/mlflow"
    mlflow.set_tracking_uri(dagshub_uri)
    mlflow.set_experiment("Flood_Prediction_Project")
    
    # 5. สร้าง Run หลักสำหรับ Summary และ Promotion
    print("\n=======================================================")
    print("🚀 เริ่มการทดลอง Grid Search และ MLflow Tracking")
    print("=======================================================\n")
    
    # List เพื่อเก็บผลลัพธ์ของแต่ละโมเดล
    results = []

    with mlflow.start_run(run_name="Summary_and_Promotion") as summary_run:
        
        # A. Logistic Regression (Run Nested)
        lr_run_id, lr_f1, lr_name = tune_train_evaluate_mlflow(
            lr_model_base, lr_params, "Logistic Regression", 
            X_train_values, y_train_flat, X_test_values, y_test_flat, 
            sample_weights=None 
        )
        results.append({"run_id": lr_run_id, "f1_weighted": lr_f1, "name": lr_name})

        # B. Random Forest (Run Nested)
        rf_run_id, rf_f1, rf_name = tune_train_evaluate_mlflow(
            rf_model_base, rf_params, "Random Forest", 
            X_train_values, y_train_flat, X_test_values, y_test_flat, 
            sample_weights=None 
        )
        results.append({"run_id": rf_run_id, "f1_weighted": rf_f1, "name": rf_name})

        # C. XGBoost (Run Nested)
        xgb_run_id, xgb_f1, xgb_name = tune_train_evaluate_mlflow(
            xgb_model_base, xgb_params, "XGBoost", 
            X_train_values, y_train_flat, X_test_values, y_test_flat, 
            sample_weights=sample_weights_array
        )
        results.append({"run_id": xgb_run_id, "f1_weighted": xgb_f1, "name": xgb_name})
        
        # 6. Promotion Logic (เปรียบเทียบ 3 โมเดล)
        
        best_model_result = max(results, key=lambda x: x['f1_weighted'])
        
        best_f1 = best_model_result['f1_weighted']
        best_name = best_model_result['name']
        best_run_id = best_model_result['run_id']
        
        print("\n=======================================================")
        print(f"🏆 Best Model Overall: {best_name} (F1 Weighted: {best_f1:.4f})")
        print("=======================================================")
        
        # set Promotion threshold
        PROMOTION_THRESHOLD = 0.80
        
        if best_f1 >= PROMOTION_THRESHOLD:
            
            model_uri = f"runs:/{best_run_id}/model"
            
            # Log Metric best model in Summary Run
            mlflow.log_metric("best_f1_weighted_overall", best_f1)
            mlflow.set_tag("best_model_name", best_name)
            
            print(f"✅ Promoting best model ({best_name}) to Production...")
            
            # register model
            mv = mlflow.register_model(model_uri, "Flood_Model_Prod")
            
            # ใช้ Client เพื่อเปลี่ยน Stage เป็น Production
            client = MlflowClient()
            client.transition_model_version_stage(
                name="Flood_Model_Prod", 
                version=mv.version, 
                stage="Production",
                archive_existing_versions=True # เก็บเวอร์ชันเก่าเข้า Archive
            )
            print(f"🎉 Model {best_name} (Version {mv.version}) Promoted to Production!")
        else:
            print(f"❌ Best F1 Score ({best_f1:.4f}) is below promotion threshold ({PROMOTION_THRESHOLD:.4f}). Skipping promotion.")


    print("\n✅ การทดลอง Grid Search เสร็จสมบูรณ์ บันทึกผลลัพธ์และทำการ Promote แล้ว")
# def train():
#     # 1. Load Processed Data
#     # X_train, X_test, y_train, y_test = preprocess_data()
#     X_train, X_test, y_train, y_test = Load_processed_data()
#     X_train = X_train.values
#     X_test = X_test.values
#     y_train_flat = y_train.values.ravel() # จัด y_train ให้อยู่ในรูป 1 มิติ
#     y_test_flat = y_test.values.ravel()

#     # 2. Calculate multiclass weights
#     multiclass_weights = calculate_multiclass_weights()
#     sample_weights = y_train.map(multiclass_weights).values

#     # -------------------------------------------------------
#     # [ส่วนที่แก้ไข] ตั้งค่า DagsHub Tracking URI
#     # -------------------------------------------------------
#     # หมายเหตุ: ชื่อ Repo เป็น Loan_Default แต่เราจะสร้าง Experiment ชื่อ Flood_Prediction ข้างในนั้นครับ
#     dagshub_uri = "https://dagshub.com/plotter.natchanon/Loan_Defualt_Prediction.mlflow"
#     mlflow.set_tracking_uri(dagshub_uri)
    
#     # ตั้งชื่อ Experiment (ให้รู้ว่าเป็นโปรเจกต์น้ำท่วม)
#     mlflow.set_experiment("Flood_Prediction_Project")

#     # -------------------------------------------------------

#     with mlflow.start_run() as run:
#         params = {
#             "objective": "binary:logistic",
#             "eval_metric": "logloss",
#             "scale_pos_weight": scale_pos_weight,
#             "max_depth": 6,
#             "learning_rate": 0.1,
#             "use_label_encoder": False
#         }
        
#         mlflow.log_params(params)

#         model = xgb.XGBClassifier(**params)
#         model.fit(X_train, y_train)

#         y_pred = model.predict(X_test)
#         acc = accuracy_score(y_test, y_pred)

#         mlflow.log_metric("accuracy", acc)
#         print(f"Model Accuracy: {acc:.4f}")
        
#         # Log Model ไปเก็บไว้บน DagsHub Artifacts
#         mlflow.xgboost.log_model(model, "model")

#         # Save Model Locally (สำหรับ Docker/Predict ใช้ใน Step ถัดไป)
#         model.save_model("models/xgboost_model.json")
#         print("Model saved locally to models/xgboost_model.json")

#         # --- Model Promotion Logic (เหมือนเดิม) ---
#         if acc > 0.85:
#             print("Promoting model to Production...")
#             model_uri = f"runs:/{run.info.run_id}/model"
#             mv = mlflow.register_model(model_uri, "Flood_Model_Prod")
            
#             # ใช้ MlflowClient เพื่อปรับ Stage (ถ้าจำเป็น)
#             # client = mlflow.tracking.MlflowClient()
#             # client.transition_model_version_stage(name="Flood_Model_Prod", version=mv.version, stage="Production")

if __name__ == "__main__":
    # calculate_multiclass_weights()
    # feature_selection()
    train()