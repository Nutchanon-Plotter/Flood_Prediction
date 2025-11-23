# src/train.py
import mlflow
import mlflow.xgboost
import mlflow.sklearn
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, f1_score
import pandas as pd
import os
import numpy as np
import shutil
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier
import joblib
from mlflow.tracking import MlflowClient


# --- 🛠️ Path Setup (เพื่อให้รันได้ทั้ง Local และ GitHub Actions) ---
# หาตำแหน่งของไฟล์นี้ (src/train.py)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# ถอยหลัง 1 ขั้นเพื่อหา Project Root
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

# Helper function เพื่อสร้าง Path ที่ถูกต้อง
def get_path(*args):
    return os.path.join(PROJECT_ROOT, *args)

# --- Helper Functions ---

def calculate_multiclass_weights():
    # Load Target Data (ใช้ get_path แก้ปัญหา FileNotFoundError)
    data_path = get_path('data', 'preprocess_data', 'y_train.csv')
    try:
        y_train = pd.read_csv(data_path, index_col=0).squeeze()
        print(f"✅ โหลดไฟล์ y_train.csv สำเร็จจาก: {data_path}")
    except FileNotFoundError:
        print(f"❌ Error: ไม่พบไฟล์ y_train.csv ที่ {data_path}")
        exit(1)

    # Calculate Weights
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    multiclass_weights = dict(zip(classes, weights))

    print("\n--- 📊 Multiclass Class Weights ---")
    for cls, weight in multiclass_weights.items():
        print(f"  Class {cls}: {weight:.4f}")

    return multiclass_weights

def feature_selection():
    y_path = get_path('data', 'preprocess_data', 'y_train.csv')
    X_path = get_path('data', 'preprocess_data', 'X_train_scaled.csv')

    try:
        y_train = pd.read_csv(y_path, index_col=0).squeeze()
        X_train_scaled_df = pd.read_csv(X_path, index_col=0)
    except FileNotFoundError:
        print("❌ Error: ไม่พบไฟล์ข้อมูลสำหรับ Feature Selection")
        return []

    # Calculate weights for XGBoost
    multiclass_weights = calculate_multiclass_weights()
    sample_weights = y_train.map(multiclass_weights)

    # Train temporary model for selection
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
    
    print("\n📊 Top Feature Importances:")
    print(importance_df.head(10))

    # Select Top 10 Features
    K = 10
    selected_features = importance_df.head(K)["Feature"].tolist()
    print(f"\n✅ Selected Top {K} Features: {selected_features}")

    return selected_features

def Load_processed_data():
    # เรียกใช้ Feature Selection
    selected_features = feature_selection()
    
    # Load full data using robust paths
    X_train_scale = pd.read_csv(get_path('data', 'preprocess_data', 'X_train_scaled.csv'), index_col=0)
    X_test_scale = pd.read_csv(get_path('data', 'preprocess_data', 'X_test_scaled.csv'), index_col=0)
    y_train = pd.read_csv(get_path('data', 'preprocess_data', 'y_train.csv'), index_col=0).squeeze()
    y_test = pd.read_csv(get_path('data', 'preprocess_data', 'y_test.csv'), index_col=0).squeeze()
    
    # Filter only selected features
    if selected_features:
        X_train = X_train_scale[selected_features]
        X_test = X_test_scale[selected_features]
    else:
        print("⚠️ Warning: Feature selection failed, using all features.")
        X_train = X_train_scale
        X_test = X_test_scale
        
    return X_train, X_test, y_train, y_test

def tune_train_evaluate_mlflow(model, params, model_name, X_train, y_train, X_test, y_test, sample_weights):
    print(f"\n--- 🧠 Starting Grid Search Tuning for: {model_name} ---")

    # สร้างโฟลเดอร์ models (ใช้ Absolute Path)
    models_dir = get_path("models")
    os.makedirs(models_dir, exist_ok=True)

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=params,
        scoring='f1_weighted', 
        cv=cv,
        n_jobs=-1,
        verbose=1
    )

    # Handle Sample Weights
    fit_params = {}
    if model_name == "XGBoost" and sample_weights is not None:
        fit_params['sample_weight'] = sample_weights

    grid_search.fit(X_train, y_train, **fit_params)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_cv_score = grid_search.best_score_
    
    # Predict & Metrics
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    report = classification_report(y_test, y_pred, output_dict=True)

    print(f"✅ Best Params: {best_params}")
    print(f"✅ Test F1 (Weighted): {f1_weighted:.4f}")
    
    # MLflow Tracking
    with mlflow.start_run(run_name=f"GridSearch_{model_name}", nested=True) as run:
        mlflow.log_params(best_params)
        mlflow.log_metric("cv_f1_weighted", best_cv_score)
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("test_f1_weighted", f1_weighted)
        
        # Log F1-score per class
        for cls, metrics in report.items():
            if isinstance(metrics, dict) and 'f1-score' in metrics:
                mlflow.log_metric(f"test_{cls}_f1_score", metrics['f1-score'])

        # Log Model to DagsHub
        if model_name == "XGBoost":
            mlflow.xgboost.log_model(best_model, "model")
        else:
            mlflow.sklearn.log_model(best_model, "model")

        # Save Model Locally
        model_filename = os.path.join(models_dir, f"{model_name}_best_model")
        if model_name == "XGBoost":
             best_model.save_model(f"{model_filename}.json")
        else:
             joblib.dump(best_model, f"{model_filename}.pkl")

        print(f"💾 Model saved locally to: {model_filename}")
        
    return run.info.run_id, f1_weighted, model_name

# --- Main Execution ---

def train():
    # 0. Setup Directories
    os.makedirs(get_path("models"), exist_ok=True)
    
    # 1. Load Data
    X_train, X_test, y_train, y_test = Load_processed_data()
    X_train_values = X_train.values
    X_test_values = X_test.values
    y_train_flat = y_train.values.ravel()
    y_test_flat = y_test.values.ravel()

    # 2. Weights
    multiclass_weights = calculate_multiclass_weights()
    sample_weights_array = y_train.map(multiclass_weights).values

    # 3. Hyperparameters
    lr_params = { 'C': [1, 10], 'solver': ['lbfgs'] }
    lr_model_base = LogisticRegression(multi_class='multinomial', random_state=42, max_iter=1000, class_weight='balanced')
    
    rf_params = { 'n_estimators': [100], 'max_depth': [10, None] }
    rf_model_base = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced')
    
    num_classes = len(np.unique(y_train))
    xgb_params = { 'n_estimators': [100], 'max_depth': [3, 5], 'learning_rate': [0.1] }
    xgb_model_base = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42, 
                                  objective='multi:softprob', num_class=num_classes)

    # 4. Initialize DagsHub & MLflow
    print("\n=======================================================")
    print("🚀 Initializing DagsHub & MLflow")
    print("=======================================================")
    
    # Init DagsHub: อ่าน Token จาก env variable 'DAGSHUB_TOKEN' อัตโนมัติ
    dagshub.init(repo_owner='plotter.natchanon', repo_name='Flood_Prediction', mlflow=True)
    mlflow.set_experiment("Flood_Prediction_Project")
    
    print("\n=======================================================")
    print("🚀 Starting Model Training & MLflow Tracking")
    print("=======================================================")
    
    results = []

    with mlflow.start_run(run_name="Summary_and_Promotion") as summary_run:
        
        # Logistic Regression
        lr_run_id, lr_f1, lr_name = tune_train_evaluate_mlflow(
            lr_model_base, lr_params, "Logistic Regression", 
            X_train_values, y_train_flat, X_test_values, y_test_flat, None 
        )
        results.append({"run_id": lr_run_id, "f1_weighted": lr_f1, "name": lr_name})

        # Random Forest
        rf_run_id, rf_f1, rf_name = tune_train_evaluate_mlflow(
            rf_model_base, rf_params, "Random Forest", 
            X_train_values, y_train_flat, X_test_values, y_test_flat, None 
        )
        results.append({"run_id": rf_run_id, "f1_weighted": rf_f1, "name": rf_name})

        # XGBoost
        xgb_run_id, xgb_f1, xgb_name = tune_train_evaluate_mlflow(
            xgb_model_base, xgb_params, "XGBoost", 
            X_train_values, y_train_flat, X_test_values, y_test_flat, sample_weights_array
        )
        results.append({"run_id": xgb_run_id, "f1_weighted": xgb_f1, "name": xgb_name})
        
        # 5. Select Best Model
        best_model_result = max(results, key=lambda x: x['f1_weighted'])
        best_f1 = best_model_result['f1_weighted']
        best_name = best_model_result['name']
        best_run_id = best_model_result['run_id']
        
        print(f"\n🏆 BEST MODEL: {best_name} (F1: {best_f1:.4f})")

        # 6. Save Production Model (Renaming)
        models_dir = get_path("models")
        source_ext = "json" if best_name == "XGBoost" else "pkl"
        source_path = os.path.join(models_dir, f"{best_name}_best_model.{source_ext}")
        target_path = os.path.join(models_dir, f"production_model.{source_ext}")
        
        if os.path.exists(source_path):
            shutil.copy(source_path, target_path)
            print(f"✅ Production model saved to: {target_path}")
            
            # Save Metadata
            with open(os.path.join(models_dir, "model_metadata.txt"), "w") as f:
                f.write(best_name)
                
            with open(os.path.join(models_dir, "model_filename.txt"), "w") as f:
                f.write(f"production_model.{source_ext}")
        else:
            print(f"⚠️ Error: Source model file not found: {source_path}")

        # 7. MLflow Promotion
        PROMOTION_THRESHOLD = 0.80
        
        if best_f1 >= PROMOTION_THRESHOLD:
            mlflow.log_metric("best_f1_weighted_overall", best_f1)
            mlflow.set_tag("best_model_name", best_name)
            
            print(f"✅ Promoting {best_name} to Production in MLflow Registry...")
            model_uri = f"runs:/{best_run_id}/model"
            
            mv = mlflow.register_model(model_uri, "Flood_Model_Prod")
            
            client = MlflowClient()
            client.transition_model_version_stage(
                name="Flood_Model_Prod", 
                version=mv.version, 
                stage="Production",
                archive_existing_versions=True
            )
            print(f"🎉 Promotion Complete! Version: {mv.version}")
        else:
            print(f"❌ Skipping Promotion (Score {best_f1:.4f} < {PROMOTION_THRESHOLD})")

    print("\n✅ Training Pipeline Completed Successfully.")

if __name__ == "__main__":
    train()