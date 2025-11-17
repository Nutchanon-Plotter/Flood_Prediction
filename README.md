# Flood Prediction MLOps System

โปรเจกต์นี้เป็นส่วนหนึ่งของรายวิชา **CPE393 Machine Learning Operations**
พัฒนาระบบทำนายความเสี่ยงน้ำท่วมแบบ End-to-End MLOps Pipeline ที่ครอบคลุมตั้งแต่การดึงข้อมูล, การตรวจสอบ Data Drift, การเทรนโมเดลอัตโนมัติ (Retraining), ไปจนถึงการ Deploy ขึ้นเว็บแอปพลิเคชัน

---

## Project Overview

ระบบนี้ทำหน้าที่พยากรณ์ความเสี่ยงน้ำท่วมล่วงหน้า 7 วัน โดยใช้ข้อมูลสภาพอากาศและระดับน้ำจาก **Open-Meteo API** ระบบถูกออกแบบให้ทำงานอัตโนมัติผ่าน **GitHub Actions** เพื่อรองรับการเปลี่ยนแปลงของข้อมูล (Data Drift) และรักษาประสิทธิภาพของโมเดลให้ทันสมัยอยู่เสมอ 

### Key Features
* **Automated Pipeline:** ทำงานอัตโนมัติทุกวัน (Daily Scheduled) ผ่าน GitHub Actions
* **Drift Detection:** ตรวจสอบความผิดปกติของข้อมูลด้วย **Evidently AI** 
* **Auto-Retraining:** เทรนโมเดลใหม่ทันทีหากพบ Data Drift
* **Experiment Tracking:** บันทึกผลการทดลองและโมเดลผ่าน **MLflow** บน **DagsHub**
* **Bias Mitigation:** ลดความลำเอียงของโมเดลด้วยเทคนิค Class Weighting และ Threshold Adjustment
* **Deployment:** ให้บริการโมเดลผ่าน **FastAPI** (REST API) และหน้าเว็บ Frontend อย่างง่าย 

---

## 🛠️ Tech Stack

* **Language:** Python 3.9
* **Machine Learning:** XGBoost, Scikit-learn
* **MLOps Tools:** MLflow, DagsHub, Evidently AI, GitHub Actions [cite: 11]
* **API & Web:** FastAPI, HTML/JS
* **Containerization:** Docker
* **Data Source:** Open-Meteo API (Flood & Weather Archive)

---

## 📂 Project Structure

```text
CPE393_Final_Project/
│
├── .github/workflows/
│   └── smart_pipeline.yml   # GitHub Actions Workflow (Drift Check -> Retrain -> Predict)
│
├── api/
│   ├── Dockerfile           # Docker setup for API
│   └── main.py              # FastAPI Backend
│
├── data/                    # Data storage (Ignored by Git)
├── models/                  # Artifacts (model.json, scaler.pkl, drift_reports)
│
├── notebooks/
│   └── bias_analysis.ipynb  # EDA & Bias Analysis Report
│
├── src/
│   ├── data_loader.py       # Fetch data from API
│   ├── preprocess.py        # Feature Engineering & Scaling
│   ├── train.py             # Model Training & MLflow Logging
│   ├── monitor.py           # Data Drift Detection (Evidently)
│   └── predict_daily.py     # Generate 7-day forecast JSON
│
├── web/
│   └── index.html           # Frontend Interface
│
├── requirements.txt         # Python Dependencies
└── README.md                # Project Documentation
```

## MLOps Pipeline Automation (GitHub Actions)
ระบบถูกตั้งค่าให้รันอัตโนมัติ ทุกวันเวลา 00:00 (UTC) หรือเมื่อมีการ Push Code ผ่านไฟล์ .github/workflows/smart_pipeline.yml โดยมี Logic ดังนี้:

* Fetch Data: ดึงข้อมูลน้ำและอากาศล่าสุด


* Drift Check: ใช้ Evidently AI เปรียบเทียบข้อมูลใหม่กับ Reference Data 
```
Conditional Logic:

🚨 ถ้าเจอ Drift: ระบบจะรัน train.py เพื่อ Retrain โมเดลใหม่โดยอัตโนมัติ

✅ ถ้าไม่เจอ Drift: ข้ามขั้นตอนเทรน ไปขั้นตอนทำนายทันที
```
* Prediction: สร้างไฟล์ prediction_results.json สำหรับ 7 วันข้างหน้า

* Deployment: อัปเดตผลการทำนายและโมเดลล่าสุดกลับเข้าสู่ GitHub Repo เพื่อให้หน้าเว็บดึงไปแสดงผล

## Monitoring & Observability

* Model Performance: ตรวจสอบค่า Accuracy, Loss, และ Parameters ได้ที่ 
 https://dagshub.com/plotter.natchanon/Loan_Defualt_Prediction.mlflow/ 

* Data Drift Report: ไฟล์ HTML report จะถูกสร้างขึ้นทุกครั้งที่มีการรัน Pipeline เก็บไว้ในโฟลเดอร์ ```models/drift_report.html```


👥 Team Members

CPE393 Final Project Group 

Student ID - Name (Role: Data Pipeline/Data Scientist) 

Student ID - Name (Role: ML Engineer/Model Training) 

Student ID - Name (Role: ML Infra/Deployment) 

Student ID - Name (Role: ML Infra/Deployment) 


**King Mongkut's University of Technology Thonburi**