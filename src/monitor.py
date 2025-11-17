# src/monitor.py
import pandas as pd
import os
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.test_suite import TestSuite
from evidently.tests import TestNumberOfDriftedColumns

def monitor_drift():
    # 1. Load Reference (จากที่เทรนไว้)
    try:
        reference_data = pd.read_csv('models/reference_data.csv')
    except FileNotFoundError:
        print("Reference data missing.")
        # ถ้าไม่มี ref ให้บังคับ drift = true เพื่อเทรนใหม่
        set_output(True)
        return

    # 2. Load Current Data (ดึงข้อมูลล่าสุดมาเทียบ)
    # ในที่นี้ดึงจาก data/raw/raw_data.csv ที่เพิ่งโหลดมาใหม่
    current_data = pd.read_csv('data/raw/raw_data.csv')
    
    # เลือกมาแค่ 50-100 แถวล่าสุดเพื่อเช็คสภาพปัจจุบัน
    current_window = current_data.tail(100).copy()

    # 3. Check Drift with Test Suite
    # เงื่อนไข: ถ้ามี Column Drift มากกว่า 1 คอลัมน์ถือว่า Drift
    drift_suite = TestSuite(tests=[TestNumberOfDriftedColumns(lt=1)])
    drift_suite.run(reference_data=reference_data, current_data=current_window)
    
    # Save Report
    drift_suite.save_html("models/drift_report.html")
    
    # 4. Check Result & Output to GitHub Actions
    result = drift_suite.as_dict()
    is_drift = not result["tests"][0]["parameters"]["condition"]["pass"]
    
    print(f"Drift Detected: {is_drift}")
    set_output(is_drift)

def set_output(is_drift):
    # เขียนค่าลง Environment Variable ของ GitHub Actions
    if 'GITHUB_OUTPUT' in os.environ:
        with open(os.environ['GITHUB_OUTPUT'], 'a') as fh:
            # ถ้า drift=True จะเขียน string 'true'
            fh.write(f"drift_detected={str(is_drift).lower()}\n")

if __name__ == "__main__":
    monitor_drift()