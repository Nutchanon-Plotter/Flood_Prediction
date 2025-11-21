import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
import numpy as np
import os  # <--- เพิ่ม import os
from datetime import datetime # <--- เพิ่ม datetime เพื่อทำ Dynamic Date

class WeatherLoader:
    def __init__(self, timezone="Asia/Bangkok"):
        self.timezone = timezone

        # Setup Client
        cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        self.client = openmeteo_requests.Client(session=retry_session)

        # --- API ENDPOINTS ---
        self.URLS = {
            "history_weather": "https://archive-api.open-meteo.com/v1/archive",
            "history_flood": "https://flood-api.open-meteo.com/v1/flood",
        }

    def _fetch_api(self, url, lat, lon, params, variables):
        """Helper to fetch and clean data from any Open-Meteo API for a single location"""

        full_params = {
            "latitude": lat,
            "longitude": lon,
            "daily": variables,
            "timezone": self.timezone,
            **params 
        }

        try:
            responses = self.client.weather_api(url, params=full_params)
            response = responses[0]
        except Exception as e:
            print(f"Error fetching {url} for ({lat}, {lon}): {e}")
            return pd.DataFrame()

        # Process Daily Data
        daily = response.Daily()

        data = {"date": pd.date_range(
            start=pd.to_datetime(daily.Time(), unit="s", utc=True),
            end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=daily.Interval()),
            inclusive="left"
        )}

        # Dynamically extract variables
        for i, var in enumerate(variables):
            data[var] = daily.Variables(i).ValuesAsNumpy()

        df = pd.DataFrame(data)

        # Convert from UTC to requested timezone
        df['date'] = df['date'].dt.tz_convert(self.timezone)

        return df

    def get_historical_data_multi(self, locations, start_date, end_date, weather_vars, flood_vars):
        all_data_frames = []
        common_params = {"start_date": start_date, "end_date": end_date}

        print(f"--- Fetching Historical Data from {start_date} to {end_date} for {len(locations)} locations ---")

        for loc in locations:
            name, lat, lon = loc['name'], loc['lat'], loc['lon']
            print(f"  > Processing: {name} (Lat: {lat}, Lon: {lon})")

            # 1. Fetch Weather History
            df_weather = self._fetch_api(
                self.URLS["history_weather"], lat, lon, common_params, weather_vars
            )

            # 2. Fetch Flood History
            df_flood = self._fetch_api(
                self.URLS["history_flood"], lat, lon, common_params, flood_vars
            )

            # 3. Merge Weather and Flood
            if not df_weather.empty and not df_flood.empty:
                merged_df = pd.merge(df_weather, df_flood, on="date", how="inner")
                merged_df['location'] = name
                all_data_frames.append(merged_df)
            else:
                print(f"    [Warning] Skipping {name}: Data incomplete or failed to load.")

        # 4. Concatenate
        if all_data_frames:
            return pd.concat(all_data_frames, ignore_index=True)

        print("--- All locations failed to load data. Returning empty DataFrame. ---")
        return pd.DataFrame()

# ----------------------------------------------------------------------
# Main Execution Flow
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Define Locations
    LOCATIONS1 = [
        {"name": "ChaoPhraya_Dam", "lat": 15.159213709346405, "lon": 100.17985248529882},
    ]
    
    LOCATIONS2 = [
        {"name": "NakhonSawan_Muang_Upstream", "lat": 15.700409309316225, "lon": 100.14120663110944},
    ]

    # 2. Define Variables
    WEATHER_VARS = ["temperature_2m_min", "apparent_temperature_mean", "apparent_temperature_max", "apparent_temperature_min", "temperature_2m_max", "temperature_2m_mean", "daylight_duration", "sunshine_duration", "precipitation_sum", "rain_sum", "snowfall_sum", "precipitation_hours", "wind_speed_10m_max", "wind_direction_10m_dominant", "shortwave_radiation_sum", "et0_fao_evapotranspiration", "soil_moisture_0_to_100cm_mean", "soil_moisture_0_to_7cm_mean", "soil_moisture_28_to_100cm_mean", "soil_moisture_7_to_28cm_mean", "soil_temperature_0_to_100cm_mean", "soil_temperature_0_to_7cm_mean", "soil_temperature_28_to_100cm_mean", "soil_temperature_7_to_28cm_mean", "wet_bulb_temperature_2m_mean", "wet_bulb_temperature_2m_max", "wet_bulb_temperature_2m_min", "vapour_pressure_deficit_max", "surface_pressure_mean", "surface_pressure_max", "surface_pressure_min", "winddirection_10m_dominant", "wind_gusts_10m_max", "wind_gusts_10m_mean", "wind_speed_10m_mean", "wind_gusts_10m_min", "wind_speed_10m_min", "pressure_msl_min", "pressure_msl_max", "pressure_msl_mean", "snowfall_water_equivalent_sum", "relative_humidity_2m_max", "relative_humidity_2m_min", "relative_humidity_2m_mean", "et0_fao_evapotranspiration_sum", "dew_point_2m_min", "dew_point_2m_max", "dew_point_2m_mean", "cloud_cover_mean", "cloud_cover_max", "cloud_cover_min"]
    FLOOD_VARS = ["river_discharge"]

    # 3. Initialize Loader
    loader = WeatherLoader(timezone="Asia/Bangkok")
    
    # --- Dynamic Date for GitHub Actions ---
    # ใช้วันปัจจุบันเป็นวันสิ้นสุด เพื่อให้ได้ข้อมูลล่าสุดเสมอเมื่อ Actions รัน
    today_str = datetime.now().strftime('%Y-%m-%d')
    
    print(f"Starting data fetch job... (Target End Date: {today_str})")

    # Fetch Data 1
    df_chaoPhraya = loader.get_historical_data_multi(
        locations=LOCATIONS1,
        start_date="2000-01-01",
        end_date=today_str, # ใช้ dynamic date
        weather_vars=WEATHER_VARS,
        flood_vars=FLOOD_VARS
    )

    # Fetch Data 2
    df_NakhonSawan = loader.get_historical_data_multi(
        locations=LOCATIONS2,
        start_date="2000-01-01",
        end_date=today_str, # ใช้ dynamic date
        weather_vars=WEATHER_VARS,
        flood_vars=FLOOD_VARS
    )

    # 4. Merge
    df_final_merged = pd.concat([df_chaoPhraya, df_NakhonSawan], ignore_index=True)
    print(f"\nTotal rows in final DF: {len(df_final_merged)}")

    # -------------------------------------------------------------------------
    # 5. Save CSV with Robust Path Handling for GitHub Actions
    # -------------------------------------------------------------------------
    
    # หาตำแหน่งของไฟล์ script นี้ (เช่น /home/runner/work/Project/Project/src/data_loader.py)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # ถอยหลังไป 1 ชั้นเพื่อกลับไป root (จาก src/ -> Project/)
    project_root = os.path.dirname(current_dir)
    
    # สร้าง path เต็มไปยัง data/raw/raw_data.csv
    output_dir = os.path.join(project_root, "data", "raw")
    output_file = os.path.join(output_dir, "raw_data.csv")
    
    # สร้างโฟลเดอร์ data/raw ถ้ายังไม่มี (สำคัญมากใน GitHub Actions)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save
    df_final_merged.to_csv(output_file, index=False)
    print(f"✅ Data successfully saved to: {output_file}")