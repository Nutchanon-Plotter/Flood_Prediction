import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
import numpy as np
import os
import time
from datetime import datetime

class WeatherLoader:
    def __init__(self, timezone="Asia/Bangkok"):
        self.timezone = timezone

        # Setup Client
        cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        self.client = openmeteo_requests.Client(session=retry_session)

        self.URLS = {
            "history_weather": "https://archive-api.open-meteo.com/v1/archive",
            "history_flood": "https://flood-api.open-meteo.com/v1/flood",
        }

    def _fetch_api(self, url, lat, lon, params, variables):
        """
        Helper function พร้อมระบบ Retry อัตโนมัติเมื่อเจอ Rate Limit
        """
        full_params = {
            "latitude": lat,
            "longitude": lon,
            "daily": variables,
            "timezone": self.timezone,
            **params 
        }

        max_retries = 3
        for attempt in range(max_retries):
            try:
                # หน่วงเวลาพื้นฐาน 2 วินาที เพื่อไม่ให้ยิงรัวเกินไป
                time.sleep(2)
                
                responses = self.client.weather_api(url, params=full_params)
                response = responses[0]
                
                # Process Daily Data
                daily = response.Daily()
                data = {"date": pd.date_range(
                    start=pd.to_datetime(daily.Time(), unit="s", utc=True),
                    end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
                    freq=pd.Timedelta(seconds=daily.Interval()),
                    inclusive="left"
                )}

                for i, var in enumerate(variables):
                    data[var] = daily.Variables(i).ValuesAsNumpy()

                df = pd.DataFrame(data)
                df['date'] = df['date'].dt.tz_convert(self.timezone)
                
                return df

            except Exception as e:
                error_msg = str(e).lower()
                print(f"⚠️ Error fetching (Attempt {attempt+1}/{max_retries}): {e}")
                
                # เช็คว่าเป็น Error เรื่อง Limit หรือไม่
                if "limit exceeded" in error_msg or "429" in error_msg:
                    print("⏳ API Limit hit! Waiting 65 seconds before retrying...")
                    time.sleep(65) # รอ 1 นาทีเศษๆ ตามที่ API แนะนำ
                else:
                    # ถ้าเป็น Error อื่น ให้รอนิดหน่อยแล้วลองใหม่
                    time.sleep(5)
        
        print(f"❌ Failed to fetch data for ({lat}, {lon}) after {max_retries} attempts.")
        return pd.DataFrame()

    def get_historical_data_multi(self, locations, start_date, end_date, weather_vars, flood_vars):
        all_data_frames = []
        common_params = {"start_date": start_date, "end_date": end_date}

        print(f"--- Fetching Data ({start_date} to {end_date}) ---")

        for loc in locations:
            name, lat, lon = loc['name'], loc['lat'], loc['lon']
            print(f"  > Processing: {name}")

            # 1. Fetch Weather
            df_weather = self._fetch_api(
                self.URLS["history_weather"], lat, lon, common_params, weather_vars
            )

            # 2. Fetch Flood
            df_flood = self._fetch_api(
                self.URLS["history_flood"], lat, lon, common_params, flood_vars
            )

            # 3. Merge
            if not df_weather.empty and not df_flood.empty:
                merged_df = pd.merge(df_weather, df_flood, on="date", how="inner")
                merged_df['location'] = name
                all_data_frames.append(merged_df)
                print(f"    ✅ Success: {len(merged_df)} rows.")
            else:
                print(f"    ❌ Failed: Incomplete data for {name}")

        if all_data_frames:
            return pd.concat(all_data_frames, ignore_index=True)

        return pd.DataFrame()

# ----------------------------------------------------------------------
# Main Execution Flow
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Define Locations
    LOCATIONS1 = [{"name": "ChaoPhraya_Dam", "lat": 15.159213709346405, "lon": 100.17985248529882}]
    LOCATIONS2 = [{"name": "NakhonSawan_Muang_Upstream", "lat": 15.700409309316225, "lon": 100.14120663110944}]

    # 2. Define Variables
    WEATHER_VARS = ["temperature_2m_min", "apparent_temperature_mean", "apparent_temperature_max", "apparent_temperature_min", "temperature_2m_max", "temperature_2m_mean", "daylight_duration", "sunshine_duration", "precipitation_sum", "rain_sum", "snowfall_sum", "precipitation_hours", "wind_speed_10m_max", "wind_direction_10m_dominant", "shortwave_radiation_sum", "et0_fao_evapotranspiration", "soil_moisture_0_to_100cm_mean", "soil_moisture_0_to_7cm_mean", "soil_moisture_28_to_100cm_mean", "soil_moisture_7_to_28cm_mean", "soil_temperature_0_to_100cm_mean", "soil_temperature_0_to_7cm_mean", "soil_temperature_28_to_100cm_mean", "soil_temperature_7_to_28cm_mean", "wet_bulb_temperature_2m_mean", "wet_bulb_temperature_2m_max", "wet_bulb_temperature_2m_min", "vapour_pressure_deficit_max", "surface_pressure_mean", "surface_pressure_max", "surface_pressure_min", "winddirection_10m_dominant", "wind_gusts_10m_max", "wind_gusts_10m_mean", "wind_speed_10m_mean", "wind_gusts_10m_min", "wind_speed_10m_min", "pressure_msl_min", "pressure_msl_max", "pressure_msl_mean", "snowfall_water_equivalent_sum", "relative_humidity_2m_max", "relative_humidity_2m_min", "relative_humidity_2m_mean", "et0_fao_evapotranspiration_sum", "dew_point_2m_min", "dew_point_2m_max", "dew_point_2m_mean", "cloud_cover_mean", "cloud_cover_max", "cloud_cover_min"]
    FLOOD_VARS = ["river_discharge"]

    # 3. Initialize Loader
    loader = WeatherLoader(timezone="Asia/Bangkok")
    
    today_str = datetime.now().strftime('%Y-%m-%d')
    print(f"Starting data fetch job... (Target End Date: {today_str})")

    # Fetch Data (ทีละจุด)
    df_chaoPhraya = loader.get_historical_data_multi(LOCATIONS1, "2000-01-01", today_str, WEATHER_VARS, FLOOD_VARS)
    df_NakhonSawan = loader.get_historical_data_multi(LOCATIONS2, "2000-01-01", today_str, WEATHER_VARS, FLOOD_VARS)

    # 4. Merge and Save
    dfs_to_merge = [df for df in [df_chaoPhraya, df_NakhonSawan] if not df.empty]
    
    if len(dfs_to_merge) == 2: # ต้องครบ 2 จุดถึงจะไปต่อได้
        df_final_merged = pd.concat(dfs_to_merge, ignore_index=True)
        print(f"\nTotal rows: {len(df_final_merged)}")

        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        output_dir = os.path.join(project_root, "data", "raw")
        output_file = os.path.join(output_dir, "raw_data.csv")
        
        os.makedirs(output_dir, exist_ok=True)
        df_final_merged.to_csv(output_file, index=False)
        print(f"✅ Data successfully saved to: {output_file}")
    else:
        print("❌ Critical Error: Data incomplete (Need both locations). Pipeline stopped.")
        exit(1) # สั่งให้ GitHub Action แดงทันที