import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime, timedelta

# ── Logging Setup ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────
START_DATE = datetime(2024, 1, 1)
END_DATE   = datetime(2024, 6, 30)
INTERVAL_MINUTES = 1
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

# Facility zones — mimicking a real pharma/medtech facility
ZONES = {
    "Z001": "RnD_Laboratory",
    "Z002": "Suture_Manufacturing",
    "Z003": "Wound_Closure_Bay",
    "Z004": "Sterile_Packaging",
    "Z005": "Cold_Storage_A",
    "Z006": "Cold_Storage_B",
    "Z007": "Quality_Control_Lab",
    "Z008": "Assembly_Floor",
    "Z009": "Cleanroom_Class_A",
    "Z010": "Warehouse_Dispatch"
}

# Normal operating ranges per zone
ZONE_PROFILES = {
    "Z001": {"temp": (20, 24),   "humidity": (40, 55), "diff_pressure": (0.010, 0.020)},
    "Z002": {"temp": (18, 22),   "humidity": (35, 50), "diff_pressure": (0.008, 0.018)},
    "Z003": {"temp": (19, 23),   "humidity": (38, 52), "diff_pressure": (0.009, 0.019)},
    "Z004": {"temp": (15, 20),   "humidity": (30, 45), "diff_pressure": (0.012, 0.022)},
    "Z005": {"temp": (2,  8),    "humidity": (60, 75), "diff_pressure": (0.005, 0.015)},
    "Z006": {"temp": (2,  8),    "humidity": (60, 75), "diff_pressure": (0.005, 0.015)},
    "Z007": {"temp": (20, 24),   "humidity": (40, 55), "diff_pressure": (0.010, 0.020)},
    "Z008": {"temp": (18, 26),   "humidity": (35, 60), "diff_pressure": (0.007, 0.017)},
    "Z009": {"temp": (20, 22),   "humidity": (30, 40), "diff_pressure": (0.020, 0.035)},
    "Z010": {"temp": (15, 28),   "humidity": (30, 65), "diff_pressure": (0.005, 0.012)},
}

# 3 sensors per zone = 30 sensors total
SENSORS_PER_ZONE = 3

# ── Helper Functions ───────────────────────────────────────────
def generate_sensor_ids(zone_id, count):
    return [f"{zone_id}_S{str(i).zfill(2)}" for i in range(1, count + 1)]

def inject_anomalies(df, zone_id):
    profile = ZONE_PROFILES[zone_id]
    n = len(df)

    # 1. Random spikes — 0.5% of readings
    spike_idx = np.random.choice(n, size=int(n * 0.005), replace=False)
    df.loc[spike_idx, "temperature"]    += np.random.uniform(5, 15, size=len(spike_idx))
    df.loc[spike_idx, "humidity"]       += np.random.uniform(10, 25, size=len(spike_idx))

    # 2. Nulls — 1% of readings across all three parameters
    for col in ["temperature", "humidity", "diff_pressure"]:
        null_idx = np.random.choice(n, size=int(n * 0.01), replace=False)
        df.loc[null_idx, col] = np.nan

    # 3. Duplicates — duplicate 0.3% of rows
    dup_idx = np.random.choice(n, size=int(n * 0.003), replace=False)
    duplicates = df.iloc[dup_idx].copy()
    df = pd.concat([df, duplicates], ignore_index=True)

    # 4. Sensor drift — gradual temperature increase over last 30 days
    drift_mask = df["timestamp"] >= (END_DATE - timedelta(days=30))
    df.loc[drift_mask, "temperature"] += np.linspace(0, 2.5, drift_mask.sum())

    # 5. Sensor offline — one sensor goes offline for 48 hours
    offline_sensor = f"{zone_id}_S01"
    offline_start  = datetime(2024, 3, 15, 2, 0, 0)
    offline_end    = datetime(2024, 3, 17, 2, 0, 0)
    offline_mask   = (
        (df["sensor_id"] == offline_sensor) &
        (df["timestamp"] >= offline_start) &
        (df["timestamp"] <= offline_end)
    )
    df = df[~offline_mask]

    return df

def generate_zone_data(zone_id, zone_name):
    logger.info(f"Generating data for zone: {zone_name} ({zone_id})")
    profile   = ZONE_PROFILES[zone_id]
    sensor_ids = generate_sensor_ids(zone_id, SENSORS_PER_ZONE)

    # Generate full timestamp range
    timestamps = pd.date_range(start=START_DATE, end=END_DATE, freq="1min")
    records    = []

    for sensor_id in sensor_ids:
        temp    = np.random.uniform(*profile["temp"],         size=len(timestamps))
        hum     = np.random.uniform(*profile["humidity"],     size=len(timestamps))
        diff_p  = np.random.uniform(*profile["diff_pressure"],size=len(timestamps))

        sensor_df = pd.DataFrame({
            "timestamp":      timestamps,
            "zone_id":        zone_id,
            "zone_name":      zone_name,
            "sensor_id":      sensor_id,
            "temperature":    np.round(temp,   2),
            "humidity":       np.round(hum,    2),
            "diff_pressure":  np.round(diff_p, 4),
            "unit_temp":      "Celsius",
            "unit_humidity":  "Percent",
            "unit_pressure":  "Bar",
            "data_source":    "IoT_Sensor_v2",
            "ingestion_date": datetime.today().strftime("%Y-%m-%d")
        })
        records.append(sensor_df)

    zone_df = pd.concat(records, ignore_index=True)
    zone_df = inject_anomalies(zone_df, zone_id)
    zone_df = zone_df.sort_values("timestamp").reset_index(drop=True)

    return zone_df

# ── Main Execution ─────────────────────────────────────────────
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    total_rows = 0

    for zone_id, zone_name in ZONES.items():
        df = generate_zone_data(zone_id, zone_name)
        output_path = os.path.join(OUTPUT_DIR, f"{zone_id}_{zone_name}.csv")
        df.to_csv(output_path, index=False)
        total_rows += len(df)
        logger.info(f"Saved {len(df):,} rows → {output_path}")

    logger.info(f"✅ Data generation complete. Total rows: {total_rows:,}")

if __name__ == "__main__":
    main()