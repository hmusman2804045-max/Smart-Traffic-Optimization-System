import pandas as pd
import numpy as np
import datetime
import os

def generate_traffic_data(output_file="traffic_data.csv", days=30):
    print(f" Generating synthetic traffic data for {days} days...")
    start_date = datetime.datetime(2024, 1, 1)
    date_list = [start_date + datetime.timedelta(hours=x) for x in range(days * 24)]
    data = []
    for dt in date_list:
        hour = dt.hour
        is_weekend = dt.weekday() >= 5
        if 7 <= hour <= 9 or 16 <= hour <= 19:
            base_density = np.random.uniform(60, 90)
        elif 0 <= hour <= 5:
            base_density = np.random.uniform(5, 15)
        else:
            base_density = np.random.uniform(30, 50)
        if is_weekend:
            base_density *= 0.7
        noise = np.random.normal(0, 5)
        final_density = max(0, base_density + noise)
        data.append({
            "timestamp": dt,
            "vehicle_count": round(final_density, 2)
        })
    df = pd.DataFrame(data)
    df.loc[100, "vehicle_count"] = 150.0
    df.loc[250, "vehicle_count"] = 0.0
    df.loc[400, "vehicle_count"] = 85.0
    df.to_csv(output_file, index=False)
    print(f" Synthetic data saved to: {output_file}")
    print(f" Total records: {len(df)}")
    return df

if __name__ == "__main__":
    generate_traffic_data()
