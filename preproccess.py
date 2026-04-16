import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import joblib

RAW_PATH = "data/raw/beijing/"
PROCESSED_PATH = "data/processed/"

# Station order must match adjacency matrix order
STATION_ORDER = [
    "Aotizhongxin", "Changping", "Dingling", "Dongsi", "Guanyuan",
    "Gucheng", "Huairou", "Nongzhanguan", "Shunyi", "Tiantan",
    "Wanliu", "Wanshouxigong"
]


def load_and_merge():
    dfs = []

    for file in os.listdir(RAW_PATH):
        if file.endswith(".csv"):
            print(f"Loading {file} ...")
            df = pd.read_csv(os.path.join(RAW_PATH, file))

            df.columns = df.columns.str.lower()

            # Convert datetime column
            df["datetime"] = pd.to_datetime(df["datetime"])

            # Rename station column
            df.rename(columns={"station": "station_id"}, inplace=True)

            # Drop unused columns
            if "no" in df.columns:
                df.drop(columns=["no"], inplace=True)

            dfs.append(df)

    merged_df = pd.concat(dfs, axis=0)
    merged_df.sort_values(["station_id", "datetime"], inplace=True)
    merged_df.reset_index(drop=True, inplace=True)
    return merged_df


def clean_missing(df):
    numeric_cols = df.select_dtypes(include=np.number).columns

    # Interpolate per station
    df[numeric_cols] = (
        df.groupby("station_id")[numeric_cols]
        .transform(lambda x: x.interpolate(limit_direction='both'))
    )

    # Backfill / forward fill remaining NaNs
    df[numeric_cols] = (
        df.groupby("station_id")[numeric_cols]
        .transform(lambda x: x.bfill().ffill())
    )

    # Fill wind direction mode per station
    df["wd"] = (
        df.groupby("station_id")["wd"]
        .transform(lambda x: x.fillna(x.mode().iloc[0] if not x.mode().empty else "N"))
    )

    return df


def add_cyclical_features(df):
    """Add cyclical encoding for temporal features (better for neural networks)."""
    hour = df["datetime"].dt.hour
    month = df["datetime"].dt.month
    weekday = df["datetime"].dt.weekday

    # Cyclical encoding using sin/cos
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)
    df["weekday_sin"] = np.sin(2 * np.pi * weekday / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * weekday / 7)

    return df


def encode_wind_direction(df):
    """One-hot encode wind direction only."""
    df = pd.get_dummies(df, columns=["wd"], prefix="wd")
    return df


def create_tensor(df):
    """
    Pivot dataframe to 3D tensor: (timesteps, num_stations, features)
    PM2.5 is placed at index 0 for easy target extraction.
    """
    # Get common timestamps across all stations
    timestamp_counts = df.groupby("datetime").size()
    common_timestamps = timestamp_counts[timestamp_counts == len(STATION_ORDER)].index
    timestamps = sorted(common_timestamps)

    print(f"Common timestamps across all stations: {len(timestamps)}")

    # Filter to only common timestamps
    df = df[df["datetime"].isin(timestamps)].copy()

    # Define feature columns (PM2.5 first!)
    target_col = ["pm2.5"]
    
    # Other pollutants and meteorological features
    pollutant_cols = ["pm10", "so2", "no2", "co", "o3"]
    meteo_cols = ["temp", "pres", "dewp", "rain", "wspm"]
    temporal_cols = ["hour_sin", "hour_cos", "month_sin", "month_cos", "weekday_sin", "weekday_cos"]
    wind_cols = sorted([c for c in df.columns if c.startswith("wd_")])

    feature_cols = target_col + pollutant_cols + meteo_cols + temporal_cols + wind_cols
    
    # Verify all columns exist
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        print(f"Warning: Missing columns {missing_cols}, skipping them")
        feature_cols = [c for c in feature_cols if c in df.columns]

    print(f"Feature columns ({len(feature_cols)}): {feature_cols[:5]}...")

    # Create mapping for station indices
    station_to_idx = {s: i for i, s in enumerate(STATION_ORDER)}
    
    # Add station index column
    df["station_idx"] = df["station_id"].map(station_to_idx)
    
    # Create timestamp index mapping
    timestamp_to_idx = {t: i for i, t in enumerate(timestamps)}
    df["time_idx"] = df["datetime"].map(timestamp_to_idx)

    # Create tensor efficiently using vectorized operations
    T = len(timestamps)
    N = len(STATION_ORDER)
    F = len(feature_cols)

    tensor = np.zeros((T, N, F), dtype=np.float32)
    
    # Sort by time_idx and station_idx for efficient assignment
    df_sorted = df.sort_values(["time_idx", "station_idx"])
    
    # Extract feature values as numpy array
    feature_values = df_sorted[feature_cols].values.astype(np.float32)
    time_indices = df_sorted["time_idx"].values
    station_indices = df_sorted["station_idx"].values
    
    # Vectorized assignment
    tensor[time_indices, station_indices, :] = feature_values

    return tensor, feature_cols, timestamps


def normalize_tensor(tensor, feature_cols):
    """
    NOTE: This function is deprecated for training.
    Scaling should be done in train.py to avoid data leakage.

    This function is kept for reference/debugging only.
    Returns the UNSCALED tensor and None scalers.
    """
    # Return unscaled tensor - scaling will be done in train.py
    # to prevent data leakage (test data stats leaking into training)
    return tensor, None, None


def check_missing_hours(timestamps):
    """Check for gaps in the time series."""
    timestamps = pd.to_datetime(timestamps)
    expected = pd.date_range(timestamps.min(), timestamps.max(), freq="h")
    missing = expected.difference(timestamps)
    if len(missing) > 0:
        print(f"[Warning] Missing {len(missing)} hours in time series")
        print(f"  First missing: {missing[:5].tolist()}")
    else:
        print("No missing hours in aligned time series.")


def save_outputs(tensor, feature_cols, timestamps, target_scaler, feature_scaler):
    """Save tensor and metadata."""
    os.makedirs(PROCESSED_PATH, exist_ok=True)

    # Save tensor
    np.save(os.path.join(PROCESSED_PATH, "data_tensor.npy"), tensor)
    print(f"Tensor saved: {tensor.shape} (T, N, F)")

    # Save scalers
    joblib.dump(target_scaler, os.path.join(PROCESSED_PATH, "target_scaler.save"))
    if feature_scaler:
        joblib.dump(feature_scaler, os.path.join(PROCESSED_PATH, "feature_scaler.save"))

    # Save metadata
    metadata = {
        "feature_cols": feature_cols,
        "station_order": STATION_ORDER,
        "timestamps": [str(t) for t in timestamps],
        "shape": tensor.shape
    }
    joblib.dump(metadata, os.path.join(PROCESSED_PATH, "metadata.save"))

    print(f"All outputs saved to {PROCESSED_PATH}")


if __name__ == "__main__":
    print("=" * 50)
    print("GCN+LSTM Preprocessing Pipeline")
    print("=" * 50)

    print("\n[1/6] Loading and merging CSVs...")
    df = load_and_merge()
    print(f"  Merged shape: {df.shape}")

    print("\n[2/6] Cleaning missing values...")
    df = clean_missing(df)
    print(f"  Remaining NaNs: {df.isnull().sum().sum()}")

    print("\n[3/6] Adding cyclical temporal features...")
    df = add_cyclical_features(df)

    print("\n[4/6] Encoding wind direction...")
    df = encode_wind_direction(df)

    print("\n[5/6] Creating 3D tensor (T, N, F)...")
    tensor, feature_cols, timestamps = create_tensor(df)
    print(f"  Tensor shape: {tensor.shape}")

    print("\n[6/6] Normalizing...")
    tensor, target_scaler, feature_scaler = normalize_tensor(tensor, feature_cols)

    print("\nChecking for missing hours...")
    check_missing_hours(timestamps)

    print("\nSaving outputs...")
    save_outputs(tensor, feature_cols, timestamps, target_scaler, feature_scaler)

    print("\n" + "=" * 50)
    print("Preprocessing complete!")
    print(f"  Output: data_tensor.npy {tensor.shape}")
    print(f"  Stations: {len(STATION_ORDER)}")
    print(f"  Features: {len(feature_cols)} (PM2.5 at index 0)")
    print(f"  Timesteps: {len(timestamps)}")
    print("=" * 50)