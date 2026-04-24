import numpy as np
import argparse
import os
import joblib

# Future meteorological feature indices (from the 33-feature space):
# temp(6), pres(7), dewp(8), rain(9), wspm(10) + wind direction one-hot (17-32)
# Excludes PM2.5 and co-pollutants (0-5) — those are what we predict / not forecastable.
# Excludes temporal encodings (11-16) — deterministic from timestamp, not oracle info.
FUTURE_MET_INDICES = [6, 7, 8, 9, 10] + list(range(17, 33))  # 21 features

# Chinese New Year dates for the years covered by the Beijing dataset (2013-2017).
# 2013 CNY (Feb 10) falls before data start (March 2013) so it is excluded.
_CNY_DATES = ['2014-01-31', '2015-02-19', '2016-02-08', '2017-01-28']


def compute_holiday_feature(timestamps):
    """Binary indicator for high-emission Chinese holiday periods.

    Marks three event types with strong PM2.5 anomalies:
      - Spring Festival fireworks window: CNY eve + day + 3 days after (5 days)
      - Golden Week (National Day):       Oct 1–7
      - Labour Day:                       May 1–3

    Args:
        timestamps: list of timestamp strings (from metadata.save)

    Returns:
        (T,) float32 array, 1.0 on holiday hours, 0.0 otherwise
    """
    import pandas as pd

    spring_festival = set()
    for cny_str in _CNY_DATES:
        cny = pd.Timestamp(cny_str)
        for delta in range(-1, 4):  # eve, day, +3 days after
            spring_festival.add((cny + pd.Timedelta(days=delta)).date())

    result = np.zeros(len(timestamps), dtype=np.float32)
    for i, ts in enumerate(timestamps):
        dt = pd.Timestamp(ts)
        if ((dt.month == 10 and 1 <= dt.day <= 7) or   # Golden Week
                (dt.month == 5 and 1 <= dt.day <= 3) or    # Labour Day
                dt.date() in spring_festival):              # Spring Festival
            result[i] = 1.0
    return result


def create_windows(data, input_len=24, horizon=6, future_met_indices=None, add_pm25_delta=False):
    T, N, F = data.shape
    X, Y, Z = [], [], []

    for i in range(T - input_len - horizon + 1):
        x_window = data[i:i+input_len]  # (input_len, N, F)

        if add_pm25_delta:
            # First-difference of PM2.5 (index 0) along the time axis.
            # For the first timestep of the window, use the preceding sample if
            # available (i > 0), otherwise fill with 0.
            prev = data[i-1:i, :, 0] if i > 0 else np.zeros((1, N), dtype=data.dtype)
            pm25 = np.concatenate([prev, x_window[:, :, 0]], axis=0)  # (input_len+1, N)
            delta = np.diff(pm25, axis=0)[:, :, np.newaxis].astype(np.float32)  # (input_len, N, 1)
            # Insert at index 17 (after 6 temporal cyclical features, before wind one-hot).
            # Wind one-hot indices shift from [17:33] to [18:34].
            x_window = np.concatenate([x_window[:, :, :17], delta, x_window[:, :, 17:]], axis=2)

        X.append(x_window)
        Y.append(data[i+input_len:i+input_len+horizon, :, 0])  # PM2.5 only
        if future_met_indices is not None:
            Z.append(data[i+input_len:i+input_len+horizon, :, :][:, :, future_met_indices])

    if future_met_indices is not None:
        return np.array(X), np.array(Y), np.array(Z)
    return np.array(X), np.array(Y)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_len", type=int, default=24,
                        help="Lookback window in hours (default: 24)")
    parser.add_argument("--horizon", type=int, default=6,
                        help="Forecast horizon in hours (default: 6)")
    parser.add_argument("--data_path", type=str, default="../data/processed/",
                        help="Path to processed data directory")
    parser.add_argument("--future_met", action="store_true",
                        help="Also extract future meteorological features (Z tensor)")
    parser.add_argument("--add_pm25_delta", action="store_true",
                        help="Insert PM2.5 first-difference as feature at index 17 (shifts wind one-hot to [18:34])")
    parser.add_argument("--add_holiday", action="store_true",
                        help="Insert Chinese holiday indicator as feature at index 17 (shifts wind one-hot to [18:34])")
    parser.add_argument("--save_y_aux", action="store_true",
                        help="Also save Y_aux_{input_len}.npy: future values of features 1-5 "
                             "(PM10, SO2, NO2, CO, O3) at the same horizon steps as Y.")
    args = parser.parse_args()

    if args.add_pm25_delta and args.add_holiday:
        raise ValueError("--add_pm25_delta and --add_holiday cannot be used together (both insert at index 17).")

    tensor_path = os.path.join(args.data_path, "data_tensor.npy")
    data = np.load(tensor_path)
    print(f"Loaded data_tensor: {data.shape}")

    # Insert holiday feature into the data tensor before windowing.
    # Loads timestamps from metadata.save so preproccess.py does not need to be re-run.
    if args.add_holiday:
        meta_path = os.path.join(args.data_path, "metadata.save")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"metadata.save not found at {meta_path}. Run preproccess.py first.")
        metadata = joblib.load(meta_path)
        timestamps = metadata["timestamps"]
        assert len(timestamps) == len(data), (
            f"Timestamp count ({len(timestamps)}) != data length ({len(data)})")
        holiday = compute_holiday_feature(timestamps)  # (T,)
        T, N, F = data.shape
        holiday_3d = np.tile(holiday[:, np.newaxis, np.newaxis], (1, N, 1))  # (T, N, 1)
        data = np.concatenate([data[:, :, :17], holiday_3d, data[:, :, 17:]], axis=2)
        n_hol = int(holiday.sum())
        print(f"Holiday feature inserted at index 17. New shape: {data.shape}")
        print(f"Holiday hours: {n_hol} / {len(timestamps)} ({100*n_hol/len(timestamps):.1f}%)")

    future_met_indices = FUTURE_MET_INDICES if args.future_met else None
    result = create_windows(data, input_len=args.input_len, horizon=args.horizon,
                            future_met_indices=future_met_indices,
                            add_pm25_delta=args.add_pm25_delta)

    if args.future_met:
        X, Y, Z = result
        z_path = os.path.join(args.data_path, f"Z_{args.input_len}.npy")
        np.save(z_path, Z)
        print(f"Z shape: {Z.shape}  -> saved to {z_path}  (future met: {len(FUTURE_MET_INDICES)} features)")
    else:
        X, Y = result

    if args.add_pm25_delta:
        suffix = "_delta"
    elif args.add_holiday:
        suffix = "_holiday"
    else:
        suffix = ""
    x_path = os.path.join(args.data_path, f"X_{args.input_len}{suffix}.npy")
    y_path = os.path.join(args.data_path, f"Y_{args.input_len}{suffix}.npy")
    np.save(x_path, X)
    np.save(y_path, Y)

    print(f"X shape: {X.shape}  -> saved to {x_path}")
    print(f"Y shape: {Y.shape}  -> saved to {y_path}")
    if args.add_pm25_delta or args.add_holiday:
        print("Extra feature inserted at index 17. Wind one-hot now at [18:34].")

    if args.save_y_aux:
        # Y_aux: future values of features 1-5 (PM10, SO2, NO2, CO, O3).
        # Uses the same sliding-window indices as Y — no future leakage.
        # data_tensor is unmodified (no delta/holiday insert shifts features 1-5).
        raw_data = np.load(tensor_path)  # reload original (before any insert)
        T_raw = raw_data.shape[0]
        Y_aux = []
        for i in range(T_raw - args.input_len - args.horizon + 1):
            Y_aux.append(raw_data[i + args.input_len: i + args.input_len + args.horizon, :, 1:6])
        Y_aux = np.array(Y_aux, dtype=np.float32)  # (N, horizon, nodes, 5)
        y_aux_path = os.path.join(args.data_path, f"Y_aux_{args.input_len}.npy")
        np.save(y_aux_path, Y_aux)
        print(f"Y_aux shape: {Y_aux.shape}  -> saved to {y_aux_path}  "
              f"(features: PM10, SO2, NO2, CO, O3)")
