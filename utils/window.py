import numpy as np
import argparse
import os

# Future meteorological feature indices (from the 33-feature space):
# temp(6), pres(7), dewp(8), rain(9), wspm(10) + wind direction one-hot (17-32)
# Excludes PM2.5 and co-pollutants (0-5) — those are what we predict / not forecastable.
# Excludes temporal encodings (11-16) — deterministic from timestamp, not oracle info.
FUTURE_MET_INDICES = [6, 7, 8, 9, 10] + list(range(17, 33))  # 21 features


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
    args = parser.parse_args()

    tensor_path = os.path.join(args.data_path, "data_tensor.npy")
    data = np.load(tensor_path)
    print(f"Loaded data_tensor: {data.shape}")

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

    suffix = "_delta" if args.add_pm25_delta else ""
    x_path = os.path.join(args.data_path, f"X_{args.input_len}{suffix}.npy")
    y_path = os.path.join(args.data_path, f"Y_{args.input_len}{suffix}.npy")
    np.save(x_path, X)
    np.save(y_path, Y)

    print(f"X shape: {X.shape}  -> saved to {x_path}")
    print(f"Y shape: {Y.shape}  -> saved to {y_path}")
    if args.add_pm25_delta:
        print("Delta feature inserted at index 17. Wind one-hot now at [18:34].")
