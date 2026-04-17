import numpy as np
import argparse
import os

# Future meteorological feature indices (from the 33-feature space):
# temp(6), pres(7), dewp(8), rain(9), wspm(10) + wind direction one-hot (17-32)
# Excludes PM2.5 and co-pollutants (0-5) — those are what we predict / not forecastable.
# Excludes temporal encodings (11-16) — deterministic from timestamp, not oracle info.
FUTURE_MET_INDICES = [6, 7, 8, 9, 10] + list(range(17, 33))  # 21 features


def create_windows(data, input_len=24, horizon=6, future_met_indices=None):
    X, Y, Z = [], [], []

    for i in range(len(data) - input_len - horizon + 1):
        X.append(data[i:i+input_len])
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
    args = parser.parse_args()

    tensor_path = os.path.join(args.data_path, "data_tensor.npy")
    data = np.load(tensor_path)
    print(f"Loaded data_tensor: {data.shape}")

    future_met_indices = FUTURE_MET_INDICES if args.future_met else None
    result = create_windows(data, input_len=args.input_len, horizon=args.horizon,
                            future_met_indices=future_met_indices)

    if args.future_met:
        X, Y, Z = result
        z_path = os.path.join(args.data_path, f"Z_{args.input_len}.npy")
        np.save(z_path, Z)
        print(f"Z shape: {Z.shape}  -> saved to {z_path}  (future met: {len(FUTURE_MET_INDICES)} features)")
    else:
        X, Y = result

    x_path = os.path.join(args.data_path, f"X_{args.input_len}.npy")
    y_path = os.path.join(args.data_path, f"Y_{args.input_len}.npy")
    np.save(x_path, X)
    np.save(y_path, Y)

    print(f"X shape: {X.shape}  -> saved to {x_path}")
    print(f"Y shape: {Y.shape}  -> saved to {y_path}")
