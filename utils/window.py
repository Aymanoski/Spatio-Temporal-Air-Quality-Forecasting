import numpy as np
import argparse
import os

def create_windows(data, input_len=24, horizon=6):
    X, Y = [], []

    for i in range(len(data) - input_len - horizon + 1):
        X.append(data[i:i+input_len])
        Y.append(data[i+input_len:i+input_len+horizon, :, 0])  # PM2.5 only

    return np.array(X), np.array(Y)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_len", type=int, default=24,
                        help="Lookback window in hours (default: 24)")
    parser.add_argument("--horizon", type=int, default=6,
                        help="Forecast horizon in hours (default: 6)")
    parser.add_argument("--data_path", type=str, default="../data/processed/",
                        help="Path to processed data directory")
    args = parser.parse_args()

    tensor_path = os.path.join(args.data_path, "data_tensor.npy")
    data = np.load(tensor_path)
    print(f"Loaded data_tensor: {data.shape}")

    X, Y = create_windows(data, input_len=args.input_len, horizon=args.horizon)

    # Save with input_len in filename so multiple window sizes can coexist
    x_path = os.path.join(args.data_path, f"X_{args.input_len}.npy")
    y_path = os.path.join(args.data_path, f"Y_{args.input_len}.npy")
    np.save(x_path, X)
    np.save(y_path, Y)

    print(f"X shape: {X.shape}  -> saved to {x_path}")
    print(f"Y shape: {Y.shape}  -> saved to {y_path}")