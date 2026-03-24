import numpy as np

def create_windows(data, input_len=24, horizon=6):
    X, Y = [], []

    for i in range(len(data) - input_len - horizon + 1):
        X.append(data[i:i+input_len])
        Y.append(data[i+input_len:i+input_len+horizon, :, 0])  # PM2.5 only

    return np.array(X), np.array(Y)


if __name__ == "__main__":
    data = np.load("../data/processed/data_tensor.npy")
    X, Y = create_windows(data)

    np.save("../data/processed/X.npy", X)
    np.save("../data/processed/Y.npy", Y)

    print("X shape:", X.shape)
    print("Y shape:", Y.shape)