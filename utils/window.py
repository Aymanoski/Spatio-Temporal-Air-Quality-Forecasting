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


def compute_met_derived_features(data):
    """
    Compute dew point depression (T-DEWP) and vapor pressure deficit (VPD) from raw data.

    Both features are derived from TEMP (index 6, Celsius) and DEWP (index 8, Celsius)
    before any normalisation so the physical relationships are preserved.

    VPD uses the Tetens equation: es(T) = 6.112 * exp(17.67*T / (T+243.5)) [hPa].

    Returns:
        (T, N, 2) float32 array: [T-DEWP, VPD]
    """
    temp = data[:, :, 6].astype(np.float64)  # (T, N)
    dewp = data[:, :, 8].astype(np.float64)  # (T, N)

    t_minus_dewp = (temp - dewp).astype(np.float32)

    es = 6.112 * np.exp(17.67 * temp / (temp + 243.5))
    ea = 6.112 * np.exp(17.67 * dewp / (dewp + 243.5))
    vpd = (es - ea).astype(np.float32)

    return np.stack([t_minus_dewp, vpd], axis=-1)  # (T, N, 2)


def create_windows(data, input_len=24, horizon=6, future_met_indices=None,
                   add_pm25_delta=False, add_wavelet=False):
    """Sliding-window feature extraction.

    Wavelet note: if add_wavelet=True, SWT is computed *per window* on the
    24-sample PM2.5 signal so that no future values can enter the computation.
    Uses db2 (filter length 4) which is the longest db-family wavelet that
    fits 24 samples at level 3: floor(log2(24/(4-1))) = floor(log2(8)) = 3.
    Appends 4 channels [cA3, cD3, cD2, cD1] at the end of the feature axis.
    """
    if add_wavelet:
        import pywt
        _wav, _level = 'db2', 3  # db4 needs 2^3*(8-1)=56 samples; db2 needs 2^3*(4-1)=24 ✓

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

        if add_wavelet:
            # Per-window SWT: only sees data[i:i+input_len] (pure past) → zero leakage.
            # trim_approx=True → [cA_level, cD_level, ..., cD_1], each length input_len.
            pm25_win = x_window[:, :, 0].astype(np.float64)  # (input_len, N)
            wav_list = []
            for n in range(N):
                coeffs = pywt.swt(pm25_win[:, n], _wav, level=_level, trim_approx=True)
                wav_list.append(np.stack(coeffs, axis=-1).astype(np.float32))  # (input_len, 4)
            wav_arr = np.stack(wav_list, axis=1)  # (input_len, N, 4)
            x_window = np.concatenate([x_window, wav_arr], axis=2)

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
    parser.add_argument("--add_met_derived", action="store_true",
                        help="Insert T-DEWP and VPD as features at indices 17-18 (shifts wind one-hot to [19:35])")
    parser.add_argument("--add_wavelet", action="store_true",
                        help="Append 4 SWT channels of PM2.5 (cA3, cD3, cD2, cD1) at indices 33-36. "
                             "input_dim becomes 37. Requires pywt (pip install PyWavelets).")
    parser.add_argument("--save_y_aux", action="store_true",
                        help="Also save Y_aux_{input_len}.npy: future values of features 1-5 "
                             "(PM10, SO2, NO2, CO, O3) at the same horizon steps as Y.")
    args = parser.parse_args()

    exclusive = sum([args.add_pm25_delta, args.add_holiday, args.add_met_derived])
    if exclusive > 1:
        raise ValueError("--add_pm25_delta, --add_holiday, and --add_met_derived cannot be combined (all modify the same insertion region).")

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

    # Insert met-derived features (T-DEWP, VPD) at indices 17-18 before windowing.
    if args.add_met_derived:
        derived = compute_met_derived_features(data)  # (T, N, 2)
        data = np.concatenate([data[:, :, :17], derived, data[:, :, 17:]], axis=2)
        print(f"Met-derived features (T-DEWP, VPD) inserted at indices 17-18. New shape: {data.shape}")
        print(f"  T-DEWP range: [{derived[:,:,0].min():.2f}, {derived[:,:,0].max():.2f}]°C")
        print(f"  VPD range:    [{derived[:,:,1].min():.2f}, {derived[:,:,1].max():.2f}] hPa")

    future_met_indices = FUTURE_MET_INDICES if args.future_met else None
    if args.add_wavelet:
        print("Wavelet mode: SWT(db2, level=3) computed per window — no future leakage.")
        print("  4 channels [cA3, cD3, cD2, cD1] appended at indices 33-36. input_dim → 37.")
    result = create_windows(data, input_len=args.input_len, horizon=args.horizon,
                            future_met_indices=future_met_indices,
                            add_pm25_delta=args.add_pm25_delta,
                            add_wavelet=args.add_wavelet)

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
    elif args.add_met_derived:
        suffix = "_metderived"
    elif args.add_wavelet:
        suffix = "_wavelet"
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
    elif args.add_met_derived:
        print("Two features inserted at indices 17-18. Wind one-hot now at [19:35].")

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
