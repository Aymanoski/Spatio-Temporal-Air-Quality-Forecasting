import numpy as np
from math import radians, sin, cos, sqrt, atan2
import torch

# Manually define station coordinates (Beijing dataset)
STATIONS = {
    "Aotizhongxin": (39.982, 116.397),
    "Changping": (40.217, 116.231),
    "Dingling": (40.292, 116.220),
    "Dongsi": (39.929, 116.417),
    "Guanyuan": (39.933, 116.356),
    "Gucheng": (39.914, 116.184),
    "Huairou": (40.357, 116.631),
    "Nongzhanguan": (39.937, 116.461),
    "Shunyi": (40.127, 116.655),
    "Tiantan": (39.886, 116.407),
    "Wanliu": (39.987, 116.287),
    "Wanshouxigong": (39.878, 116.352)
}

# Wind direction mapping (Beijing dataset uses these categories)
# Maps wind direction categories to angles in degrees (0 = North, 90 = East, etc.)
# Alphabetically sorted as pandas.get_dummies would create them
WIND_DIRECTION_MAP = {
    "E": 90, "ENE": 67.5, "ESE": 112.5,
    "N": 0, "NE": 45, "NNE": 22.5, "NNW": 337.5, "NW": 315,
    "S": 180, "SE": 135, "SSE": 157.5, "SSW": 202.5, "SW": 225,
    "W": 270, "WNW": 292.5, "WSW": 247.5
}

# Wind direction categories in alphabetical order (as get_dummies creates them)
WIND_CATEGORIES = sorted(WIND_DIRECTION_MAP.keys())  # For mapping one-hot indices

def haversine(coord1, coord2):
    R = 6371
    lat1, lon1 = radians(coord1[0]), radians(coord1[1])
    lat2, lon2 = radians(coord2[0]), radians(coord2[1])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))

    return R * c


def compute_bearing(coord1, coord2):
    """
    Compute bearing (direction) from coord1 to coord2 in degrees.
    0° = North, 90° = East, 180° = South, 270° = West
    """
    lat1, lon1 = radians(coord1[0]), radians(coord1[1])
    lat2, lon2 = radians(coord2[0]), radians(coord2[1])

    dlon = lon2 - lon1

    x = sin(dlon) * cos(lat2)
    y = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dlon)

    bearing = atan2(x, y)
    bearing = np.degrees(bearing)
    bearing = (bearing + 360) % 360

    return bearing


def wind_direction_to_angle(wind_one_hot, wind_categories):
    """
    Convert one-hot encoded wind direction back to angle in degrees.

    Args:
        wind_one_hot: One-hot encoded wind direction (17 categories)
        wind_categories: List of wind direction category names in order

    Returns:
        angle in degrees (0-360) or -1 for calm/variable
    """
    if isinstance(wind_one_hot, torch.Tensor):
        wind_one_hot = wind_one_hot.cpu().numpy()

    idx = np.argmax(wind_one_hot)
    if idx >= len(wind_categories):
        return -1

    category = wind_categories[idx]
    return WIND_DIRECTION_MAP.get(category, -1)


def angle_to_nearest_category_one_hot(angle, wind_categories):
    """Map a continuous angle to the nearest categorical wind direction (one-hot)."""
    if angle < 0:
        # Calm/variable fallback: uniform-like weak prior on "N"
        one_hot = np.zeros(len(wind_categories), dtype=np.float32)
        if "N" in wind_categories:
            one_hot[wind_categories.index("N")] = 1.0
        return one_hot

    min_diff = float("inf")
    best_idx = 0
    for idx, category in enumerate(wind_categories):
        ref_angle = WIND_DIRECTION_MAP.get(category, 0.0)
        diff = abs(ref_angle - angle)
        if diff > 180:
            diff = 360 - diff
        if diff < min_diff:
            min_diff = diff
            best_idx = idx

    one_hot = np.zeros(len(wind_categories), dtype=np.float32)
    one_hot[best_idx] = 1.0
    return one_hot


def _temporal_weights(timesteps, mode="recent_weighted", recency_beta=3.0):
    """Create temporal aggregation weights for a history window."""
    if timesteps <= 0:
        raise ValueError("timesteps must be > 0")

    if mode == "last":
        w = np.zeros(timesteps, dtype=np.float32)
        w[-1] = 1.0
        return w

    if mode == "mean":
        return np.full(timesteps, 1.0 / timesteps, dtype=np.float32)

    # recent_weighted (default): exponential increase toward latest timestep
    t = np.linspace(0.0, 1.0, timesteps, dtype=np.float32)
    w = np.exp(recency_beta * t)
    w = w / np.sum(w)
    return w.astype(np.float32)


def aggregate_wind_over_time(
    wind_speeds,
    wind_directions,
    wind_categories,
    mode="recent_weighted",
    recency_beta=3.0,
    direction_method="circular",
    calm_speed_threshold=0.1
):
    """
    Aggregate temporal wind inputs for one sample.

    Args:
        wind_speeds: (timesteps, nodes)
        wind_directions: (timesteps, nodes, categories)
        mode: 'mean', 'last', or 'recent_weighted'
        recency_beta: strength of recent-weight emphasis for 'recent_weighted'
        direction_method: 'argmax_mean' or 'circular'
        calm_speed_threshold: threshold under which angle is treated as calm/variable

    Returns:
        agg_speeds: (nodes,)
        agg_dirs_one_hot: (nodes, categories)
        agg_angles: (nodes,)
    """
    timesteps, num_nodes = wind_speeds.shape
    _, _, num_categories = wind_directions.shape

    weights = _temporal_weights(timesteps, mode=mode, recency_beta=recency_beta)

    agg_speeds = np.tensordot(weights, wind_speeds, axes=(0, 0)).astype(np.float32)

    if direction_method == "argmax_mean":
        mean_dirs = np.tensordot(weights, wind_directions, axes=(0, 0))
        agg_dirs_one_hot = np.zeros((num_nodes, num_categories), dtype=np.float32)
        best_idx = np.argmax(mean_dirs, axis=-1)
        agg_dirs_one_hot[np.arange(num_nodes), best_idx] = 1.0
        agg_angles = np.array(
            [wind_direction_to_angle(agg_dirs_one_hot[n], wind_categories) for n in range(num_nodes)],
            dtype=np.float32
        )
        return agg_speeds, agg_dirs_one_hot, agg_angles

    # circular (default): robust angular aggregation with periodicity awareness
    raw_angles = np.zeros((timesteps, num_nodes), dtype=np.float32)
    for t in range(timesteps):
        for n in range(num_nodes):
            raw_angles[t, n] = wind_direction_to_angle(wind_directions[t, n], wind_categories)

    agg_angles = np.zeros(num_nodes, dtype=np.float32)
    agg_dirs_one_hot = np.zeros((num_nodes, num_categories), dtype=np.float32)

    for n in range(num_nodes):
        # Combine temporal recency and wind intensity so strong, recent winds dominate
        node_speeds = np.maximum(wind_speeds[:, n], 0.0)
        dir_weights = weights * (0.5 + node_speeds)
        dir_weights_sum = np.sum(dir_weights)
        if dir_weights_sum <= 0:
            dir_weights = weights
            dir_weights_sum = np.sum(dir_weights)
        dir_weights = dir_weights / dir_weights_sum

        x = 0.0
        y = 0.0
        for t in range(timesteps):
            angle = raw_angles[t, n]
            if angle < 0:
                continue
            rad = np.radians(angle)
            x += dir_weights[t] * np.cos(rad)
            y += dir_weights[t] * np.sin(rad)

        if agg_speeds[n] < calm_speed_threshold or (abs(x) < 1e-6 and abs(y) < 1e-6):
            agg_angles[n] = -1.0
            agg_dirs_one_hot[n] = angle_to_nearest_category_one_hot(-1.0, wind_categories)
        else:
            angle = np.degrees(np.arctan2(y, x))
            angle = (angle + 360.0) % 360.0
            agg_angles[n] = angle
            agg_dirs_one_hot[n] = angle_to_nearest_category_one_hot(angle, wind_categories)

    return agg_speeds, agg_dirs_one_hot, agg_angles


def compute_wind_alignment(wind_angle, bearing, wind_speed):
    """
    Compute alignment factor between wind direction and station-to-station bearing.
    Higher values mean pollution is more likely to be transported from source to target.

    Args:
        wind_angle: Wind direction in degrees (direction wind is blowing FROM)
        bearing: Geographic bearing from station i to station j
        wind_speed: Wind speed (m/s)

    Returns:
        alignment factor (0 to 1)
    """
    if wind_angle < 0 or wind_speed < 0.1:  # Calm or variable wind
        return 0.5  # Neutral influence

    # Wind blows FROM wind_angle, so pollution travels TO (wind_angle + 180) % 360
    transport_direction = (wind_angle + 180) % 360

    # Compute angular difference
    angle_diff = abs(transport_direction - bearing)
    if angle_diff > 180:
        angle_diff = 360 - angle_diff

    # Convert to alignment: 0° diff = 1.0 (perfect alignment), 180° diff = 0.0 (opposite)
    # Use cosine for smooth falloff
    alignment = (np.cos(np.radians(angle_diff)) + 1) / 2

    # Weight by wind speed (stronger wind = stronger transport)
    # Normalize wind speed (typical range 0-20 m/s)
    wind_factor = np.tanh(wind_speed / 5.0)  # Saturates around 10-15 m/s

    return alignment * wind_factor


def compute_receiving_alignment(wind_angle_j, bearing_ij, wind_speed_j):
    """
    Compute how receptive station j is to pollution arriving from station i.

    Checks whether the wind at receiver j is consistent with transport from i
    (i.e., wind at j blows FROM i's direction, meaning air flows from i toward j).

    Args:
        wind_angle_j: Wind direction at station j (degrees, blowing FROM)
        bearing_ij: Geographic bearing from i to j
        wind_speed_j: Wind speed at station j (m/s)

    Returns:
        receiving factor (0 to 1): 1.0 = supportive, 0.0 = blocking
    """
    if wind_angle_j < 0 or wind_speed_j < 0.1:
        return 0.5  # Neutral when calm/variable

    # Bearing from j to i (reverse of i→j)
    bearing_ji = (bearing_ij + 180) % 360

    # If wind at j blows FROM bearing_ji, air flows from i→j at j's location
    # This is supportive for transport from i to j
    angle_diff = abs(wind_angle_j - bearing_ji)
    if angle_diff > 180:
        angle_diff = 360 - angle_diff

    receiving = (np.cos(np.radians(angle_diff)) + 1) / 2
    return receiving


def build_adjacency():
    """Build static distance-based adjacency matrix with self-loops."""
    station_names = list(STATIONS.keys())
    n = len(station_names)

    A = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i != j:
                d = haversine(
                    STATIONS[station_names[i]],
                    STATIONS[station_names[j]]
                )
                A[i, j] = np.exp(-d**2 / 1800)

    # Add self-loops (critical for GCN - nodes must aggregate their own features)
    A = A + np.eye(n)

    # Symmetric normalization: D^{-1/2} (A + I) D^{-1/2}
    D = np.diag(np.sum(A, axis=1))
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D)))
    A_hat = D_inv_sqrt @ A @ D_inv_sqrt

    np.save("../data/processed/adjacency.npy", A_hat)
    print("Adjacency matrix saved.")


def build_wind_aware_adjacency(
    wind_speeds,
    wind_directions,
    wind_categories,
    alpha=0.6,
    distance_sigma=1800,
    normalization="row",
    wind_angles=None
):
    """
    Build dynamic wind-aware directed adjacency matrix with bidirectional alignment.

    Args:
        wind_speeds: (num_nodes,) array of wind speeds at each station
        wind_directions: (num_nodes, num_categories) array of one-hot wind directions
        wind_categories: List of wind direction category names in order
        alpha: Weighting factor (0 to 1).
               alpha=1.0 means pure wind-based,
               alpha=0.0 means pure distance-based,
               alpha=0.6 is a balanced hybrid (recommended)
        distance_sigma: Distance decay parameter for Gaussian kernel
        normalization: 'row' for directed row-stochastic, 'symmetric' for D^(-1/2) A D^(-1/2)
        wind_angles: Optional continuous angles (num_nodes,). If provided, this
                bypasses one-hot conversion and preserves angular fidelity.

    Returns:
        A_hat: (num_nodes, num_nodes) normalized directed adjacency matrix
    """
    station_names = list(STATIONS.keys())
    n = len(station_names)

    # Precompute distance and bearing matrices
    dist_matrix = np.zeros((n, n))
    bearing_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i != j:
                dist_matrix[i, j] = haversine(
                    STATIONS[station_names[i]],
                    STATIONS[station_names[j]]
                )
                bearing_matrix[i, j] = compute_bearing(
                    STATIONS[station_names[i]],
                    STATIONS[station_names[j]]
                )

    # Distance-based component (undirected)
    A_dist = np.exp(-dist_matrix**2 / distance_sigma)
    # Keep diagonal as 1.0 for self-loops (critical for GCN)
    np.fill_diagonal(A_dist, 1.0)

    # Wind-based component (directed, with bidirectional alignment)
    A_wind = np.zeros((n, n))

    for i in range(n):
        if wind_angles is not None:
            wind_angle_i = float(wind_angles[i])
        else:
            wind_angle_i = wind_direction_to_angle(wind_directions[i], wind_categories)
        wind_speed_i = wind_speeds[i]

        for j in range(n):
            if i != j:
                bearing_ij = bearing_matrix[i, j]

                # Source alignment: does wind at i transport pollution toward j?
                source_alignment = compute_wind_alignment(wind_angle_i, bearing_ij, wind_speed_i)

                # Receiver modulation: is j's wind consistent with transport from i?
                if wind_angles is not None:
                    wind_angle_j = float(wind_angles[j])
                else:
                    wind_angle_j = wind_direction_to_angle(wind_directions[j], wind_categories)
                wind_speed_j = wind_speeds[j]
                receiving = compute_receiving_alignment(wind_angle_j, bearing_ij, wind_speed_j)

                # Combined bidirectional alignment
                alignment = source_alignment * (0.5 + 0.5 * receiving)

                # Combine alignment with distance decay
                A_wind[i, j] = alignment * np.exp(-dist_matrix[i, j]**2 / distance_sigma)

    # Add self-loops to wind component (nodes retain their own information)
    np.fill_diagonal(A_wind, 1.0)

    # Hybrid adjacency: combine distance-based and wind-based
    A = (1 - alpha) * A_dist + alpha * A_wind

    # Normalize directed graph. Row-normalization preserves flow direction better.
    row_sum = np.sum(A, axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0

    if normalization == "symmetric":
        deg = row_sum.squeeze(-1)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(deg))
        A_hat = D_inv_sqrt @ A @ D_inv_sqrt
    else:
        A_hat = A / row_sum

    return A_hat


def build_wind_aware_adjacency_batch(
    wind_speeds,
    wind_directions,
    wind_categories,
    alpha=0.6,
    distance_sigma=1800,
    aggregation_mode="recent_weighted",
    recency_beta=3.0,
    direction_method="circular",
    normalization="row",
    calm_speed_threshold=0.1
):
    """
    Build wind-aware adjacency for a batch of timesteps.

    Args:
        wind_speeds: (batch, timesteps, num_nodes) or (batch, num_nodes) tensor
        wind_directions: (batch, timesteps, num_nodes, num_categories) or
                        (batch, num_nodes, num_categories) tensor
        wind_categories: List of wind direction category names
        alpha: Wind influence weight
        distance_sigma: Distance decay parameter

    Returns:
        adjacency_batch: numpy array of adjacency matrices
    """
    if isinstance(wind_speeds, torch.Tensor):
        wind_speeds = wind_speeds.cpu().numpy()
    if isinstance(wind_directions, torch.Tensor):
        wind_directions = wind_directions.cpu().numpy()

    # Handle different input shapes
    if wind_speeds.ndim == 3:  # (batch, timesteps, nodes)
        batch_size, timesteps, num_nodes = wind_speeds.shape
        temporal_input = True
    elif wind_speeds.ndim == 2:  # (batch, nodes)
        batch_size, num_nodes = wind_speeds.shape
        temporal_input = False
    else:
        raise ValueError(f"Unexpected wind_speeds shape: {wind_speeds.shape}")

    adjacency_batch = np.zeros((batch_size, num_nodes, num_nodes))

    for b in range(batch_size):
        if temporal_input:
            agg_speeds, agg_dirs, agg_angles = aggregate_wind_over_time(
                wind_speeds[b],
                wind_directions[b],
                wind_categories,
                mode=aggregation_mode,
                recency_beta=recency_beta,
                direction_method=direction_method,
                calm_speed_threshold=calm_speed_threshold
            )
            adjacency_batch[b] = build_wind_aware_adjacency(
                agg_speeds,
                agg_dirs,
                wind_categories,
                alpha=alpha,
                distance_sigma=distance_sigma,
                normalization=normalization,
                wind_angles=agg_angles
            )
        else:
            adjacency_batch[b] = build_wind_aware_adjacency(
                wind_speeds[b],
                wind_directions[b],
                wind_categories,
                alpha=alpha,
                distance_sigma=distance_sigma,
                normalization=normalization,
                wind_angles=None
            )

    return adjacency_batch


# ============================================================================
# GPU-Optimized Adjacency Computation
# ============================================================================

# Precomputed static matrices (computed once, reused for all batches)
_PRECOMPUTED_CACHE = {}


def _precompute_static_matrices(device):
    """
    Precompute distance decay and bearing matrices on GPU.
    These are static and only need to be computed once.
    """
    cache_key = str(device)
    if cache_key in _PRECOMPUTED_CACHE:
        return _PRECOMPUTED_CACHE[cache_key]

    station_names = list(STATIONS.keys())
    n = len(station_names)

    # Compute distance and bearing matrices
    dist_matrix = np.zeros((n, n), dtype=np.float32)
    bearing_matrix = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        for j in range(n):
            if i != j:
                dist_matrix[i, j] = haversine(
                    STATIONS[station_names[i]],
                    STATIONS[station_names[j]]
                )
                bearing_matrix[i, j] = compute_bearing(
                    STATIONS[station_names[i]],
                    STATIONS[station_names[j]]
                )

    # Convert to tensors on device
    dist_tensor = torch.from_numpy(dist_matrix).to(device)
    bearing_tensor = torch.from_numpy(bearing_matrix).to(device)

    # Precompute wind category angles tensor
    category_angles = torch.tensor(
        [WIND_DIRECTION_MAP.get(cat, -1) for cat in WIND_CATEGORIES],
        dtype=torch.float32,
        device=device
    )

    _PRECOMPUTED_CACHE[cache_key] = {
        'dist': dist_tensor,
        'bearing': bearing_tensor,
        'category_angles': category_angles,
        'n_nodes': n
    }

    return _PRECOMPUTED_CACHE[cache_key]


def build_wind_aware_adjacency_gpu(
    wind_speeds,
    wind_angles,
    alpha=0.6,
    distance_sigma=1800.0,
    calm_speed_threshold=0.1
):
    """
    GPU-optimized wind-aware adjacency with bidirectional alignment.

    Args:
        wind_speeds: (batch, num_nodes) tensor of aggregated wind speeds
        wind_angles: (batch, num_nodes) tensor of continuous wind angles in degrees,
                     or -1 for calm/variable
        alpha: Wind influence weight (0=distance-only, 1=wind-only)
        distance_sigma: Distance decay parameter
        calm_speed_threshold: Speed below which wind direction is ignored

    Returns:
        adj_batch: (batch, num_nodes, num_nodes) normalized adjacency tensor
    """
    device = wind_speeds.device
    batch_size, num_nodes = wind_speeds.shape

    # Get precomputed static matrices
    static = _precompute_static_matrices(device)
    dist = static['dist']      # (N, N)
    bearing = static['bearing']  # (N, N)

    # Calm mask: calm or explicitly marked as -1
    calm_mask = (wind_angles < 0) | (wind_speeds < calm_speed_threshold)

    # Distance-based component (broadcasted for batch)
    dist_decay = torch.exp(-dist.pow(2) / distance_sigma)  # (N, N)
    A_dist = dist_decay + torch.eye(num_nodes, device=device)  # Add self-loops

    # --- Source alignment ---
    # Transport direction is opposite of wind direction (wind FROM -> pollution TO)
    transport_dir = (wind_angles + 180.0) % 360.0  # (batch, N)
    transport_expanded = transport_dir.unsqueeze(2)  # (batch, N, 1)
    bearing_expanded = bearing.unsqueeze(0)            # (1, N, N)

    angle_diff = torch.abs(transport_expanded - bearing_expanded)  # (batch, N, N)
    angle_diff = torch.where(angle_diff > 180, 360 - angle_diff, angle_diff)
    source_alignment = (torch.cos(angle_diff * np.pi / 180.0) + 1) / 2  # (batch, N, N)

    # Wind speed factor (source)
    wind_factor = torch.tanh(wind_speeds / 5.0).unsqueeze(2)  # (batch, N, 1)

    # --- Receiver alignment (bidirectional) ---
    # bearing.T[i,j] = bearing[j,i] = bearing from j to i
    bearing_ji = bearing.T  # (N, N)
    # wind_angles for receiver j: expand over source i dimension
    wind_angles_recv = wind_angles.unsqueeze(1)  # (batch, 1, N)
    bearing_ji_exp = bearing_ji.unsqueeze(0)      # (1, N, N)

    recv_diff = torch.abs(wind_angles_recv - bearing_ji_exp)  # (batch, N, N)
    recv_diff = torch.where(recv_diff > 180, 360 - recv_diff, recv_diff)
    receiving = (torch.cos(recv_diff * np.pi / 180.0) + 1) / 2  # (batch, N, N)

    # Neutral receiving for calm receivers
    calm_recv = calm_mask.unsqueeze(1)  # (batch, 1, N) — j dim
    receiving = torch.where(calm_recv, torch.full_like(receiving, 0.5), receiving)

    # Combined bidirectional alignment
    alignment = source_alignment * (0.5 + 0.5 * receiving)  # (batch, N, N)

    # Wind component: alignment * speed_factor * distance_decay
    A_wind = alignment * wind_factor * dist_decay  # (batch, N, N)

    # Handle calm sources: set to neutral (0.5 * distance decay)
    calm_src = calm_mask.unsqueeze(2)  # (batch, N, 1)
    neutral_value = 0.5 * dist_decay  # (N, N)
    A_wind = torch.where(calm_src, neutral_value.unsqueeze(0), A_wind)

    # Add self-loops to wind component
    eye_batch = torch.eye(num_nodes, device=device).unsqueeze(0)  # (1, N, N)
    A_wind = A_wind + eye_batch

    # Hybrid adjacency
    A = (1 - alpha) * A_dist.unsqueeze(0) + alpha * A_wind  # (batch, N, N)

    # Row normalization
    row_sum = A.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    A_norm = A / row_sum

    return A_norm


def aggregate_wind_gpu(wind_speeds, wind_directions, mode="recent_weighted",
                       recency_beta=3.0, calm_speed_threshold=0.1):
    """
    Aggregate wind over temporal dimension on GPU using circular averaging.

    Uses sin/cos decomposition for proper circular mean (handles North/South
    wraparound correctly) and weights by both temporal recency and wind speed.

    Args:
        wind_speeds: (batch, timesteps, nodes) tensor
        wind_directions: (batch, timesteps, nodes, categories) tensor
        mode: Aggregation mode ('recent_weighted', 'last', 'mean')
        recency_beta: Weight emphasis for recent timesteps
        calm_speed_threshold: Speed below which direction is unreliable

    Returns:
        agg_speeds: (batch, nodes) tensor
        agg_angles: (batch, nodes) tensor — continuous angles in degrees [0, 360),
                    or -1 for calm/variable winds
    """
    batch, timesteps, nodes = wind_speeds.shape
    device = wind_speeds.device

    # Get precomputed category angles
    static = _precompute_static_matrices(device)
    category_angles_deg = static['category_angles']  # (C,)
    category_angles_rad = category_angles_deg * (np.pi / 180.0)  # (C,)

    # Create temporal weights
    if mode == "last":
        weights = torch.zeros(timesteps, device=device)
        weights[-1] = 1.0
    elif mode == "mean":
        weights = torch.ones(timesteps, device=device) / timesteps
    else:  # recent_weighted
        t = torch.linspace(0, 1, timesteps, device=device)
        weights = torch.exp(recency_beta * t)
        weights = weights / weights.sum()

    # Aggregate speeds: (batch, timesteps, nodes) @ (timesteps,) -> (batch, nodes)
    agg_speeds = torch.einsum('btn,t->bn', wind_speeds, weights)

    # --- Circular direction aggregation with speed weighting ---
    # Convert one-hot directions to sin/cos components
    sin_components = torch.sin(category_angles_rad)  # (C,)
    cos_components = torch.cos(category_angles_rad)  # (C,)

    # Weighted sin/cos per timestep per node
    # wind_directions: (B, T, N, C) * (C,) -> sum over C -> (B, T, N)
    sin_per_step = (wind_directions * sin_components).sum(dim=-1)  # (B, T, N)
    cos_per_step = (wind_directions * cos_components).sum(dim=-1)  # (B, T, N)

    # Combine temporal recency and speed weighting
    # Stronger, more recent winds have more influence on direction
    speed_weight = 0.5 + torch.clamp(wind_speeds, min=0.0)  # (B, T, N)
    combined_weights = weights.view(1, -1, 1) * speed_weight  # (B, T, N)
    combined_weights = combined_weights / combined_weights.sum(dim=1, keepdim=True).clamp(min=1e-8)

    # Aggregate sin/cos with combined weights
    sin_agg = (combined_weights * sin_per_step).sum(dim=1)  # (B, N)
    cos_agg = (combined_weights * cos_per_step).sum(dim=1)  # (B, N)

    # Convert back to angle
    agg_angles = torch.atan2(sin_agg, cos_agg) * (180.0 / np.pi)  # (B, N)
    agg_angles = (agg_angles + 360.0) % 360.0

    # Mark calm/variable winds as -1
    calm_mask = agg_speeds < calm_speed_threshold
    resultant = torch.sqrt(sin_agg.pow(2) + cos_agg.pow(2))
    variable_mask = resultant < 1e-6
    agg_angles = torch.where(calm_mask | variable_mask,
                             torch.full_like(agg_angles, -1.0), agg_angles)

    return agg_speeds, agg_angles


def build_dynamic_adjacency_gpu(X_batch, config):
    """
    Build dynamic wind-aware adjacency on GPU (no CPU transfer).

    Args:
        X_batch: (batch, timesteps, num_nodes, features) tensor ON GPU
        config: Configuration dict

    Returns:
        adj_batch: (batch, num_nodes, num_nodes) tensor ON GPU
    """
    wind_speed_idx = config['wind_speed_idx']
    wind_dir_start = config['wind_dir_start_idx']
    wind_dir_end = config['wind_dir_end_idx']

    # Extract wind features
    wind_speeds = X_batch[:, :, :, wind_speed_idx]  # (batch, timesteps, nodes)
    wind_directions = X_batch[:, :, :, wind_dir_start:wind_dir_end]  # (batch, timesteps, nodes, cats)

    # Aggregate over time (circular averaging + speed weighting)
    agg_speeds, agg_angles = aggregate_wind_gpu(
        wind_speeds,
        wind_directions,
        mode=config.get('wind_aggregation_mode', 'recent_weighted'),
        recency_beta=config.get('wind_recency_beta', 3.0),
        calm_speed_threshold=config.get('wind_calm_speed_threshold', 0.1)
    )

    # Build adjacency (with bidirectional alignment)
    adj_batch = build_wind_aware_adjacency_gpu(
        agg_speeds,
        agg_angles,
        alpha=config['wind_alpha'],
        distance_sigma=config['distance_sigma'],
        calm_speed_threshold=config.get('wind_calm_speed_threshold', 0.1)
    )

    return adj_batch


if __name__ == "__main__":
    build_adjacency()