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


def build_adjacency():
    """Build static distance-based adjacency matrix."""
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
                A[i, j] = np.exp(-d**2 / 100)

    D = np.diag(np.sum(A, axis=1))
    D_inv_sqrt = np.linalg.inv(np.sqrt(D))
    A_hat = D_inv_sqrt @ A @ D_inv_sqrt

    np.save("../data/processed/adjacency.npy", A_hat)
    print("Adjacency matrix saved.")


def build_wind_aware_adjacency(wind_speeds, wind_directions, wind_categories,
                                alpha=0.6, distance_sigma=100):
    """
    Build dynamic wind-aware directed adjacency matrix.

    Args:
        wind_speeds: (num_nodes,) array of wind speeds at each station
        wind_directions: (num_nodes, num_categories) array of one-hot wind directions
        wind_categories: List of wind direction category names in order
        alpha: Weighting factor (0 to 1).
               alpha=1.0 means pure wind-based,
               alpha=0.0 means pure distance-based,
               alpha=0.6 is a balanced hybrid (recommended)
        distance_sigma: Distance decay parameter for Gaussian kernel

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
    np.fill_diagonal(A_dist, 0)

    # Wind-based component (directed)
    A_wind = np.zeros((n, n))

    for i in range(n):
        wind_angle = wind_direction_to_angle(wind_directions[i], wind_categories)
        wind_speed = wind_speeds[i]

        for j in range(n):
            if i != j:
                bearing = bearing_matrix[i, j]
                alignment = compute_wind_alignment(wind_angle, bearing, wind_speed)

                # Combine alignment with distance decay
                A_wind[i, j] = alignment * np.exp(-dist_matrix[i, j]**2 / distance_sigma)

    # Hybrid adjacency: combine distance-based and wind-based
    A = (1 - alpha) * A_dist + alpha * A_wind

    # Normalize using degree matrix (symmetric normalization)
    row_sum = np.sum(A, axis=1)
    row_sum[row_sum == 0] = 1  # Avoid division by zero
    D_inv_sqrt = np.diag(1.0 / np.sqrt(row_sum))

    A_hat = D_inv_sqrt @ A @ D_inv_sqrt

    return A_hat


def build_wind_aware_adjacency_batch(wind_speeds, wind_directions, wind_categories,
                                      alpha=0.6, distance_sigma=100):
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
        # Average over timesteps for each station
        wind_speeds = wind_speeds.mean(axis=1)  # (batch, nodes)
        wind_directions = wind_directions.mean(axis=1)  # (batch, nodes, categories)
    elif wind_speeds.ndim == 2:  # (batch, nodes)
        batch_size, num_nodes = wind_speeds.shape
    else:
        raise ValueError(f"Unexpected wind_speeds shape: {wind_speeds.shape}")

    adjacency_batch = np.zeros((batch_size, num_nodes, num_nodes))

    for b in range(batch_size):
        adjacency_batch[b] = build_wind_aware_adjacency(
            wind_speeds[b],
            wind_directions[b],
            wind_categories,
            alpha=alpha,
            distance_sigma=distance_sigma
        )

    return adjacency_batch


if __name__ == "__main__":
    build_adjacency()