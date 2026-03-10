import numpy as np
from math import radians, sin, cos, sqrt, atan2

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

def haversine(coord1, coord2):
    R = 6371
    lat1, lon1 = radians(coord1[0]), radians(coord1[1])
    lat2, lon2 = radians(coord2[0]), radians(coord2[1])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))

    return R * c


def build_adjacency():
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


if __name__ == "__main__":
    build_adjacency()