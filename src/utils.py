import numpy as np
from scipy.spatial.transform import Rotation


def db2pow(x_db):
    return np.power(10, x_db / 10)


def to_rotm(ang_eul: np.ndarray) -> np.ndarray:
    return Rotation.from_euler('ZYX', ang_eul, degrees=True).as_matrix()


def rank(x: np.ndarray) -> int:
    return len(x.shape)


def tp(x: np.ndarray) -> np.ndarray:
    if rank(x) == 1:
        return x.reshape(-1, 1)

    else:
        return x.T
