import numpy as np
from scipy.spatial.transform import Rotation


def db2pow(x_db):
    return np.power(10, x_db / 10)


def to_rotm(ang_eul: np.ndarray) -> np.ndarray:
    return Rotation.from_euler('ZYX', ang_eul.flatten(), degrees=True).as_matrix()


def rank(x: np.ndarray) -> int:
    return len(x.shape)
