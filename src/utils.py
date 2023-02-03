import numpy as np


def db2pow(x_db):
    return np.power(10, x_db / 10)


def npa(x) -> np.ndarray:
    return np.array(x)