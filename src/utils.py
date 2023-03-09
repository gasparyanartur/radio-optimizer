import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from typing import Tuple


def db2pow(x_db):
    return np.power(10, x_db / 10)


def to_rotm(ang_eul: np.ndarray) -> np.ndarray:
    return Rotation.from_euler('ZYX', ang_eul.flatten(), degrees=True).as_matrix()


def rank(x: np.ndarray) -> int:
    return len(x.shape)


def get_angle_from_dir(tv: np.ndarray, in_degrees=True) -> Tuple[float, float]:
    phi = np.arctan2(tv[1, :], tv[0, :])
    theta = np.arcsin(tv[2, :])

    if in_degrees:
        phi = np.rad2deg(phi)
        theta = np.rad2deg(theta)

    return phi, theta


def get_linexline(L1x, L1y, L2x, L2y):
    """ Find the intersection of two line segments
    
    Converted to Python from https://github.com/preethamam

    args:
        L1x: Line 1 x1 and x2 coordinates [x1, x2]
        L1y: Line 1 y1 and y2 coordinates [y1, y2]
        L2x: Line 2 x1 and x2 coordinates [x3, x4]
        L2y: Line 2 y1 and y2 coordinates [y3, y4]

    outputs:
        xi: Intersection point x, return NaN if no intersection
        yi: Intersectoin point y, return NaN if no intersection
    """

    x1, x2 = L1x
    y1, y2 = L1y
    x3, x4 = L2x
    y3, y4 = L2y

    x21, y21 = x1 - x2, y1 - y2
    x12, y12 = x2 - x1, y2 - y1
    x31, y31 = x1 - x3, y1 - y3
    x43, y43 = x3 - x4, y3 - y4
    x34, y34 = x4 - x3, y4 - y3

    # Line segment intersect parameters
    d = (x21*y43 - y21*x43) 
    if d == 0:
        return np.NaN, np.NaN

    u = (x31*y21 - y31*x21) / d
    t = (x31*y43 - y31*x43) / d

    # Check if intersection exists, if so then store the value
    if (0 <= u <= 1) and (0 <= t <= 1):
        xi = (x3 + u*x34 + x1 + t * x12) / 2
        yi = (y3 + u*y34 + y1 + t * y12) / 2

    else:
        xi = yi = np.NaN

    return xi, yi


def get_cdf(y, xgrid):
    """ Get the CDF of y """

    return 1 - np.sum(y.reshape(1, -1) >= xgrid.reshape(-1, 1), axis=1) / y.size


def rand(*shape):
    """ Generates a random array that matches the Matlab random array, assuming that the seed is the same. """
    
    if not shape:
        return np.random.rand()

    return np.random.rand(*shape).flatten().reshape(*shape, order='F')


def is_invertible(a):
    return a.shape[0] == a.shape[1] and np.linalg.matrix_rank(a) == a.shape[0]


def setup_fig():
    plt.figure(figsize=(9, 6))
    plt.xlim(-5.25, 5.25)
    plt.ylim(-0.25, 5.25)
    plt.xticks([-5, 0, 5])
    plt.xlabel('x-axis [m]')
    plt.ylabel('y-axis [m]')