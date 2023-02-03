"""
"""
import numpy as np
from .utils import tp

def get_array_layout(N_dim: np.ndarray) -> np.ndarray:
    r, c = N_dim
    X, Y = np.meshgrid(np.arange(c) - (c-1)/2, np.arange(r) - (r-1)/2)
    XF = np.hstack((X[0], X[1]))
    YF = np.hstack((Y[0], Y[1]))
    layout = np.vstack((np.zeros(r*c), XF, YF))
    return layout