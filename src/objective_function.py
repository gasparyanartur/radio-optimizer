import math
import numpy as np
import matplotlib.pyplot as plt
from src.channel_mmWave_parameters import ChannelmmWaveParameters
from typing import Tuple

def get_slots_from_side(side: np.ndarray, n: int) -> np.ndarray:
    """ Split side to n evenly spaced slots. 

    Args:
        side <N>: The discrete positions of the side.
        n: The number of slots to divide into.

    Returns:
        slots <n>: The positions of the slots. 
    """
    if len(side.shape) > 1:
        raise ValueError(f"Side must be flat. Received side with dimensions {side.shape}.")

    if n < 1:
        raise ValueError(f"Cannot divide side of length {len(side)} into {n} slots.")

    N = len(side)
    gap = N/(n+1)
    idx = (np.arange(1, n+1) * gap).astype(int)
    return side[idx]


class ObjectiveFunction:
    def __init__(self,
                 resolution: Tuple[float, float] = (20, 10),
                 x_range: Tuple[float] = (-5, 5), 
                 y_range: Tuple[float] = (0, 5),
                 n_pos_hori: int = 8,
                 n_pos_vert: int = 4
                 ) -> None:
        x_res, y_res = resolution
        x_min, x_max = x_range
        y_min, y_max = y_range



    def get_score(self, debug=False):
        c = ChannelmmWaveParameters(
            PB = np.array([0, -0.01, 0]).reshape(-1, 1),
            OB = np.array([45, 0, 0]).reshape(-1, 1),
            NB_dim = np.array([1, 1]).reshape(-1, 1),
            PR = np.array([[-5.01, 2, 0.5], [-1, 5.01, 0.5], [5.01, 2.5, 0.5]]).T,
            OR = np.array([[0, 0, 0], [-90, 0, 0], [180, 0, 0]]).T,
            NR_dim = np.array([[10, 10], [10, 10], [10, 10]]).T,
            PU = np.array([-2, 3, -1.5]).reshape(-1, 1),
            OU = np.array([0, 0, 0]).reshape(-1, 1),
            NU_dim = np.array([1, 1]).reshape(-1, 1),
            K = 64,
            G = 10,
            seed=1
        )

        if debug:
            print("Starting objective function...")
                
        c.get_beam_matrix()
        c.get_tx_symbol()
        c.get_path_parameters_PWM()
        c.get_FIM_PWM()

        fim = c.FIM
        c.get_crlb_from_fim_PWM(fim)

        grid_size = 0.5     # ~7s
        xgrid = np.arange(-5, 5+grid_size, grid_size)
        ygrid = np.arange(0, 5+grid_size, grid_size)
        PEB_cell = c.get_PEB_cell(xgrid, ygrid, parallel=True, verbosity=1, print_runtimes=True)
        error_grid, PEB_CDF = c.get_PEB_CDF(PEB_cell)
        score = c.get_CDF_threshold(error_grid, PEB_CDF, threshold=0.9)

        if debug:
            c.plot_CDF_PEB(error_grid, PEB_CDF)
            c.plot_PEB(PEB_cell)
            print(f'Score: {score: .4f}')
            print("Finished objective function")
            plt.show()


        return score