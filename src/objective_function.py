""" objective_function.py

Author: Artur Gasparyan
gasparyanartur99@gmail.com

Author:  Ruiqi Qiu
qiuruiqi1991@gmail.com 

Contains functionality for generating objective scores of positioning in environment.
"""


import math
import numpy as np
import matplotlib.pyplot as plt
from src.channel_mmWave_parameters import ChannelmmWaveParameters
from typing import Tuple


def get_slots_from_side(side: np.ndarray, offset: np.ndarray, n: int) -> np.ndarray:
    """ Split side to n evenly spaced slots. 

    Args:
        side <N, 2>: The discrete positions of the side.
        offset: <1, 2>: The offset of the slots to the side.
        n: The number of slots to divide into.

    Returns:
        slots <n>: The positions of the slots. 
    """
    if len(side.shape) != 2:
        raise ValueError(
            f"Side must be 2D. Received side with dimensions {side.shape}.")

    if n < 1:
        raise ValueError(
            f"Cannot divide side of length {len(side)} into {n} slots.")

    N = len(side)
    gap = N/(n+1)
    idx = (np.arange(1, n+1) * gap).astype(int)
    slots = side[idx] + offset
    return slots


def get_slots(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    n_slots: Tuple[int, int, int, int],
    slot_dists: Tuple[float, float, float, float]
) -> Tuple[np.ndarray]:
    """ Generate slots around a grid at a given distance.

    Args:
        grid_x <Nx>: The x-positions of the grid.
        grid_y <Ny>: The y-positions of the grid.
        n_slots (left, right, bot, top): The number of slots at a given side.
        slot_dists (left, right, bot, top): The distance between the slots and the sides (always positive).

    Returns:
        slots (left, right, bot, top) <N, 2>: The two-dimensional position of each slot for each side.
    """
    n_x = len(grid_x)
    n_y = len(grid_y)

    ys = (
        grid_y,
        grid_y,
        np.repeat(grid_y[0], n_x),
        np.repeat(grid_y[-1], n_x)
    )
    xs = (
        np.repeat(grid_x[0], n_y),
        np.repeat(grid_x[-1], n_y),
        grid_x,
        grid_x
    )

    sd_l, sd_r, sd_b, sd_t = slot_dists
    offsets = np.array([(0, -sd_l), (0, sd_r), (-sd_b, 0), (sd_t, 0)])

    slots = []
    for y, x, offset, n in zip(ys, xs, offsets, n_slots):
        p = np.vstack((y, x)).T
        slot = get_slots_from_side(p, offset, n)
        slots.append(slot)

    return tuple(slots)


class ObjectiveFunction:
    def __init__(self,
                 grid_range: Tuple[float, float, float, float] = (-5, 5, 0, 5),
                 grid_step: Tuple[float, float] = (0.5, 0.5),
                 n_slots: Tuple[int, int, int, int] = (8, 4, 8, 4),
                 slot_dists: Tuple[float, float, float,
                                   float] = (0.1, 0.1, 0.1, 0.1),
                 n_BS: int = 1,
                 n_UE: int = 3,
                 K: float = 64,

                 walls: Tuple[np.ndarray] = (
                     np.array([[-3, 2], [-1, 2]]),
                     np.array([[3, 2], [1.5, 4]])
                 ),

                 ) -> None:
        """ Defines an objective function for a given configuration of the environment.

        Args:
            grid_range (left, right, bot, top): Which coordinates the grid spans.
            grid_step (x, y): How small steps the grid is dividied into.
            n_slots (left, right, bot, top): How many possible locations the equipment can be placed at.
            slot_dists (left, right, bot, top): The distance each slot has to the corresponding side.
            n_BS: Number of base stations to consider during simulation.
            n_UE: Number of user equipments to consider during simulation.
            K: Parameter for simulation. Can be lowered for faster, but less precise simulation.
            walls <2,2> ((y1, x1), (y2, x2)): The edge positions of each wall in the environment.
        """
        x_min, x_max, y_min, y_max = grid_range
        x_step, y_step = grid_step

        grid_x = np.arange(x_min, x_max+x_step, x_step)
        grid_y = np.arange(y_min, y_max+y_step, y_step)

        self.n_BS = n_BS
        self.n_UE = n_UE

        self.K = K

        self.walls = walls

        self.slots = get_slots(grid_x, grid_y, n_slots, slot_dists)

    def get_score(self, debug=False):
        # TODO: Fix initial points
        # TODO: Include orientation in slots (always 90 deg)

        c = ChannelmmWaveParameters(
            PB=np.array([0, -0.01, 0]).reshape(-1, 1),
            OB=np.array([45, 0, 0]).reshape(-1, 1),
            NB_dim=np.array([1, 1]).reshape(-1, 1),
            PR=np.array(
                [[-5.01, 2, 0.5], [-1, 5.01, 0.5], [5.01, 2.5, 0.5]]).T,
            OR=np.array([[0, 0, 0], [-90, 0, 0], [180, 0, 0]]).T,
            NR_dim=np.array([[10, 10], [10, 10], [10, 10]]).T,
            PU=np.array([-2, 3, -1.5]).reshape(-1, 1),
            OU=np.array([0, 0, 0]).reshape(-1, 1),
            NU_dim=np.array([1, 1]).reshape(-1, 1),
            K=self.K,
            Wall=list(self.walls),
            G=10,
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
        PEB_cell = c.get_PEB_cell(
            xgrid, ygrid, parallel=True, verbosity=1, print_runtimes=True)
        error_grid, PEB_CDF = c.get_PEB_CDF(PEB_cell)
        score = c.get_CDF_threshold(error_grid, PEB_CDF, threshold=0.9)

        if debug:
            c.plot_CDF_PEB(error_grid, PEB_CDF)
            c.plot_PEB(PEB_cell)
            print(f'Score: {score: .4f}')
            print("Finished objective function")
            plt.show()

        return score
