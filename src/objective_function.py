import numpy as np
import matplotlib.pyplot as plt
from src.channel_mmWave_parameters import ChannelmmWaveParameters
from typing import Tuple

def get_placements(n_placements, xgrid, ygrid):
    print(xgrid, ygrid)
    n_y, n_x = len(xgrid), len(ygrid)
    rat_y = n_y / n_x
    rat_x = 1

    circ = (2*rat_y + 2*rat_x)
    rat_y /= circ
    rat_x /= circ
    
    norm_circ = 2*rat_y + 2*rat_x
    print(norm_circ)


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