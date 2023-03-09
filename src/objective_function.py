import numpy as np
import matplotlib.pyplot as plt
from src.channel_mmWave_parameters import ChannelmmWaveParameters

class ObjectiveFunction:
    def __init__(self) -> None:
        ...

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