""" channel_mmWave_parameters.py  

Author:  Ruiqi Qiu
qiuruiqi1991@gmail.com 

Author: Artur Gasparyan
gasparyanartur99@gmail.com

Initialize default channel parameters, including several parts.
1). General model parameters: set channel and optimization features
2). Signal Parameters: central carrier frequency, bandwidth and so on.
3). Geometry Parameters: set position and orientation for BS/RIS/UE/VA/SP
4). Infrastructure Parameters: array & AOSA dimensions, signal and antenna parameters
5). Environment Parameters: noise figure, noise level
"""


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import multiprocessing as mp
from enum import Enum, auto, unique
from .utils import db2pow, to_rotm, get_angle_from_dir, get_linexline, rand, is_invertible, get_cdf, setup_fig
import copy
import time
from typing import Tuple, List


@unique
class SynType(Enum):
    Syn = auto()
    Asyn = auto()


@unique
class LinkType(Enum):
    Uplink = auto()
    Downlink = auto()


@unique
class BeamType(Enum):
    Random = auto()
    Directional = auto()
    Derivative = auto()


@unique
class RisProfileType(Enum):
    Random = auto()
    Directional = auto()
    Derivative = auto()
    Customized = auto()


@unique
class ArrayType(Enum):
    # Digital: # of RFCs = # of Elements, M = N
    Analog = auto()
    Digital = auto()
    Hybrid = auto()


@unique
class RadiationType(Enum):
    Omni = auto()
    Cos = auto()
    Cos2 = auto()


@unique
class UpdateArgsType(Enum):
    All = auto()
    Signal = auto()
    Geometry = auto()
    Channel = auto()


@unique
class PathType(Enum):
    L = auto()
    R = auto()
    V = auto()
    S = auto()
    B = auto()


def get_array_layout(N_dim: np.ndarray) -> np.ndarray:
    r, c = N_dim

    X, Y = np.meshgrid(np.arange(c) - (c-1)/2, np.arange(r) - (r-1)/2)

    XF = X.ravel(order="F")
    YF = Y.ravel(order="F")
    Z = np.zeros(r*c)

    layout = np.vstack((Z, XF, YF))
    return layout


def get_D_Phi_t(phi_deg, theta_deg):
    """ Get the derivative of Phi (phi/theta) from direction vector t """
    phi_rad = np.deg2rad(phi_deg)
    theta_rad = np.deg2rad(theta_deg)

    cos_phi = np.cos(phi_rad)
    sin_phi = np.sin(phi_rad)
    cos_theta = np.cos(theta_rad)
    sin_theta = np.sin(theta_rad)

    D_phi_t = np.array(
        (-sin_phi*cos_theta, cos_phi*cos_theta, 0)).reshape(-1, 1)
    D_theta_t = np.array((-cos_phi*sin_theta, -sin_phi *
                         sin_theta, cos_theta)).reshape(-1, 1)

    return D_phi_t, D_theta_t


def get_EFIM_from_FIM(FIM, N_states):
    if isinstance(N_states, int):
        F1 = FIM[:N_states, :N_states]
        F2 = FIM[:N_states, N_states:]
        F4 = FIM[N_states:, N_states:]
        EFIM = F1 - F2 @ np.linalg.inv(F4) @ F2.T

    else:
        F1 = FIM[:N_states[0], :N_states[0]]
        F2 = FIM[:N_states[0], N_states[0]:]
        F4 = FIM[N_states[0]:, N_states[0]:]
        EFIM = F4 - F2.T @ np.linalg.inv(F1) @ F2

        if N_states[-1] != FIM.shape[0]:
            N_states = N_states[-1] - N_states[0] + 1
            F1 = EFIM[:N_states, :N_states]
            F2 = EFIM[:N_states, N_states:]
            F4 = EFIM[N_states:, N_states:]
            EFIM = F1 - F2 @ np.linalg.inv(F4) @ F2.T

    return EFIM


def get_PEB_and_CEB(CRLB):
    complex_CRLB = CRLB.astype('complex_')
    PEB = np.sqrt(np.trace(complex_CRLB[:3, :3]))
    CEB = np.sqrt(complex_CRLB[3, 3])

    return PEB, CEB


def get_PEB_from_PU(c, PU):
    c.PU = PU
    c.update_parameters()
    c.get_path_parameters_PWM()
    c.get_FIM_PWM()
    c.get_crlb_from_fim_PWM(c.FIM)

    blockage = c.get_blockage(c.PU[:2].T)
    c.get_crlb_blockage(blockage)

    return c.PEB


def get_D_mu_channel_parameters(
        L, path_info, class_index,
        muB_cell, rho_cell, doppler_cell, H_cell,
        MB, K, G, link_type,
        XU_mat, WB_mat, lambdac, fdk, c,
        AstRB_cell, AstRU_cell,
        phiRU_loc, thetaRU_loc, omega,
        beamsplit_coe, R0
    ):
    """ Get the FIM of the measurement vector (PWM)

        phiBU, thetaBU, phiUB, thetaUB, tau, v, rho, xi
    """
    D_muB_cell = [None for _ in range(L)]

    pi_2j = 2j * np.pi
    npi_2j_factor = -pi_2j / c * fdk
    pi2j_lambdac = pi_2j / lambdac

    for lp in range(L):
        curr_type = path_info[lp]
        lc = class_index[lp]
        muB = muB_cell[lp]
        rho = rho_cell[lp]
        doppler_mat = doppler_cell[lp]
        H = H_cell[lp]

        # 2 Gain, 2 DOA, 2 DOD, 1 tau, 1 velocity (doppler)
        D_muB = np.zeros((MB, 3, K, G), dtype='complex_')

        # LOS channel
        if curr_type == PathType.L:
            # Calculate FIM Uplink
            if link_type == LinkType.Uplink:
                for g in range(G):
                    XUg = XU_mat[:, :, g]
                    WB = WB_mat[:, :, g]          # This too
                    doppler_k = doppler_mat[:, g].T

                    for k in range(K):
                        # BU channel
                        # received symbols at BS
                        muBg = muB[:, k, g]
                        D_muB_rhoBU = muBg / rho[k]
                        D_muB_xiBU = pi2j_lambdac * muBg
                        D_muB_dL = WB.T @ H[:, :, k] @ XUg[:, k] * \
                            npi_2j_factor[k] * doppler_k[k]

                        D_muB[:, 0, k, g]= D_muB_dL
                        D_muB[:, 1, k, g]= D_muB_rhoBU
                        D_muB[:, 2, k, g]= D_muB_xiBU

        elif curr_type == PathType.R:
            rho = rho_cell[lp]
            D_muB = np.zeros((MB, 5, K, G),
                             dtype='complex_')
            AstRB = AstRB_cell[lp]
            AstRU = AstRU_cell[lp]
            AstR = AstRB * AstRU
            D_tRU_phiRU_loc, D_tRU_thetaRU_loc = get_D_Phi_t(phiRU_loc[lc], thetaRU_loc[lc])

            scale_AstRU = AstRU * \
                np.array(
                    np.matrix(pi_2j / (lambdac * beamsplit_coe)).H)

            D_AstRU_phiRU = scale_AstRU * (R0[lc].T @ D_tRU_phiRU_loc)
            D_AstRU_thetaRU = scale_AstRU * \
                (R0[lc].T @ D_tRU_thetaRU_loc)

            # Calculate FIM Uplink
            if link_type == LinkType.Uplink:
                for g in range(G):
                    XUg = XU_mat[:, :, g]
                    WB = WB_mat[:, :, g]
                    doppler_k = doppler_mat[:, g].T
                    Omega_g = omega[lc][:, g]
                    ris_g = Omega_g.T @ AstR        # RIS gain of the g-th transmission

                    for k in range(K):
                        # RIS channel
                        # Received symbols at BN
                        muBg = muB[:, k, g]

                        D_muB_rhoR = muBg / rho[k]
                        D_muB_xiR = pi2j_lambdac * muBg

                        factor = WB.T @  H[:, :, k] @ XUg[:, k]
                        factor_doppler = factor * doppler_k[k]
                        AstRB_k = AstRB[:, k]
                        D_muB_dR = factor * \
                            npi_2j_factor[k] * doppler_k[k]*ris_g[k]
                        D_muB_phiRU = factor_doppler * (Omega_g.T @
                                            (AstRB_k * D_AstRU_phiRU[:, k]))
                        D_muB_thetaRU = factor_doppler * (Omega_g.T @
                                            (AstRB_k * D_AstRU_thetaRU[:, k]))

                        D_muB[:, 0, k, g] = D_muB_phiRU
                        D_muB[:, 1, k, g] = D_muB_thetaRU
                        D_muB[:, 2, k, g] = D_muB_dR
                        D_muB[:, 3, k, g] = D_muB_rhoR
                        D_muB[:, 4, k, g] = D_muB_xiR

        D_muB_cell[lp] = D_muB

    return D_muB_cell


class ChannelmmWaveParameters:
    def __init__(
        self,
        syn_type: SynType = SynType.Asyn,
        link_type: LinkType = LinkType.Uplink,

        # Signal
        beam_type: BeamType = BeamType.Random,
        ris_profile_type: RisProfileType = RisProfileType.Random,
        beamforming_angle_std: float = 0,        # beamforming angle variation
        array_type: ArrayType = ArrayType.Analog,
        radio_directional: float = 1,        # ratio of directional beams

        UE_radiation: RadiationType = RadiationType.Omni,
        RIS_radiation: RadiationType = RadiationType.Omni,
        BS_radiation: RadiationType = RadiationType.Omni,

        # Signal parameters
        c: float = 3E8,      # speed of light
        fc: float = 28E9,    # carrier frequency
        BW: float = 400E6,   # 100M bandwidth
        # in [mW]. 1000mW = 30dBm. 1mW = 0dBm.
        P: float = db2pow(20),
        # Synchronization offset (in [meter], 10 us -> 3000 m)
        beta: float = 30,
        K: int = 10,      # Number of carrier frequencies
        # Measurement interval [sec] (min value = K/BW)
        T: float = 1/1000,

        # Geometry parameters
        # Global position: 3x1 vector (2D by settings P(3) as zero)

        # BS: Base Station
        PB: np.ndarray = np.array([0, 0, 0]).reshape(-1, 1),
        OB: np.ndarray = np.array([90, 0, 0]).reshape(-1, 1),

        # UE: User Equipment
        PU: np.ndarray = np.array([5, 2, 0]).reshape(-1, 1),
        OU: np.ndarray = np.array([0, 0, 0]).reshape(-1, 1),
        VU: np.ndarray = np.array([0, 0, 0]).reshape(-1, 1),

        # RIS: Reconfigurable Intelligent Surfaces
        PR: np.ndarray = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]),
        OR: np.ndarray = np.array([0, 0, 0]).reshape(-1, 1),
        VR: np.ndarray = np.array([0, 0, 0]).reshape(-1, 1),

        # Infrastructure Parameters
        # Number of RFCs at BS and UE
        MB: int = 1,
        MU: int = 1,

        # Array Dimensions
        NB_dim: np.ndarray = np.array([4, 4]).reshape(-1, 1),
        NR_dim: np.ndarray = np.array([10, 10]).reshape(-1, 1),
        NU_dim: np.ndarray = np.array([4, 4]).reshape(-1, 1),

        # Environment Parameters
        operationBW: float = 400E6,      # Operation bandwidth for Thermal noise,
        K_boltzmann: float = 1.3806E-23,  # Boltzmann constant,
        temperature: float = 298.15,     # Temperature 25 celsius,

        noise_figure: float = 10,                # Noise figure 3dB
        G: int = 10,

        Wall: List[np.ndarray] = [
            np.array([[-3, 2], [-1, 2]]),
            np.array([[3, 2], [1.5, 4]])
        ],

        seed=1          # Random seed
    ):
        self.seed = seed
        np.random.seed(self.seed)

        # Model parameters
        self.syn_type: SynType = syn_type
        self.link_type: LinkType = link_type

        # Signal
        self.beam_type: BeamType = beam_type
        self.ris_profile_type: RisProfileType = ris_profile_type
        # beamforming angle variation
        self.beamforming_angle_std: float = beamforming_angle_std
        self.array_type: ArrayType = array_type
        # ratio of directional beams
        self.radio_directional: float = radio_directional

        self.UE_radiation: RadiationType = UE_radiation
        self.RIS_radiation: RadiationType = RIS_radiation
        self.BS_radiation: RadiationType = BS_radiation

        # Signal parameters
        self.c: float = c   # speed of light
        self.fc: float = fc    # carrier frequency
        self.lambdac: float = c/fc
        self.BW: float = BW   # 100M bandwidth
        self.P: float = P   # in [mW]. 1000mW = 30dBm. 1mW = 0dBm.
        # Synchronization offset (in [meter], 10 us -> 3000 m)
        self.beta: float = beta
        self.K: int = K      # Number of carrier frequencies
        self.T: float = T   # Measurement interval [sec] (min value = K/BW)

        # Geometry parameters
        # Global position: 3x1 vector (2D by settings P(3) as zero)

        # BS: Base Station
        self.PB: np.ndarray = PB
        self.OB: np.ndarray = OB

        # UE: User Equipment
        self.PU: np.ndarray = PU
        self.OU: np.ndarray = OU
        self.VU: np.ndarray = VU

        # RIS: Reconfigurable Intelligent Surfaces
        self.PR: np.ndarray = PR
        self.OR: np.ndarray = OR
        self.VR: np.ndarray = VR

        # Infrastructure Parameters
        # Number of RFCs at BS and UE
        self.MB: int = MB
        self.MU: int = MU

        # Array Dimensions
        self.NB_dim: np.ndarray = NB_dim
        self.NR_dim: np.ndarray = NR_dim
        self.NR = np.prod(self.NR_dim, 0)
        self.NU_dim: np.ndarray = NU_dim

        # Number of elements at BS/RIS/UE
        # Number of antennas for conventional array
        self.NB: np.ndarray = np.prod(self.NB_dim)    # Number of BS elements
        self.NU: np.ndarray = np.prod(self.NU_dim)    # Number of UE elements
        self.NR: np.ndarray = np.prod(
            self.NR_dim, 0)    # Number of RIS elements

        # Environment Parameters
        # Operation bandwidth for Thermal noise
        self.operationBW: float = operationBW
        self.K_boltzmann: float = K_boltzmann  # Boltzmann constant
        self.temperature: float = temperature     # Temperature 25 celsius

        self.Pn: float = (
            K_boltzmann * temperature * operationBW * 1000
        )      # Thermal noise linear (in mW)
        self.Pn_dBm: float = 10 * np.log10(self.Pn)       # Thermal noise in dB

        # Johnson-Nyquist noise: sigma^2 = N_0
        self.sigma_in: float = np.sqrt(self.Pn)
        self.noise_figure: float = noise_figure                # Noise figure 3dB
        self.sigma: float = np.sqrt(
            np.power(10, noise_figure/10)
        ) * self.sigma_in      # Output noise level
        self.sigma0: float = np.sqrt(self.Pn)

        self.G: int = G

        # Half wavelength (default distance between antennas/antenna spacing)
        self.dant: float = self.lambdac/2
        self.fdk: np.ndarray = (
            -self.BW/2 + self.BW /
            (2*self.K) + (self.BW/self.K) * np.arange(self.K).reshape(-1, 1)
        )    # Subcarrier frequency
        self.fk: np.ndarray = self.fdk + self.fc

        # Wavelength for different subcarriers (wide BW systems)
        self.lambdak: np.ndarray = self.c / self.fk
        self.beamsplit_coe = np.ones(len(self.lambdak))
        self.LR: int = self.PR.shape[1]   # Number of RISs
        self.L: int = self.LR + 1

        # Rotation matrix from euler angles
        self.RB: np.ndarray = to_rotm(self.OB.T)
        self.tB: np.ndarray = self.RB @ np.array([1, 0, 0]).reshape(-1, 1)
        self.B0: np.ndarray = self.dant * \
            get_array_layout(self.NB_dim)    # Local AE position
        self.B: np.ndarray = self.PB + self.RB @ self.B0     # Global AE position

        # Rotation matrix from euler angles
        self.RU: np.ndarray = to_rotm(self.OU.T)
        self.tU: np.ndarray = self.RU @ np.array([1, 0, 0]).reshape(-1, 1)
        self.U0: np.ndarray = self.dant * \
            get_array_layout(self.NU_dim)     # Local AE position
        self.U: np.ndarray = self.PU + self.RU @ self.U0    # Global AE position

        # TODO: Optimize using 3D tensor
        self.RR: List[np.ndarray] = []      # List of rotation matrixes
        self.tR: np.ndarray = np.zeros((3, self.LR))
        self.R0: List[np.ndarray] = []      # List of local AE positions
        self.R: List[np.ndarray] = []       # List of global AE positions

        # LOS Channel

        self.dBU: float = np.linalg.norm(self.PB - self.PU)
        self.tauBU: float = self.dBU / self.c
        self.dL: float = self.dBU + self.beta
        self.tauL: float = (self.dBU + self.beta) / self.c

        # global DOD/DOA: global = Rotm*local; local = Rotm^-1*global
        self.tBU: np.ndarray = (self.PU - self.PB) / self.dBU
        self.tUB: np.ndarray = -self.tBU
        self.phiBU: float = 0
        self.thetaBU: float = 0

        # local DOD/DOA: global = Rotm*local; local = Rotm^-1*global
        # Unit direction vector (local) from Tx to Rx
        self.tBU_loc: np.ndarray = self.RB.T @ self.tBU
        self.phiBU_loc: float = 0
        self.thetaBU_loc: float = 0
        # Unit direction vector (local) from Tx to RX
        self.tUB_loc: np.ndarray = self.RU.T @ (-self.tBU)
        self.phiUB_loc: float = 0
        self.thetaUB_loc: float = 0
        self.rhoL: float = 0

        # RIS Channel

        self.dBR: np.ndarray = np.linalg.norm(self.PB - self.PR, axis=0)
        self.dRB: np.ndarray = self.dBR
        self.dRU: np.ndarray = np.linalg.norm(self.PU - self.PR, axis=0)
        self.dUR: np.ndarray = self.dRU
        self.dBRU: np.ndarray = self.dBR + self.dRU
        # Delay of the LOS path, in [m]
        self.dR: np.ndarray = self.dBRU + self.beta
        self.tauBRU: np.ndarray = self.dBRU / self.c
        # Signal delay, propagation time + offset
        self.tauR: np.ndarray = (self.dBRU + self.beta) / self.c

        # global DOD/DOA: global = Rotm*local; local = Rotm^-1*global
        self.tBR: np.ndarray = (self.PR-self.PB) / self.dBR
        self.tRB: np.ndarray = -self.tBR
        self.tUR: np.ndarray = (self.PR-self.PU) / self.dUR
        self.tRU: np.ndarray = -self.tUR

        self.phiBR: float = 0
        self.thetaBR: float = 0
        self.phiUR: float = 0
        self.thetaUR: float = 0

        # local DOD/DOA: global = Rotm*local; local = rotm^-1*global
        # Unit direction vector (local) from Tx to Rx
        self.tBR_loc: np.ndarray = self.RB.T @ self.tBR
        self.phiBR_loc: float = 0
        self.thetaBR_loc: float = 0

        # Unit direction vector (local) from Tx to Rx
        self.tUR_loc: np.ndarray = self.RU.T @ self.tUR
        self.phiUR_loc: float = 0
        self.thetaUR_loc: float = 0

        self.tRB_loc = np.zeros((3, self.LR))
        self.tRU_loc = np.zeros((3, self.LR))

        self.phiRB_loc: float = 0
        self.thetaRB_loc: float = 0
        self.phiRU_loc: float = 0
        self.thetaRU_loc: float = 0

        self.rhoR: np.ndarray = np.zeros((1, self.LR))
        self.xiR: np.ndarray = np.zeros((1, self.LR))

        self.N_measures: int = 3 + 5 * self.LR
        self.N_unknowns: int = 6 + 2 * self.LR

        self.path_type: List[PathType] = []
        self.path_info: List[PathType] = []

        self.class_index: np.ndarray = np.ones(self.L, dtype=int)

        self.WU_mat: np.ndarray = np.zeros((self.NU, self.MU, self.G))
        self.WB_mat: np.ndarray = np.zeros((self.NB, self.MB, self.G))
        self.omega: List[np.ndarray] = [None for _ in range(self.LR)]

        self.XU_mat: np.ndarray = np.zeros(
            (self.NU, self.K, self.G))     # Each cell has size N x Ks

        self.alpha_cell: List[np.ndarray] = [None for _ in range(self.L)]
        self.rho_cell: List[np.ndarray] = [None for _ in range(self.L)]
        self.Xi_cell: List[np.ndarray] = [None for _ in range(self.L)]
        self.v_cell: List[np.ndarray] = [None for _ in range(self.L)]
        self.H_cell: List[np.ndarray] = [None for _ in range(self.L)]

        # Steering vector from BS to UE, RIS, IP
        self.AstBX_cell: List[np.ndarray] = [None for _ in range(self.L)]
        self.AstUX_cell: List[np.ndarray] = [None for _ in range(self.L)]
        self.AstRB_cell: List[np.ndarray] = [None for _ in range(self.L)]
        self.AstRU_cell: List[np.ndarray] = [None for _ in range(self.L)]

        # Received symbols at BS
        self.muB_cell: List[np.ndarray] = [None for _ in range(self.L)]
        self.doppler_cell: List[np.ndarray] = [None for _ in range(self.L)]

        self.D_muB_cell: List[np.ndarray] = [None for _ in range(self.L)]
        self.D_muB_UR_cell: List[np.ndarray] = [None for _ in range(self.LR)]
        self.muB: np.ndarray = np.zeros((self.MB, self.K, self.G))

        self.JM_cell: List[np.ndarray] = [None for _ in range(self.L)]

        # FIM of the measurement vector
        self.FIM_M: np.ndarray = np.zeros(self.N_measures)
        self.FIM: np.ndarray = np.zeros(self.N_unknowns)

        self.JS_all: np.ndarray = np.zeros((self.N_unknowns, self.N_measures))
        self.JS: np.ndarray = np.zeros((self.N_unknowns, self.N_measures))

        self.Wall: List[np.ndarray] = Wall
        self.Anchor: np.ndarray = np.hstack((self.PB[:2], self.PR[:2, :]))

        self.update_parameters()

    def update_parameters(self, args: UpdateArgsType = UpdateArgsType.All):
        # Update Signal parameters
        if args == UpdateArgsType.All or args == UpdateArgsType.Signal:
            self.lambdac = self.c/self.fc
            self.dant = self.lambdac/2
            self.fdk = (
                -self.BW/2 +
                self.BW / (2*self.K) +
                (self.BW/self.K)*np.arange(self.K).reshape(-1, 1)
            )
            self.fk = (self.fdk + self.fc)
            self.lambdak = (self.c / self.fk)
            self.beamsplit_coe = np.ones(self.lambdak.shape)

        # Update Geometry parameters
        if args == UpdateArgsType.All or args == UpdateArgsType.Geometry:
            self.update_geometry()

            self.N_measures = 3 + 5 * self.LR
            self.N_unknowns = 6 + 2 * self.LR

            # TODO: Optimize
            self.path_type = np.array([PathType.R for _ in range(self.L)])
            self.path_info = np.array([PathType.R for _ in range(self.L)])
            self.path_type[0] = PathType.L
            self.path_info[0] = PathType.L

            self.class_index = np.zeros(self.L, dtype=int)
            for lp in range(self.L):
                if self.path_type[lp] == PathType.R:
                    self.class_index[lp] = lp-1

        # Update environment parameters
        self.operationBW = self.BW
        self.Pn = self.K_boltzmann*self.temperature * \
            self.operationBW*1000       # Thermal noise linear (in mW)
        self.Pn_dBm = 10 * np.log10(self.Pn)            # Thermal noise (in dB)
        self.sigma0 = np.sqrt(self.Pn)
        self.sigma = np.sqrt(10**(self.noise_figure/10)) * self.sigma0

    def update_geometry(self):
        self.NR = np.prod(self.NR_dim, 0)
        self.NB = np.prod(self.NB_dim)
        self.NU = np.prod(self.NU_dim)

        self.LR = self.PR.shape[1]
        self.L = self.LR + 1

        self.RB = to_rotm(self.OB.T)
        self.tB = self.RB @ np.array([1, 0, 0]).reshape(-1, 1)
        self.B0 = self.dant * get_array_layout(self.NB_dim)
        self.B = self.PB + self.RB @ self.B0

        self.RU = to_rotm(self.OU.T)
        self.tU = self.RU @ np.array([1, 0, 0]).reshape(-1, 1)
        self.U0 = self.dant * get_array_layout(self.NU_dim)
        self.U = self.PU + self.RU @ self.U0

        if self.LR > 0:
            # TODO: Optimize using 3D tensor
            # List of rotation matrixes
            self.RR = [None for _ in range(self.LR)]
            self.tR = np.zeros((3, self.LR))
            # List of local AE positions
            self.R0 = [None for _ in range(self.LR)]
            # List of global AE positions
            self.R = [None for _ in range(self.LR)]

            for i in range(self.LR):
                self.RR[i] = to_rotm(self.OR[:, i].T)
                self.tR[:, i] = self.RR[i][:, 0]
                self.R0[i] = self.dant * get_array_layout(self.NR_dim[:, i])
                self.R[i] = self.PR[:, i][:, np.newaxis] + \
                    self.RR[i] @ self.R0[i]

        # LOS Channel

        self.dBU = np.linalg.norm(self.PB - self.PU)
        self.tauBU = self.dBU / self.c
        self.dL = self.dBU + self.beta
        self.tauL = (self.dBU + self.beta) / self.c

        # global DOD/DOA: global = Rotm*local; local = Rotm^-1*global
        self.tBU = (self.PU - self.PB) / self.dBU
        self.tUB = -self.tBU
        self.phiBU, self.thetaBU = get_angle_from_dir(self.tBU)

        # local DOD/DOA: global = Rotm*local; local = Rotm^-1*global
        self.tBU_loc = self.RB.T @ self.tBU
        self.phiBU_loc, self.thetaBU_loc = get_angle_from_dir(self.tBU_loc)
        self.tUB_loc = self.RU.T @ (-self.tBU)
        self.phiUB_loc, self.thetaUB_loc = get_angle_from_dir(self.tUB_loc)
        self.rhoL = 0

        # RIS Channel
        if self.LR > 0:
            self.dBR = np.linalg.norm(self.PB - self.PR, axis=0)
            self.dRB = self.dBR
            self.dRU = np.linalg.norm(self.PU - self.PR, axis=0)
            self.dUR = self.dRU
            self.dBRU = self.dBR + self.dRU
            # Delay of the LOS path, in [m]
            self.dR = self.dBRU + self.beta
            self.tauBRU = self.dBRU / self.c
            # Signal delay, propagation time + offset
            self.tauR = (self.dBRU + self.beta) / self.c

            # global DOD/DOA: global = Rotm*local; local = Rotm^-1*global
            self.tBR = (self.PR-self.PB) / self.dBR
            self.tRB = -self.tBR
            self.tUR = (self.PR-self.PU) / self.dUR
            self.tRU = -self.tUR

            self.phiBR, self.thetaBR = get_angle_from_dir(self.tBR)
            self.phiUR, self.thetaUR = get_angle_from_dir(self.tUR)

            # local DOD/DOA: global = Rotm*local; local = rotm^-1*global
            # Unit direction vector (local) from Tx to Rx
            self.tBR_loc = self.RB.T @ self.tBR
            self.phiBR_loc, self.thetaBR_loc = get_angle_from_dir(self.tBR_loc)

            # Unit direction vector (local) from Tx to Rx
            self.tUR_loc = self.RU.T @ self.tUR
            self.phiUR_loc, self.thetaUR_loc = get_angle_from_dir(self.tUR_loc)

            self.tRB_loc = np.zeros((3, self.LR))
            self.tRU_loc = np.zeros((3, self.LR))

            for i in range(self.LR):
                # Unit direction vector (local) from Tx to Rx
                self.tRB_loc[:, i] = self.RR[i].T @ self.tRB[:, i]
                # Unit direction vector (local) from Tx to Rx
                self.tRU_loc[:, i] = self.RR[i].T @ self.tRU[:, i]

            self.phiRB_loc, self.thetaRB_loc = get_angle_from_dir(self.tRB_loc)
            self.phiRU_loc, self.thetaRU_loc = get_angle_from_dir(self.tRU_loc)

            self.rhoR = np.zeros((1, self.LR))
            self.xiR = np.zeros((1, self.LR))

    def get_beam_matrix(self):
        """Get the combining matrix and precoding matrix for hybrid MIMO.

            For fully digital array: W is a diagonal matrix with all 1s diagonal elements.

            # Random: Generate directional beams pointing to [phi, theta]
            # Directional: Generate directional beams pointing to [phi, theta]
            # Derivative: Generative derivative beams pointing to [phi, theta]
            # Customized: Do nothing...
        """
        self.WU_mat = np.zeros((self.NU, self.MU, self.G), dtype='complex_')
        self.WB_mat = np.zeros((self.NB, self.MB, self.G), dtype='complex_')

        # TODO: Optimize
        if self.array_type == ArrayType.Digital:
            for g in range(self.G):
                WU = np.eye(self.NU)
                WB = np.eye(self.NB)
                self.WU_mat[:, :, g] = WU
                self.WB_mat[:, :, g] = WB

        elif self.beam_type == BeamType.Random:
            for g in range(self.G):
                WU = np.exp(2j * np.pi * rand(self.NU, self.MU)) / \
                    np.sqrt(self.NU)
                self.WU_mat[:, :, g] = WU

            # Do the loop twice to match the order of the Matlab code for Debugging purposes.
            for g in range(self.G):
                WB = np.exp(2j * np.pi * rand(self.NB, self.MB)) / \
                    np.sqrt(self.NB)
                self.WB_mat[:, :, g] = WB

        if self.ris_profile_type == RisProfileType.Random:
            for i in range(self.LR):
                self.omega[i] = np.exp(2j * np.pi * rand(self.NR[i], self.G))

    def get_tx_symbol(self):
        """ Get the transmitted smybols (after precoder)

            For fully diagonal array: W is a diagonal matrix with all 1s diagonal elements.
            Random: Generate random phase shifter coefficients.
            Directional: Generate directional beams pointing to [phi, theta].
            Derivative: Generate derivative beams pointing to [phi, theta].
            Customized: Do nothing...

            symbol: s = M x K (# of RFCs x # or subcarriers)
            symbol cell: LU x G (# of UE x)
            precoder/combiner: W = N x M x G (N x M x K x G if beamsplit/widband is considered)

            Notes: PA for each antenna, TX power is calculated based Xub/Xuk
        """

        # Each cell has size N x Ks
        self.XU_mat = np.zeros((self.NU, self.K, self.G), dtype='complex_')

        # TODO: Optimize
        if self.beam_type == BeamType.Random:
            for g in range(self.G):
                XU0 = np.exp(2j * np.pi * rand(self.MU, self.K))
                WU = self.WU_mat[:, :, g]
                XU = WU * XU0
                self.XU_mat[:, :, g] = XU / np.linalg.norm(XU, axis=0)

    def get_path_parameters_PWM(self):
        self.get_channel_matrix()       # Return H_cell
        self.get_rx_symbols()           # Return muB_cell, muU_cell
        #self.get_D_mu_channel_parameters()      # Return D_muB_cell, D_muU_cell

        self.D_muB_cell = get_D_mu_channel_parameters(
            self.L, self.path_info, self.class_index, self.muB_cell, self.rho_cell, self.doppler_cell, self.H_cell,
            self.MB, self.K, self.G, self.link_type, self.XU_mat, self.WB_mat, self.lambdac, self.fdk, self.c, 
            self.AstRB_cell, self.AstRU_cell, self.phiRU_loc, self.thetaRU_loc, self.omega, self.beamsplit_coe, self.R0
        )

        self.get_jacobian_matrix()

    def get_channel_matrix(self):
        """ Get channel matrices for all the c.L paths

            order: LOS, RIS, Reflection NLOS, scattering NLOS
        """
        # lp: index of current path
        # lc: index of path in current type (e.g., lc-th RIS path)

        # TODO: Optimize
        # These probably do not need to be defined here since we just overwrite them anyways.
        # Define them once in __init__ and just write to each cell.
        self.alpha_cell = [None for _ in range(self.L)]
        self.rho_cell = [None for _ in range(self.L)]
        self.Xi_cell = [None for _ in range(self.L)]
        self.v_cell = [None for _ in range(self.L)]
        self.H_cell = [None for _ in range(self.L)]

        # Steering vector from BS to UE, RIS, IP
        self.AstBX_cell = [None for _ in range(self.L)]
        self.AstUX_cell = [None for _ in range(self.L)]
        self.AstRB_cell = [None for _ in range(self.L)]
        self.AstRU_cell = [None for _ in range(self.L)]

        pi_2j = 2j * np.pi

        for lp in range(self.L):
            curr_type = self.path_info[lp]
            lc = self.class_index[lp]       # Index of the same class

            # TODO: Optimize, put these outside of loop, no need to reinit them
            H = np.zeros((self.NB, self.NU, self.K),
                         dtype='complex_')      # Channel matrix
            alpha = np.zeros(self.K, dtype='complex_')
            AstBU = np.zeros((self.NB, self.K), dtype='complex_')
            AstUB = np.zeros((self.NU, self.K), dtype='complex_')
            AstBR = np.zeros((self.NB, self.K), dtype='complex_')
            AstRB = np.zeros((self.NR[lc], self.K), dtype='complex_')
            AstRU = np.zeros((self.NR[lc], self.K), dtype='complex_')
            AstUR = np.zeros((self.NU, self.K), dtype='complex_')

            # LOS channel
            if curr_type == PathType.L:
                # Channel gain (antenna directionality sin(theta))
                rho = self.beamsplit_coe * self.lambdac / 4 / np.pi / \
                    self.dBU      # TODO: Optimize: Extract this parameter
                xi = -self.dL

                # TODO: Extract -2j*pi, it gets computed very often
                # TODO: Vectorize
                # Delay part e^(-2pij * fdk(k) * tauL)
                Xi = np.exp(-pi_2j * self.fdk * self.tauL)
                for k in range(self.K):
                    factor = pi_2j / (self.lambdac * self.beamsplit_coe[k])
                    # Complex channel gain of the LOS path
                    alpha[k] = rho[k] * np.exp(xi * factor)
                    # Steering vector of BU
                    AstBU[:, k] = np.exp(factor * (self.B0.T @ self.tBU_loc))
                    # Steering vector of UB
                    AstUB[:, k] = np.exp(factor * (self.U0.T @ self.tUB_loc))
                    H[:, :, k] = ((alpha[k] * Xi[k]) * AstBU) @ AstUB.T

                self.AstBX_cell[lp] = AstBU
                self.AstUX_cell[lp] = AstUB

            # RIS channel
            elif curr_type == PathType.R:
                # Element gain (antenna directionality cos(theta))
                # TODO: Extract constant
                rho = (self.beamsplit_coe*self.lambdac/4 /
                       np.pi)**2/self.dBR[lc]/self.dRU[lc]
                xi = -self.dR[lc]

                # Delay part e^(-2pij * dfk(k) * tauR)
                Xi = np.exp(-pi_2j*(self.fdk*self.tauR[lc]))
                for k in range(self.K):
                    factor = pi_2j / (self.lambdac * self.beamsplit_coe[k])
                    alpha[k] = rho[k] * np.exp(xi * factor)
                    # Steering vector of BR
                    AstBR[:, k] = np.exp(
                        factor * (self.B0.T @ self.tBR_loc[:, lc]))
                    # Steering vector of RB
                    AstRB[:, k] = np.exp(
                        factor * (self.R0[lc].T @ self.tRB_loc[:, lc]))
                    # Steering vector of RU
                    AstRU[:, k] = np.exp(
                        factor * (self.R0[lc].T @ self.tRU_loc[:, lc]))
                    # Steering vector or UR
                    AstUR[:, k] = np.exp(
                        factor * (self.U0.T @ self.tUR_loc[:, lc]))
                    # HR without coeffecients
                    H[:, :, k] = ((alpha[k] * Xi[k]) * AstBR) @ AstUR.T

                # Note: This RIS channel does not consider the RIS coeffecient
                # See complete RIS channel in get_rx_symbols_per_path
                self.AstBX_cell[lp] = AstBR
                self.AstRB_cell[lp] = AstRB
                self.AstRU_cell[lp] = AstRU
                self.AstUX_cell[lp] = AstUR

            self.alpha_cell[lp] = alpha
            self.rho_cell[lp] = rho
            self.Xi_cell[lp] = Xi
            self.H_cell[lp] = np.sqrt(self.P) * H

    def get_rx_symbols(self):
        # TODO: Optimize,
        # There should be no need to init these symbols, since they get overwritten here anyways
        # Also, none of the variables in the function call need to be stored in the parameter, they always get overwritten

        # Received symbols at BS.
        #self.muB_cell = [None for _ in range(self.L)]
        #self.doppler_cell = [None for _ in range(self.L)]

        for lp in range(self.L):
            curr_type = self.path_info[lp]
            lc = self.class_index[lp]       # index of the same class
            muB = np.zeros((self.MB, self.K, self.G), dtype='complex_')
            # without considering Doppler: set as ones
            doppler_mat = np.ones((self.K, self.G))
            H = self.H_cell[lp]

            AstRB = self.AstRB_cell[lp]
            AstRU = self.AstRU_cell[lp]

            if curr_type == PathType.R:
                ris_prefix = (AstRB[:, 0] * AstRU[:, 0]).T

            for g in range(self.G):
                # Just one user
                XUg = self.XU_mat[:, :, g]
                WB = self.WB_mat[:, :, g]

                # RIS coefficient component
                ris_g = (ris_prefix @ self.omega[lc][:, g]
                         if (curr_type == PathType.R)
                         else 1)

                # Uplink channel
                for k in range(self.K):
                    muB[:, k, g] = WB.T @ H[:, :, k] @ XUg[:, k] * ris_g

            self.muB_cell[lp] = muB
            self.doppler_cell[lp] = doppler_mat


    def get_jacobian_matrix(self):
        """ Get the Jacobian matrix from path parametser to unknowns """
        self.JM_cell = [None for _ in range(self.L)]

        for lp in range(self.L):
            curr_type = self.path_info[lp]
            lc = self.class_index[lp]

            # LOS: size(J) = 6x3
            if curr_type == PathType.L:
                D_PU_dL = (self.PU - self.PB) / self.dBU
                D_PU_rhoL = np.zeros((3, 1))
                D_PU_xiL = np.zeros((3, 1))
                JPU_LOS = np.hstack((D_PU_dL, D_PU_rhoL, D_PU_xiL))
                J = np.zeros((6, 3))

                # row: 3PU, 1C, 2Gains
                # col: 1T, 2Gain
                J[:3, :] = JPU_LOS
                J[(3, 4, 5), (0, 1, 2)] = 1         # Clock offset

            # RIS: size(J) = 6x5
            elif curr_type == PathType.R:
                tRU_loc = self.tRU_loc[:, lc]
                dUR = self.dUR[lc]
                dRU = dUR
                PR = self.PR[:, lc]
                tRU = self.tRU[:, lc]
                RotR = self.RR[lc]

                D_PU_phiRU_loc = (tRU_loc[0] * RotR[:, 1] - tRU_loc[1]
                                  * RotR[:, 0]) / (tRU_loc[0]**2 + tRU_loc[1]**2) / dRU
                D_PU_thetaRU_loc = (
                    RotR[:, 2] - tRU_loc[2]*tRU) / np.sqrt(1 - tRU_loc[2] ** 2) / dRU
                D_PU_dR = (self.PU.flatten() - PR) / dRU
                D_PU_rhoR = np.zeros(3)
                D_PU_xiR = np.zeros(3)

                JPU_RIS = np.vstack(
                    (D_PU_phiRU_loc, D_PU_thetaRU_loc, D_PU_dR, D_PU_rhoR, D_PU_xiR))
                J = np.zeros((6, 5))
                J[:3, :] = JPU_RIS.T
                J[(3, 4, 5), (2, 3, 4)] = 1        # Clock offset

            self.JM_cell[lp] = J

    def get_FIM_PWM(self):
        """
            output:
                self.FIM_M: FIM of the geometric channel parameters
                self.FIM: FIM of unknown states
        """

        # FIM for measurement vector
        FIM_M_cell = [None for _ in range(self.G)]
        FIM_S_cell = [None for _ in range(self.G)]      # FIM for state vector

        # Number of unknowns (including complexx channel gain)
        self.N_unknowns = 6 + 2 * self.LR

        # Number of measurements (LOS + RIS channels)
        self.N_measures = 3 + 5 * self.LR

        JS_all = np.zeros((self.N_unknowns, self.N_measures))

        JS_all[:6, :3] = self.JM_cell[0][:6, :]
        for lp in range(1, self.L):
            lc = lp - 1
            col_ind = 3 + 5 * lc + np.arange(5)
            row_ind = 6 + 2 * lc + np.arange(2)

            J_temp = self.JM_cell[lp]
            JS_all[:4, col_ind] = J_temp[:4, :]     # PU, OU, VU
            JS_all[np.ix_(row_ind, col_ind)] = J_temp[4:6, :]
        JS = JS_all

        # Get FIM
        # TODO: Optimize, remove this initialization
        # FIM of the measurement vector
        self.FIM_M = np.zeros((self.N_measures, self.N_measures))
        self.FIM = np.zeros((self.N_unknowns, self.N_unknowns))

        for g in range(self.G):
            FIM_M = np.zeros((self.N_measures, self.N_measures))

            for k in range(self.K):
                # Combine derivative from all paths
                # TODO: Optimize, preallocate list

                D_mu_Mat_li = []
                for lp in range(self.L):
                    D_muB = self.D_muB_cell[lp][:, :, k, g]
                    D_mu_Mat_li.append(D_muB)
                D_mu_Mat = np.matrix(np.hstack(D_mu_Mat_li))
                
                I_ns_k = D_mu_Mat.H @ D_mu_Mat
                FIM_M += 2 / self.sigma**2 * np.real(I_ns_k)

            FIM_M_cell[g] = FIM_M
            FIM_S_cell[g] = JS @ FIM_M @ JS.T

            self.FIM_M += FIM_M_cell[g]

        self.FIM = JS @ self.FIM_M @ JS.T

        self.JS_all = JS_all
        self.JS = JS

    def get_crlb_from_fim_PWM(self, FIM):
        """ Get CRLB, position error bound and clock-bias error bound """
        # 4 UE states to be estimated from: 3D position and 1D clock offset

        EFIM = get_EFIM_from_FIM(FIM, 4)
        CRLB = np.linalg.inv(EFIM)

        self.PEB, self.CEB = get_PEB_and_CEB(CRLB)

    def get_blockage(self, point):
        block_vec = np.zeros(self.L)
        for i in range(self.L):     # Iterate all the paths
            for j in range(len(self.Wall)):
                # TODO: Optimize
                L1 = self.Wall[j].T
                L2 = np.vstack((self.Anchor[:, i], point)).T

                xi, _ = get_linexline(L1[0], L1[1], L2[0], L2[1])

                if not np.isnan(xi):
                    block_vec[i] += 1

        return block_vec > 0

    def get_crlb_blockage(self, blockage):
        # TODO: Vectorize
        row_ind = [np.arange(4)]
        col_ind = []

        if blockage[0]:
            col_ind.append(np.arange(3))
        else:
            row_ind.append(4 + np.arange(2))

        for i in range(1, len(blockage)):
            if blockage[i]:
                col_ind.append(3 + (i-1)*5 + np.arange(5))
            else:
                row_ind.append(4 + i*2 + np.arange(2))

        row_ind = np.hstack(row_ind)
        JS1 = self.JS[row_ind, :]

        if len(col_ind) > 0:
            col_ind = np.hstack(col_ind)
            JS1[:, col_ind] = 0

        FIM1 = JS1 @ self.FIM_M @ JS1.T
        EFIM = get_EFIM_from_FIM(FIM1, 4)

        if not is_invertible(EFIM):
            self.PEB = self.CEB = np.Inf
            return

        CRLB = np.linalg.inv(EFIM)
        self.PEB, self.CEB = get_PEB_and_CEB(CRLB)

    def get_PEB_cell(self, xgrid, ygrid, parallel=True, verbosity=1, print_runtimes=True):
        if parallel:
            cs = []
            PUs = []

            for xi in range(xgrid.size):
                for yi in range(ygrid.size):
                    c = self.copy()
                    PU = np.array([xgrid[xi], ygrid[yi], 1]).reshape(-1, 1)

                    cs.append(c)
                    PUs.append(PU)

            t_start = time.time()
            with mp.Pool() as pool:
                PEBs = pool.starmap(get_PEB_from_PU, zip(cs, PUs))
            t_end = time.time()

            PEB_cells = np.array(PEBs, dtype='complex_').reshape(
                xgrid.size, ygrid.size)

            if print_runtimes:
                print(f"Total Runtime:\t{t_end-t_start:.3f} [s]")


        else:
            runtimes = np.zeros((xgrid.size, ygrid.size))
            PEB_cells = np.zeros((xgrid.size, ygrid.size), dtype='complex_')
            for xi in range(xgrid.size):
                if verbosity >= 1:
                    print(f"xi: {xi}/{xgrid.size}")

                for yi in range(ygrid.size):
                    if verbosity >= 2:
                        print(f"yi: {yi}/{ygrid.size}")

                    c = self.copy()
                    PU = np.array([xgrid[xi], ygrid[yi], 1]).reshape(-1, 1)

                    t_start = time.time()
                    PEB_cells[xi, yi] = get_PEB_from_PU(c, PU)
                    t_end = time.time()
                    runtimes[xi, yi] = t_end - t_start

            if print_runtimes:
                print(f"Total Runtime:\t{runtimes.sum():.3f} [s]")
                print(f"Runtime per X:\t{runtimes.sum(axis=0).mean():.3f} ± {runtimes.sum(axis=0).std():.3f} [s]")
                print(f"Runtime per Y:\t{runtimes.mean():.3f} ± {runtimes.std():.3f} [s]")   

        return PEB_cells

    def get_PEB_CDF(self, PEB_cell):
        PEB_mat = PEB_cell.copy()
        PEB_mat[np.isnan(PEB_cell) | (np.abs(PEB_cell) > 100)] = 100

        Z = PEB_mat.copy().T
        Z[Z == 100] = 10

        error_grid = 10.0 ** np.linspace(-2, 0, 100)
        PEB_CDF = get_cdf(Z.T, error_grid)

        return error_grid, PEB_CDF
    
    def get_CDF_threshold(self, error_grid, PEB_CDF, threshold=0.9):
        return error_grid[np.argmax(PEB_CDF>=threshold)]

    def plot_scene(self):
        self.plot_walls()
        self.plot_PB()
        self.plot_PR()

    def plot_walls(self):
        for wall in self.Wall:
            plt.plot(wall[:, 0], wall[:, 1], c='black', linewidth=5)

    def plot_PB(self):
        plt.scatter(self.PB[0], self.PB[1], marker='x', c='r', s=200)

    def plot_PR(self):
        plt.scatter(self.PR[0], self.PR[1], marker='o', c='r', s=200)

    def plot_PEB(self, PEB_cell, margin=0.25, cbar_shrink=0.65):
        PEB_mat = PEB_cell.copy()
        PEB_mat[np.isnan(PEB_cell) | (np.abs(PEB_cell) > 100)] = 100

        setup_fig()
        plt.title('PEB of ... []')
        img = plt.imshow(PEB_mat.T.real, origin='lower', norm=mpl.colors.LogNorm(vmin=0.01, vmax=1), extent=[-5-margin, 5+margin, -margin, 5+margin])
        plt.colorbar(img, pad=0.04, shrink=cbar_shrink, aspect=20*cbar_shrink)
        self.plot_scene()

    def plot_CDF_PEB(self, error_grid, PEB_CDF):
        plt.scatter(error_grid, PEB_CDF)
        plt.xscale('log')
        plt.ylabel(r'Percentage of error < $\epsilon$')
        plt.xlabel(r'Error $\epsilon$ [m]')
        plt.grid(True, which='both')

    def copy(self):
        return copy.deepcopy(self)
