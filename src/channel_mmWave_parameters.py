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
from dataclasses import dataclass
from enum import Enum, auto, unique
from .utils import db2pow, to_rotm, get_angle_from_dir


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


@dataclass(init=True)
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


        seed=0          # Random seed
    ):
        self.seed = seed
        self.rng = np.random.default_rng(seed=self.seed)

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
        self.NU_dim: np.ndarray = NU_dim

        # Number of elements at BS/RIS/UE
        # Number of antennas for conventional array
        self.NB: int = np.prod(self.NB_dim, 0)    # Number of BS elements
        self.NR: int = np.prod(self.NR_dim, 0)    # Number of RIS elements
        self.NU: int = np.prod(self.NU_dim, 0)    # Number of UE elements

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

        self.G: float = G

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

        self.RB: np.ndarray = to_rotm(self.OB.T)    # Rotation matrix from euler angles
        self.tB: np.ndarray = self.RB @ np.array([1, 0, 0]).reshape(-1, 1)
        self.B0: np.ndarray = self.dant * get_array_layout(self.NB_dim)    # Local AE position
        self.B: np.ndarray = self.PB + self.RB @ self.B0     # Global AE position

        self.RU: np.ndarray = to_rotm(self.OU.T)    # Rotation matrix from euler angles
        self.tU: np.ndarray = self.RU @ np.array([1, 0, 0]).reshape(-1, 1)
        self.U0: np.ndarray = self.dant * get_array_layout(self.NU_dim)     # Local AE position
        self.U: np.ndarray = self.PU + self.RU @ self.U0    # Global AE position

        # TODO: Optimize using 3D tensor
        self.RR: list[np.ndarray] = []      # List of rotation matrixes
        self.tR: np.ndarray = np.zeros((3, self.LR))
        self.R0: list[np.ndarray] = []      # List of local AE positions
        self.R: list[np.ndarray] = []       # List of global AE positions

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
        self.tBU_loc: np.ndarray = self.RB.T @ self.tBU     # Unit direction vector (local) from Tx to Rx
        self.phiBU_loc: float = 0
        self.thetaBU_loc: float = 0
        self.tUB_loc: np.ndarray = self.RU.T @ (-self.tBU)  # Unit direction vector (local) from Tx to RX
        self.phiUB_loc: float = 0
        self.thetaUB_loc: float = 0
        self.rhoL: float = 0
        self.xiL: float = 0

        # RIS Channel

        self.dBR: np.ndarray = np.linalg.norm(self.PB - self.PR, axis=0)
        self.dRB: np.ndarray = self.dBR
        self.dRU: np.ndarray = np.linalg.norm(self.PU - self.PR, axis=0)
        self.dUR: np.ndarray = self.dRU
        self.dBRU: np.ndarray = self.dBR + self.dRU
        self.dR: np.ndarray = self.dBRU + self.beta      # Delay of the LOS path, in [m]
        self.tauBRU: np.ndarray = self.dBRU / self.c
        self.tauR: np.ndarray = (self.dBRU + self.beta) / self.c     # Signal delay, propagation time + offset
        
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
        self.tBR_loc: np.ndarray = self.RB.T @ self.tBR     # Unit direction vector (local) from Tx to Rx
        self.phiBR_loc: float = 0
        self.thetaBR_loc: float = 0

        self.tUR_loc: np.ndarray = self.RU.T @ self.tUR     # Unit direction vector (local) from Tx to Rx
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

        self.path_type: list[PathType] = []
        self.path_info: list[PathType] = []

        self.class_index: np.ndarray = np.ones(self.L)

        self.WU_mat: np.ndarray = np.zeros(self.NU, self.MU, self.G)
        self.WB_mat: np.ndarray = np.zeros(self.NB, self.MB, self.G)
        self.omega: list[np.ndarray] = [None for _ in range(self.LR)]

        self.XU_mat: np.ndarray = np.zeros(self.NU, self.K, self.G)     # Each cell has size N x Ks

        self.alpha_cell: list[np.ndarray] = [None for _ in range(self.L)]
        self.rho_cell: list[np.ndarray] = [None for _ in range(self.L)]
        self.Xi_cell: list[np.ndarray] = [None for _ in range(self.L)]
        self.v_cell: list[np.ndarray] = [None for _ in range(self.L)]
        self.H_cell: list[np.ndarray] = [None for _ in range(self.L)]

        # Steering vector from BS to UE, RIS, IP
        self.AstBX_cell: list[np.ndarray] = [None for _ in range(self.L)]
        self.AstUX_cell: list[np.ndarray] = [None for _ in range(self.L)]
        self.AstRB_cell: list[np.ndarray] = [None for _ in range(self.L)]
        self.AstRU_cell: list[np.ndarray] = [None for _ in range(self.L)]

        # Received symbols at BS
        self.muB_cell: list[np.ndarray] = [None for _ in range(self.L)] 
        self.doppler_cell: list[np.ndarray] = [None for _ in range(self.L)]
        self.XUg: np.ndarray = np.zeros((self.NU, self.K))
        self.WU: np.ndarray = np.zeros((self.NU, self.MU))
        self.WB: np.ndarray = np.zeros((self.NB, self.MB))

        self.D_muB_cell: list[np.ndarray] = [None for _ in range(self.L)]
        self.D_muB_UR_cell: list[np.ndarray] = [None for _ in range(self.LR)]
        self.muB: np.ndarray = np.zeros((self.MB, self.K, self.G))

        self.update_parameters()

    def update_parameters(self, args: UpdateArgsType = UpdateArgsType.All):
        # Update Signal parameters
        if args == UpdateArgsType.All or args == UpdateArgsType.Signal:
            self.lambdac = self.c/self.fc
            self.dant = self.lambdac/2
            self.fdk = (
                -self.BW/2 +
                 self.BW / (2*self.K) + 
                 (self.BW/self.K)*np.arange(self.K).T
            )
            self.fk = self.fdk + self.fc
            self.lambdak = self.c / self.fk
            self.beamsplit_coe = np.ones(len(self.lambdak))

        # Update Geometry parameters
        if args == UpdateArgsType.All or args == UpdateArgsType.Geometry:
            self.update_geometry()

            self.N_measures = 3 + 5 * self.LR
            self.N_unknowns = 6 + 2 * self.LR

            # TODO: Optimize
            self.path_type = [PathType.R for _ in range(self.L)]
            self.path_info = [PathType.R for _ in range(self.L)]
            self.path_type[0] = PathType.L
            self.path_info[0] = PathType.L

            self.class_index = np.zeros(self.L)
            for lp in range(self.L):
                if self.path_type[lp] == PathType.R:
                    self.class_index[lp] = lp-1

        # Update environment parameters
        self.operationBW = self.BW
        self.Pn = self.K_boltzmann*self.temperature*self.operationBW*1000       # Thermal noise linear (in mW)
        self.Pn_dBm = 10 * np.log10(self.Pn)            # Thermal noise (in dB)
        self.sigma0 = np.sqrt(self.Pn)
        self.sigma = np.sqrt(10**(self.noise_figure/10)) * self.sigma0

            

    def update_geometry(self):
        self.NB = np.prod(self.NB_dim)
        self.NR = np.prod(self.NR_dim)
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
            self.RR = [None for _ in range(self.LR)]      # List of rotation matrixes
            self.tR = np.zeros((3, self.LR))
            self.R0 = [None for _ in range(self.LR)]      # List of local AE positions
            self.R = [None for _ in range(self.LR)]       # List of global AE positions 

            for i in range(self.LR):
                self.RR[i] = to_rotm(self.OR[:, i].T)
                self.tR[:, i] = self.RR[i][:, 0] 
                self.R0[i] = self.dant * get_array_layout(self.NR_dim[:, i])
                self.R[i] = self.PR[:, i][:, np.newaxis] + self.RR[i] @ self.R0[i]

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
        self.xiL = 0

        # RIS Channel
        if self.LR > 0:
            self.dBR = np.linalg.norm(self.PB - self.PR, axis=0)
            self.dRB = self.dBR
            self.dRU = np.linalg.norm(self.PU - self.PR, axis=0)
            self.dUR = self.dRU
            self.dBRU = self.dBR + self.dRU
            self.dR = self.dBRU + self.beta      # Delay of the LOS path, in [m]
            self.tauBRU = self.dBRU / self.c
            self.tauR = (self.dBRU + self.beta) / self.c     # Signal delay, propagation time + offset
            
            # global DOD/DOA: global = Rotm*local; local = Rotm^-1*global 
            self.tBR = (self.PR-self.PB) / self.dBR
            self.tRB = -self.tBR
            self.tUR = (self.PR-self.PU) / self.dUR
            self.tRU = -self.tUR

            self.phiBR, self.thetaBR = get_angle_from_dir(self.tBR)
            self.phiUR, self.thetaUR = get_angle_from_dir(self.tUR)

            # local DOD/DOA: global = Rotm*local; local = rotm^-1*global
            self.tBR_loc = self.RB.T @ self.tBR     # Unit direction vector (local) from Tx to Rx
            self.phiBR_loc, self.thetaBR_loc = get_angle_from_dir(self.tBR_loc)

            self.tUR_loc = self.RU.T @ self.tUR     # Unit direction vector (local) from Tx to Rx
            self.phiUR_loc, self.thetaUR_loc = get_angle_from_dir(self.tUR_loc)

            self.tRB_loc = np.zeros((3, self.LR))
            self.tRU_loc = np.zeros((3, self.LR))

            for i in range(self.LR):
                self.tRB_loc[:, i] = self.RR[i].T @ self.tRB[:, i]  # Unit direction vector (local) from Tx to Rx
                self.tRU_loc[:, i] = self.RR[i].T @ self.tRU[:, i]  # Unit direction vector (local) from Tx to Rx

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

        self.WU_mat = np.zeros(self.NU, self.MU, self.G)
        self.WB_mat = np.zeros(self.NB, self.MB, self.G)

        # TODO: Optimize
        if self.array_type == ArrayType.Digital:
            for g in range(self.G):
                WU = np.eye(self.NU)
                WB = np.eye(self.NB)
                self.WU_mat[:, :, g] = WU
                self.WB_mat[:, :, g] = WB

        elif self.beam_type == BeamType.Random:
            for g in range(self.G):
                WU = np.exp(2j * np.pi * self.rng.uniform(size=(self.NU, self.MU)))/np.sqrt(self.NU)
                WB = np.exp(2j * np.pi * self.rng.uniform(size=(self.NU, self.MU)))/np.sqrt(self.NB)
                self.WU_mat[:, :, g] = WU
                self.WB_mat[:, :, g] = WB

        if self.ris_profile_type == RisProfileType.Random:
            for i in range(self.LR):
                self.omega[i] = np.exp(2j * np.pi * self.rng.uniform(size=(self.NR[i], self.G)))

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
        
        self.XU_mat = np.zeros(self.NU, self.K, self.G)     # Each cell has size N x Ks

        # TODO: Optimize
        if self.beam_type == BeamType.Random:
            for g in range(self.G):
                XU0 = np.exp(2j * np.PI * self.rng.uniform(size=(self.MU, self.K)))
                WU = self.WU_mat[:, :, g]
                XU = WU * XU0
                self.XU_matrix[:, :, g] = XU / np.linalg.norm(XU, axis=0)


    def get_path_parameters_PWM(self):
        self.get_channel_matrix()
        self.get_rx_symbols()
        self.get_D_mu_channel_parameters()
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
            H = np.zeros((self.NB, self.NU, self.K))      # Channel matrix
            alpha = np.zeros(self.K)
            AstBU = np.zeros((self.NB, self.K))
            AstUB = np.zeros((self.NU, self.K))
            AstBR = np.zeros((self.NB, self.K))
            AstRB = np.zeros((self.NR[lc], self.K))
            AstRU = np.zeros((self.NR[lc], self.K))
            AstUR = np.zeros((self.NU, self.K))

            # LOS channel
            if curr_type == PathType.L:
                # Channel gain (antenna directionality sin(theta))
                rho = self.beamsplit_coe * self.lambdac / 4 / np.pi / self.dBU      # TODO: Optimize: Extract this parameter
                xi = -self.dL

                # TODO: Extract -2j*pi, it gets computed very often
                # TODO: Vectorize
                Xi = np.exp(-pi_2j * self.fdk * self.tauL)       # Delay part e^(-2pij * fdk(k) * tauL)
                for k in range(self.K):
                    factor = pi_2j / (self.lambdac * self.beamsplit_coe[k])
                    alpha[k] = rho[k] * np.exp(xi * factor)        # Complex channel gain of the LOS path
                    AstBU[:, k] = np.exp(factor * (self.B0.T @ self.tBU_loc))   # Steering vector of BU
                    AstUB[:, k] = np.exp(factor * (self.U0.T @ self.tUB_loc))   # Steering vector of UB

                self.AstBX_cell[lp] = AstBU
                self.AstUX_cell[lp] = AstUB

            # RIS channel
            elif curr_type == PathType.R:
                # Element gain (antenna directionality cos(theta))
                # TODO: Extract constant
                rho = (self.beamsplit_coe*self.lambdac/4/np.pi)**2/self.dBR[lc]/self.dRU[lc]
                xi = -self.dR[lc]

                Xi = np.exp(-pi_2j*(self.fdk*self.tauR[lc]))     # Delay part e^(-2pij * dfk(k) * tauR)
                for k in range(self.K):
                    factor = pi_2j / (self.lambdac * self.beamsplit_coe[k])
                    alpha[k] = rho[k] * np.exp(xi * factor)
                    AstBR[:, k] = np.exp(factor * (self.B0.T @ self.tBR_loc[:, lc]))        # Steering vector of BR
                    AstRB[:, k] = np.exp(factor * (self.R0[lc].T @ self.tRB_loc[:, lc]))    # Steering vector of RB
                    AstRU[:, k] = np.exp(factor * (self.R0[lc].T @ self.tRU_loc[:, lc]))    # Steering vector of RU
                    AstUR[:, k] = np.exp(factor * (self.U0.T @ self.tUR_loc[:, lc]))        # Steering vector or UR
                    H[:, :, k] = alpha[k] * Xi[k] * AstBR * AstUR.T     # HR without coeffecients

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
        self.muB_cell = [None for _ in range(self.L)]       
        self.doppler_cell = [None for _ in range(self.L)]

        for lp in range(self.L):
            curr_type = self.path_info[lp]
            lc = self.class_index[lp]       # index of the same class
            muB = np.zeros((self.MB, self.K, self.G))
            doppler_mat = np.ones((self.K, self.G))     # without considering Doppler: set as ones
            H = self.H_cell[lp]

            AstRB = self.AstRB_cell[lp]
            AstRU = self.AstRU_cell[lp]

            for g in range(self.G):
                # Just one user
                self.XUg = self.XU_mat[:, :, g]
                self.WU = self.WU_mat[:, :, g]
                self.WB = self.WB_mat[:, :, g]

                # RIS coefficient component
                if curr_type == PathType.R:     # TODO: Fix flat array transpose bug
                    ris_g = (AstRB[:, 0] * AstRU[:, 0]).T @ self.omega[lc][:, g]
                else:
                    ris_g = 1

                # Uplink channel
                muBg = np.zeros((self.MB, self.K))
                for k in range(self.K):     # TODO: Vectorize
                    muBg[:, k] = self.WB.T @ self.H[:, :, k] * self.XUg[:, k] * ris_g
                muB[:, :, g] = muBg

            self.muB_cell[lp] = muB
            self.doppler_cell[lp] = doppler_mat

    

    def get_D_mu_channel_parameters(self):
        """ Get the FIM of the measurement vector (PWM)

            phiBU, thetaBU, phiUB, thetaUB, tau, v, rho, xi
        """
        # TODO: Optimize by removing initializing of these.
        self.D_muB_cell = [None for _ in range(self.L)]
        self.D_muB_UR_cell = [None for _ in range(self.LR)]

        pi_2j = 2j * np.pi

        for lp in range(self.L):
            curr_type = self.path_info[lp]
            lc = self.class_index[lp]
            self.muB = self.muB_cell[lp]        # TODO: Investigate if this can be made local
            rho = self.rho_cell[lp]
            doppler_mat = self.doppler_cell[lp]
            H = self.H_cell[lp]

            D_muB = np.zeros((self.MB, 3, self.K, self.G))  # 2 Gain, 2 DOA, 2 DOD, 1 tau, 1 velocity (doppler)

            # LOS channel
            if curr_type == PathType.L:
                # Calculate FIM Uplink
                if self.link_type == LinkType.Uplink:
                    for g in range(self.G):
                        self.XUg = self.XU_mat[:, :, g]        # TODO: Investigate if this can be made local
                        self.WB = self.WB_mat[:, :, g]          # This too
                        doppler_k = doppler_mat[:, g].T

                        for k in range(self.K):
                            # BU channel
                            muBg = self.muB[:, k, g]        # received symbols at BS
                            D_muB_rhoBU = muBg / rho[k]
                            D_muB_xiBU = pi_2j / self.lambdac * muBg
                            D_muB_dL = self.WB.T @ 

    def get_jacobian_matrix(self):
        ...