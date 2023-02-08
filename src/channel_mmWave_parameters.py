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
from .utils import db2pow, to_rotm
from scipy.spatial.transform import Rotation


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
        G: float = 10
    ):

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
        self.tB: np.ndarray = self.RB * np.array([1, 0, 0]).reshape(-1, 1)
        self.B0: np.ndarray = self.dant * get_array_layout(self.NB_dim)    # Local AE position
        print("PB SHAPE", self.PB.shape)
        print("RB SHAPE", self.RB.shape)
        print("B0 SHAPE", self.B0.shape)
        self.B: np.ndarray = self.PB + self.RB * self.B0                        # Global AE position

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
            ...

    def update_geometry(self):
        self.NB = np.prod(self.NB_DIM)
        self.NR = np.prod(self.NR_DIM)
        self.NU = np.prod(self.NU_DIM)

        self.LR = self.PB.shape[1]
        self.L = self.LR + 1

