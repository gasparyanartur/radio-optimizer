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
from .utils import db2pow, tr, to_rotm
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


@dataclass(init=True)
class ChannelmmWaveParameters:
    # Model parameters
    syn_type: SynType = SynType.Asyn
    link_type: LinkType = LinkType.Uplink

    # Signal 
    beam_type: BeamType = BeamType.Random
    ris_profile_type: RisProfileType = RisProfileType.Random
    beamforming_angle_std: float = 0        # beamforming angle variation
    array_type: ArrayType = ArrayType.Analog
    radio_directional: float = 1        # ratio of directional beams

    UE_radiation: RadiationType = RadiationType.Omni
    RIS_radiation: RadiationType = RadiationType.Omni
    BS_radiation: RadiationType = RadiationType.Omni

    # Signal parameters
    c: float = 3E8      # speed of light 
    fc: float = 28E9    # carrier frequency
    lambdac: float = c/fc
    BW: float = 400E6   # 100M bandwidth
    P: float = db2pow(20)   # in [mW]. 1000mW = 30dBm. 1mW = 0dBm.
    beta: float = 30    # Synchronization offset (in [meter], 10 us -> 3000 m)
    K: int = 10      # Number of carrier frequencies
    T: float = 1/1000   # Measurement interval [sec] (min value = K/BW)

    # Geometry parameters
    # Global position: 3x1 vector (2D by settings P(3) as zero)

    # BS: Base Station
    PB: np.ndarray = tr(np.array([0, 0, 0]))
    OB: np.ndarray = tr(np.array([90, 0, 0]))

    # UE: User Equipment
    PU: np.ndarray = tr(np.array([5, 2, 0]))
    OU: np.ndarray = tr(np.array([0, 0, 0]))
    VU: np.ndarray = tr(np.array([0, 0, 0]))

    # RIS: Reconfigurable Intelligent Surfaces
    PR: np.ndarray = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    OR: np.ndarray = tr(np.array([0, 0, 0]))
    VR: np.ndarray = tr(np.array([0, 0, 0]))

    # Infrastructure Parameters
    # Number of RFCs at BS and UE
    MB: int = 1 
    MU: int = 1

    # Array Dimensions
    NB_dim: np.ndarray = tr(np.array([4, 4]))
    NR_dim: np.ndarray = tr(np.array([10, 10]))
    NU_dim: np.ndarray = tr(np.array([4, 4]))

    # Number of elements at BS/RIS/UE
    # Number of antennas for conventional array 
    NB: int = np.prod(NB_dim, 0)    # Number of BS elements
    NR: int = np.prod(NR_dim, 0)    # Number of RIS elements
    NU: int = np.prod(NU_dim, 0)    # Number of UE elements

    # Environment Parameters
    operationBW: float = 400E6      # Operation bandwidth for Thermal noise
    K_boltzmann: float = 1.3806E-23 # Boltzmann constant
    temperature: float = 298.15     # Temperature 25 celsius
    Pn: float = K_boltzmann * temperature * operationBW * 1000      # Thermal noise linear (in mW)
    Pn_dBm: float = 10 * np.log10(Pn)       # Thermal noise in dB
    sigma_in: float = np.sqrt(Pn)           # Johnson-Nyquist noise: sigma^2 = N_0
    noise_figure: float = 10                # Noise figure 3dB
    sigma: float = np.sqrt(np.power(10, noise_figure/10)) * sigma_in      # Output noise level

    G: float = 10
    dant: float = lambdac/2     # Half wavelength (default distance between antennas/antenna spacing)
    fdk: np.ndarray = -BW/2 + BW/(2*K) + (BW/K)*tr(np.arange(K))    # Subcarrier frequency
    fk: np.ndarray = fdk + fc
    lambdak: np.ndarray = c / fk     # Wavelength for different subcarriers (wide BW systems)
    beamsplit_coe = np.ones(len(lambdak))
    LR: int = PR.shape[1]   # Number of RISs
    L: int = LR + 1
    RB: np.ndarray = to_rotm(tr(OB))    # Rotation matrix from euler angles
    tB: np.ndarray = RB * tr(np.array([1, 0, 0]))
    B0: np.ndarray = dant * get_array_layout(NB_dim)    # Local AE position
    B: np.ndarray = PB + RB * B0                        # Global AE position


    def update_parameters(self, args: UpdateArgsType = UpdateArgsType.All):
        # Update Signal parameters
        if args == UpdateArgsType.All or args == UpdateArgsType.Signal:
            self.lambdac = self.c/self.fc
            self.dant = self.lambdac/2     
            self.fdk = -self.BW/2 + self.BW/(2*self.K) + (self.BW/self.K)*np.arange(self.K).T
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

