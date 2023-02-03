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
from utils import db2pow, npa

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


@dataclass
class ChannelmmWaveParameters:
    # Model parameters
    syn_type = SynType.Asyn
    link_type = LinkType.Uplink

    # Signal 
    beam_type = BeamType.Random
    ris_profile_type = RisProfileType.Random
    beamforming_angle_std: float = 0        # beamforming angle variation
    array_type = ArrayType.Analog
    radio_directional: float = 1        # ratio of directional beams

    UE_radiation = RadiationType.Omni
    RIS_radiation = RadiationType.Omni
    BS_radiation = RadiationType.Omni

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
    PB = npa([0, 0, 0]).T
    OB = npa([90, 0, 0]).T

    # UE: User Equipment
    PU = npa(5, 2, 0).T
    OU = npa(0, 0, 0).T
    VU = npa(0, 0, 0).T

    # RIS: Reconfigurable Intelligent Surfaces
    PR = npa([])
    OR = npa([0, 0, 0]).T
    VR = npa([0, 0, 0]).T

    # Infrastructure Parameters
    # Number of RFCs at BS and UE
    MB: int = 1 
    MU: int = 1

    # Array Dimensions
    NB_dim = npa([4, 4]).T
    NR_dim = npa([10, 10]).T
    NU_dim = npa([4, 4]).T

    # Number of elements at BS/RIS/UE
    # Number of antennas for conventional array 
    NB: int = np.prod(NB_dim, 1)    # Number of BS elements
    NR: int = np.prod(NR_dim, 1)    # Number of RIS elements
    NU: int = np.prod(NU_dim, 1)    # Number of UE elements

    # Environment Parameters
    operationBW: float = 400E6      # Operation bandwidth for Thermal noise
    K_boltzmann: float = 1.3806E-23 # Boltzmann constant
    temperature: float = 298.15     # Temperature 25 celsius
    Pn: float = K_boltzmann * temperature * operationBW * 1000      # Thermal noise linear (in mW)
    Pn_dBm: float = 10 * np.log10(Pn)       # Thermal noise in dB
    sigma_in: float = np.sqrt(Pn)           # Johnson-Nyquist noise: sigma^2 = N_0
    noise_figure: float = 10                # Noise figure 3dB
    sigma: float = np.sqrt(np.pow(10, noise_figure/10)) * sigma_in      # Output noise level


