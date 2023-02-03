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

    # BS: base station
    PB =  npa([0, 0, 0]).T
    OB = npa([90, 0, 0]).T

    # UE: user equipment
    PU = npa(5, 2, 0).T
    OU = npa(0, 0, 0).T
    VU = npa(0, 0, 0).T

    # RIS: Reconfigurable Intelligent Surfaces
    PP = npa([])
