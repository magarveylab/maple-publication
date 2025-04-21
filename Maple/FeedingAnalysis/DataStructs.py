from typing import List, Tuple, TypedDict


class MS2ion(TypedDict):
    mz: float
    intensity: float


class ShiftedMS2ion(TypedDict):
    c13_mz: float
    c12_mz: List[float]


class OverlapPeak(TypedDict):
    ms1_peak_id: int
    skew: float


class MS1PeakQuery(TypedDict):
    # from Cactus Database
    ms1_peak_id: int
    mz: float
    rt: float
    charge: int
    intensity_raw: float
    isotopic_distribution: List[Tuple[float, float]]
    MS2ion: List[MS2ion]


class MS1PeakTarget(TypedDict):
    mz: float
    rt: float
    charge: int
    isotopic_distribution: List[Tuple[float, float]]  # intensity, mz
    intensity_raw: float
    skew: float
    ms2: List[MS2ion]
    shifted_ms2: ShiftedMS2ion
    overlap_peaks: List[OverlapPeak]
