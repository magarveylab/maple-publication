import itertools as it
from typing import List, Optional, Tuple

import networkx as nx
import numpy as np
from numba import njit
from pyopenms import *
from pyopenms import MSExperiment, MzXMLFile
from scipy.stats import skew
from tqdm import tqdm

from Maple.FeedingAnalysis.DataStructs import (
    MS1PeakQuery,
    MS1PeakTarget,
    MS2ion,
)
from Maple.PeakPicker.utils import calc_ppm, ppm_lower_end, ppm_upper_end

isotope_mass_delta = 1.00335


@njit
def quick_search(
    a: np.array, index: int, max_v: float, min_v: float
) -> np.array:
    return a[((a[:, index] <= max_v) & (a[:, index] >= min_v))]


def normalize_iso_dist(
    iso_dist: List[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    # normalize to first intensity value
    denominator = float(iso_dist[0][0])
    for iso in iso_dist:
        iso[0] = round(iso[0] / denominator, 4)
    return iso_dist


def dereplicate_ms1(
    peaks: List[MS1PeakQuery], ppm_tol: int = 10, rt_tol: int = 15
) -> List[MS1PeakQuery]:
    # lookup
    peak_dict = {p["ms1_peak_id"]: p for p in peaks}
    # sort ms1 peaks by rt and mass for fast query
    G = nx.Graph()
    G.add_nodes_from(peak_dict.keys())
    peaks = sorted(peaks, key=lambda x: (x["rt"], x["mz"]))
    for idx, p1 in tqdm(enumerate(peaks), total=len(peaks)):
        for p2 in peaks[idx + 1 :]:
            if p2["rt"] - p1["rt"] > rt_tol:
                break
            if calc_ppm(p1["mz"], p2["mz"]) > ppm_tol:
                break
            G.add_edge(p1["ms1_peak_id"], p2["ms1_peak_id"])
    groups = nx.connected_components(G)
    # chose peak with most isotopes
    filtered_ms1 = []
    for peak_ids in groups:
        best_peak_id = max(
            peak_ids,
            key=lambda x: (
                len(peak_dict[x]["isotopic_distribution"]),
                peak_dict[x]["intensity_raw"],
            ),
        )
        best_peak = peak_dict[best_peak_id]
        best_peak["overlap"] = peak_ids
        filtered_ms1.append(best_peak)
    return filtered_ms1


def dereplicate_ms2(
    ms2_ions: List[MS2ion], threshold: float = 0.05, top_n: int = 50
) -> List[MS2ion]:
    # lookup
    ms2_dict = {i["mz"]: i["intensity"] for i in ms2_ions}
    # group ms2 ions by mass
    G = nx.Graph()
    G.add_nodes_from(ms2_dict.keys())
    combs = it.combinations(ms2_dict, 2)
    G.add_edges_from(
        [[m1, m2] for m1, m2 in combs if abs(m1 - m2) <= threshold]
    )
    filtered_ms2 = []
    for group in nx.connected_components(G):
        # choose ion with highest intensity
        best_ion = max(group, key=lambda x: ms2_dict[x])
        filtered_ms2.append({"mz": best_ion, "intensity": ms2_dict[best_ion]})
    # keep top n ions
    filtered_ms2 = sorted(
        filtered_ms2, key=lambda x: x["intensity"], reverse=True
    )
    return filtered_ms2[:top_n]


def iso_dist_skewness(
    iso_dist: List[Tuple[float, float]], n: int = 1000, max_isotopes: int = 3
) -> float:
    return round(
        skew(
            [
                v
                for idx, (i, mz) in enumerate(iso_dist[:max_isotopes], 1)
                for v in [idx] * int(n * i * idx)
            ]
        ),
        4,
    )


def find_shifted_ms2(original_ms2, feeding_ms2, n_limit=3):
    shifted_ions = {ion["mz"]: set() for ion in feeding_ms2}
    for original_ion in original_ms2:
        original_mz = original_ion["mz"]
        for n_isotope in range(1, n_limit + 1):
            theoretical_shifted_mz = (
                original_mz + isotope_mass_delta * n_isotope
            )
            found_original_mz = in_ms2(theoretical_shifted_mz, original_ms2)
            if found_original_mz is None:
                found_feeding_mz = in_ms2(theoretical_shifted_mz, feeding_ms2)
                if found_feeding_mz is not None:
                    shifted_ions[found_feeding_mz].add(original_mz)
    shifted_ions = [
        {"shifted_mz": k, "original_mz": list(v)}
        for k, v in shifted_ions.items()
        if len(v) > 0
    ]
    return shifted_ions


def in_ms2(mz, ms2):
    for ion in ms2:
        if abs(ion["mz"] - mz) < 0.01:
            return ion["mz"]
    return None


class MSspectra:

    def __init__(self, mzXML_fp: str):
        # load mzXML
        self.experiment = MSExperiment()
        MzXMLFile().load(mzXML_fp, self.experiment)
        # cache parsed data
        self.scan_array = []  # [scan_id, rt]
        self.scan_to_ms1_array = {}  # scan_id -> [intensity, mz]
        self.precursor_array = []  # [scan_id, mz, rt]
        self.precursor_to_ms2 = {}  # (scan_id, mz, rt) -> [mz, intensity]
        self._parse_data()

    @property
    def num_scans(self) -> int:
        return self.experiment.size()

    def _parse_data(self):
        for scan_id, scan in tqdm(
            enumerate(self.experiment), total=self.num_scans
        ):
            # skip if noting is in the scan
            if scan.size() == 0:
                continue
            # determine MS1 vs MS2
            ms_level = scan.getMSLevel()
            # determine retention time
            rt = scan.getRT()
            # cache MS1 scans
            if ms_level == 1:
                self.scan_array.append([scan_id, rt])
                ms1_peaks = [[i, mz] for mz, i in zip(*scan.get_peaks())]
                self.scan_to_ms1_array[scan_id] = np.array(ms1_peaks)
            # cache MS2 scans
            if ms_level == 2:
                ms2_ions = [
                    {"mz": round(mz, 3), "intensity": int(i)}
                    for mz, i in zip(*scan.get_peaks())
                ]
                # len of precursor is always 1 for this analysis
                mz = scan.getPrecursors()[0].getMZ()
                self.precursor_array.append([scan_id, mz, rt])
                self.precursor_to_ms2[(scan_id, mz, rt)] = ms2_ions
        # cast numpy arrays
        self.scan_array = np.array(self.scan_array)
        self.precursor_array = np.array(self.precursor_array)

    def find_ms1(
        self,
        query_peak: MS1PeakQuery,
        ppm_tol: int = 10,
        rt_tol: int = 15,
        min_isotopes: int = 2,
    ) -> Optional[MS1PeakTarget]:
        min_rt = query_peak["rt"] - rt_tol
        max_rt = query_peak["rt"] + rt_tol
        rt_filtered = quick_search(self.scan_array, 1, max_rt, min_rt)
        # if no hits present, return nothing
        if len(rt_filtered) == 0:
            return None
        # calculate mz thresholds
        min_mz = [
            ppm_lower_end(p[1], ppm_tol)
            for p in query_peak["isotopic_distribution"]
        ]
        max_mz = [
            ppm_upper_end(p[1], ppm_tol)
            for p in query_peak["isotopic_distribution"]
        ]
        # parse scans
        hits = []
        for row in rt_filtered:
            scan_id = row[0]
            rt = row[1]
            ms1_array = self.scan_to_ms1_array[scan_id]
            # build isotopic distribution from scan
            isotopic_distribution = []
            for idx, query_isotope in enumerate(
                query_peak["isotopic_distribution"]
            ):
                mz_filtered = quick_search(
                    ms1_array, 1, max_mz[idx], min_mz[idx]
                )
                if len(mz_filtered) == 0:
                    break
                # find the peak in the scan with highest intensity
                intensity = mz_filtered[:, 0]
                best_peak = mz_filtered[
                    np.where(intensity == np.max(intensity))
                ].tolist()[0]
                isotopic_distribution.append(
                    [int(best_peak[0]), round(best_peak[1], 3)]
                )
            if len(isotopic_distribution) < min_isotopes:
                continue
            # cache isotopic_distribution
            sum_intensity = sum([i[0] for i in isotopic_distribution])
            hits.append(
                {
                    "scan_id": int(scan_id),
                    "mz": isotopic_distribution[0][1],
                    "charge": query_peak["charge"],
                    "rt": rt,
                    "isotopic_distribution": normalize_iso_dist(
                        isotopic_distribution
                    ),
                    "intensity_raw": sum_intensity,
                }
            )
        if len(hits) > 0:
            return max(hits, key=lambda x: x["intensity_raw"])
        else:
            return None

    def find_ms2(
        self,
        query_peak: MS1PeakQuery,
        ppm_tol: int = 10,
        rt_tol: int = 15,
        scan_isotopes: bool = False,
        ms2_limit: int = 50,
    ) -> List[MS2ion]:
        # append ms2 data for target peak
        min_rt = query_peak["rt"] - rt_tol
        max_rt = query_peak["rt"] + rt_tol
        rt_filtered = quick_search(self.precursor_array, 2, max_rt, min_rt)
        if len(rt_filtered) == 0:
            return []
        # calculate mz thresholds
        min_mz = [
            ppm_lower_end(p[1], ppm_tol)
            for p in query_peak["isotopic_distribution"]
        ]
        max_mz = [
            ppm_upper_end(p[1], ppm_tol)
            for p in query_peak["isotopic_distribution"]
        ]
        hits = []
        for idx, query_isotope in enumerate(
            query_peak["isotopic_distribution"]
        ):
            mz_filtered = quick_search(
                rt_filtered, 1, max_mz[idx], min_mz[idx]
            )
            if len(mz_filtered) > 0:
                hits.extend([tuple(i) for i in mz_filtered.tolist()])
            if idx >= 1 and scan_isotopes == False:
                break
        if len(hits) == 0:
            return []
        # group hits based on scan
        G = nx.Graph()
        G.add_nodes_from(hits)
        G.add_edges_from(
            [[x, y] for x, y in it.combinations(hits, 2) if x[0] == y[0]]
        )
        cache = []
        for group in nx.connected_components(G):
            ms2 = [ion for m in group for ion in self.precursor_to_ms2[m]]
            ms2 = dereplicate_ms2(ms2, top_n=ms2_limit)
            sum_intensity = sum([ion["intensity"] for ion in ms2])
            cache.append(
                {"ms2": ms2, "intensity": sum_intensity, "length": len(ms2)}
            )
        return max(cache, key=lambda x: (x["length"], x["intensity"]))["ms2"]
