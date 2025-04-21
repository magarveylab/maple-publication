from typing import List

from tqdm import tqdm

from Maple.FeedingAnalysis.DataStructs import MS1PeakQuery
from Maple.FeedingAnalysis.Spectra import (
    MSspectra,
    dereplicate_ms1,
    find_shifted_ms2,
    iso_dist_skewness,
)


def feeding_analysis(
    mzXML_fp: str,
    query_peaks: List[MS1PeakQuery],
    ppm_tol: int = 10,
    rt_tol: int = 15,
    min_isotopes: int = 3,
    ms2_limit: int = 50,
):
    # load C13 spectra
    spectra = MSspectra(mzXML_fp)
    # calculate base skewness for each peak
    query_dict = {}
    for peak in query_peaks:
        query_dict[peak["ms1_peak_id"]] = peak["isotopic_distribution"]
    # filter queries
    query_peaks_filtered = dereplicate_ms1(
        peaks=query_peaks, ppm_tol=ppm_tol, rt_tol=rt_tol
    )
    # find peaks in spectra
    out = []
    for ms1_peak in tqdm(query_peaks_filtered):
        if len(ms1_peak["isotopic_distribution"]) < min_isotopes:
            continue
        target_peak = spectra.find_ms1(
            query_peak=ms1_peak,
            ppm_tol=ppm_tol,
            rt_tol=rt_tol,
            min_isotopes=min_isotopes,
        )
        if target_peak != None:
            # append ms2 data to found peak
            target_ms2 = spectra.find_ms2(
                query_peak=target_peak,
                ppm_tol=ppm_tol,
                rt_tol=rt_tol,
                scan_isotopes=True,
                ms2_limit=ms2_limit,
            )
            target_peak["ms2"] = target_ms2
            # calculate isotope skeweness with 3
            target_skew = iso_dist_skewness(
                target_peak["isotopic_distribution"][:3]
            )
            target_peak["skew"] = target_skew
            target_peak["shifted_ms2"] = find_shifted_ms2(
                original_ms2=ms1_peak["ms2"],
                feeding_ms2=target_ms2,
            )
            overlap_peaks = []
            for query_peak_id in ms1_peak["overlap"]:
                # limit isotopic skewness to 3
                query_skew = iso_dist_skewness(query_dict[query_peak_id][:3])
                overlap_peaks.append(
                    {"ms1_peak_id": query_peak_id, "skew": query_skew}
                )
                target_peak["overlap_peaks"] = overlap_peaks
            out.append(target_peak)
    return out
