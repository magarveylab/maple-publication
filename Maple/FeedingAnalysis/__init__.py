import json

from Maple.FeedingAnalysis.Stats import does_peak_shift


def run_feeding_analysis(c13_mzXML_fp: str, c12_peaks_fp: str, output_fp: str):
    from Maple.FeedingAnalysis.FeedingAnalysis import feeding_analysis

    query_peaks = json.load(open(c12_peaks_fp, "r"))
    out = feeding_analysis(
        c13_mzXML_fp=c13_mzXML_fp,
        query_peaks=query_peaks,
    )
    json.dump(out, open(output_fp, "w"))


def get_isot_dist_skewness(iso_mz: list, iso_intens: list):
    from Maple.FeedingAnalysis.Spectra import iso_dist_skewness

    # normalize intensity
    highest_intens = max(iso_intens)
    iso_intens = [i / highest_intens for i in iso_intens]
    # reformat isotopic distribution
    isotopic_distribution = list(zip(iso_intens, iso_mz))
    return iso_dist_skewness(isotopic_distribution)
