import networkx as nx
import numpy as np

from Maple.PeakPicker.utils import (
    ppm_lower_end,
    ppm_upper_end,
    quick_point_comparison,
)


class eicWizard:

    def __init__(self, scan_trace, eic_ppm_tol=10, eic_scan_count_cutoff=5):
        self.scan_trace = scan_trace
        self.signal_dict = None
        self.eic_graph = None
        self.scans_organized = sorted(self.scan_trace.keys())
        self.eic_ppm_tol = eic_ppm_tol
        self.eic_scan_count_cutoff = eic_scan_count_cutoff
        self.length = len(self.scans_organized)

    def process_scan(self, scan_id):
        idx = self.scans_organized.index(scan_id)
        scan_peaks = self.scan_trace[scan_id]
        related_peaks = []
        for peak in scan_peaks:
            peak_id = peak[0]
            query_mz = peak[1]
            max_mz = ppm_upper_end(query_mz, self.eic_ppm_tol)
            min_mz = ppm_lower_end(query_mz, self.eic_ppm_tol)
            for n in range(1, self.eic_scan_count_cutoff + 1):
                if (idx + n) >= self.length:
                    continue
                next_scan_id = self.scans_organized[idx + n]
                next_scan_peaks = self.scan_trace[next_scan_id]
                if next_scan_peaks.size == 0:
                    continue
                match_peak_id = find_closest_match_by_intensity(
                    next_scan_peaks, max_mz, min_mz
                )
                if match_peak_id != None:
                    related_peaks.append([peak_id, match_peak_id])
                    break
                else:
                    continue
        return related_peaks

    def add_eic_graph(self, signal_dict, eic_graph):
        self.signal_dict = signal_dict
        self.eic_graph = eic_graph

    def get_scan_count(self, nodes):
        unique_scans = set(self.signal_dict[n]["scan_id"] for n in nodes)
        scan_count = len(unique_scans)
        if len(unique_scans) >= self.eic_scan_count_cutoff:
            return True, scan_count
        else:
            return False, scan_count

    def get_top_eic_signals(self, nodes, eic_i_cut=0.7, apply_threshold=True):
        # find highest intensity
        intens_peak_id = max(
            nodes, key=lambda p: self.signal_dict[p]["intensity"]
        )
        max_intensity = self.signal_dict[intens_peak_id]["intensity"]
        intensity_cutoff = max_intensity * eic_i_cut
        # filter nodes based on intensity cutoff
        filtered_nodes = [
            n
            for n in nodes
            if self.signal_dict[n]["intensity"] >= intensity_cutoff
        ]
        G = self.eic_graph.subgraph(filtered_nodes)
        # find connected graphs
        components = list(nx.connected_components(G))
        if len(components) == 1:
            if apply_threshold:
                return self.capture_by_threshold(
                    filtered_nodes, intens_peak_id
                )
            else:
                return [intens_peak_id]
        else:
            to_send = []
            for group in components:
                intens_peak_id = max(
                    group, key=lambda p: self.signal_dict[p]["intensity"]
                )
                if apply_threshold:
                    to_send.extend(
                        self.capture_by_threshold(group, intens_peak_id)
                    )
                else:
                    to_send.append(intens_peak_id)
            return to_send

    def capture_by_threshold(self, nodes, intens_peak_id, threshold=0.01):
        max_intensity = self.signal_dict[intens_peak_id]["intensity"]
        threshold = 0.01 * max_intensity
        return [
            n for n in nodes if self.signal_dict[n]["intensity"] >= threshold
        ]


########################################################################
# helper functions
########################################################################


def find_closest_match_by_intensity(a, max_mz, min_mz):
    # note a -> [peak_id, mz, intensity]
    # find matching peaks based on mz
    A = quick_point_comparison(a, max_mz, min_mz, pos=1)
    # choose peak with lowest error
    if A.size == 0:
        return None
    else:
        return A[np.argmax(A, axis=0)[-1]][0]
