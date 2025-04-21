import operator
import os
import random
from multiprocessing import Pool

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

from Maple.PeakPicker.Logger import logging
from Maple.PeakPicker.utils import (
    ppm_lower_end,
    ppm_upper_end,
    quick_filtering,
    quick_point_comparison,
)

# load adduct database
curdir = os.path.abspath(os.path.dirname(__file__))
adducts_library_fp = os.path.join(curdir, "database/adducts.csv")
adducts_df = pd.read_csv(adducts_library_fp)
intense_adducts = set(adducts_df[adducts_df.intense].adduct_id)
adducts_lib = {row["adduct_id"]: row for _, row in adducts_df.iterrows()}


class MS1Ion:

    def __init__(self, peak):
        self.id = peak["peak_id"]
        self.mz = peak["mz"]
        self.charge = peak["charge"]
        self.rt = peak["rt"]
        self.intensity = peak["intensity"]
        self.scan_count = peak["scan_count"]
        self.generate_adduct_possibilities()

    def generate_adduct_possibilities(self):
        # generate all adduct possibilities
        # a -> [peak_id, adduct_type, mass]
        self.adduct_dict = {}
        all_adducts = []
        self.base_adducts = []
        for aid, a in adducts_lib.items():
            if a["charge"] != self.charge:
                continue
            else:
                mass = self.get_adduct_mass(self.mz, a)
                self.adduct_dict[aid] = mass
                all_adducts.append([self.id, aid, mass])
                if a["base"]:
                    self.base_adducts.append([aid, mass])
        all_adducts = np.array(all_adducts)
        self.all_adducts = all_adducts[all_adducts[:, 2].argsort()]

    @staticmethod
    def get_adduct_mass(mz, a):
        return (mz * a["charge"] - a["offset"]) / a["multiplier"]


class AdductCluster:

    def __init__(self, ac_id):
        self.ac_id = ac_id
        self.peak_cache = {}
        self.peaks = set()

    def add_adduct(self, peak, adduct):
        self.peak_cache[peak.id] = (peak, adduct)

    def remove_peaks(self, peak_ids):
        peaks_to_remove = set(self.peak_cache.keys()) & peak_ids
        for peak_id in peaks_to_remove:
            del self.peak_cache[peak_id]

    def cast_arrays(self):
        self.ac = []
        self.rt_cache = []
        self.nodes = set()
        for peak_id, (p, adduct) in self.peak_cache.items():
            self.ac.append([peak_id, adduct, p.intensity, p.scan_count])
            self.nodes.add((peak_id, adduct))
            self.rt_cache.append(p.rt)
        self.ac = np.array(self.ac)
        self.peaks = set(self.peak_cache.keys())

    def _compute_average_rt(self):
        self.rt = np.mean(self.rt_cache)

    def _update_intensity(self):
        intensity_vector = self.ac[:, 2]
        self.intensity = np.sum(intensity_vector)
        self.max_intensity = np.max(intensity_vector)

    def _update_size(self):
        unique_adducts = np.unique(self.ac[:, 1])
        self.priority = sum(
            [adducts_lib[a]["priority"] for a in unique_adducts]
        )
        self.size = len(unique_adducts)

    def _update_top_adducts(self, awr=1):
        intensity_cutoff = self.max_intensity * awr - 1
        # acf -> adduct cluster filtered based on intensity
        acf = quick_filtering(self.ac, intensity_cutoff, pos=2)
        self.top_adducts = set(np.unique(acf[:, 1]))

    def _update_peak_confidence_status(self):
        # acf -> adduct cluster filtered based on scan count
        acf = quick_filtering(self.ac, 2, pos=3)
        if acf.size > 0:
            self.has_confident_peaks = True
        else:
            self.has_confident_peaks = False

    def _adduct_intensity_check(self, aic=3000, cc=0.6):
        # acf -> adduct cluster filtered based on intensity
        acf = quick_filtering(self.ac, aic, pos=2)
        if acf.size / self.ac.size >= cc:
            return True
        else:
            return False

    def update_properties(self, awr):
        # update properties
        self.cast_arrays()
        if len(self.peaks) > 0:
            self._compute_average_rt()
            self._update_size()
            self._update_intensity()
            self._update_top_adducts(awr=awr)
            self._update_peak_confidence_status()

    def validate(self, aic, cc, atr):
        # validation
        intensity_condition = self._adduct_intensity_check(aic=aic, cc=cc)
        if self.size > 1 and intensity_condition and self.has_confident_peaks:
            if atr == True:
                if len(self.top_adducts & intense_adducts) > 0:
                    return True
                else:
                    return False
            else:
                return True
        else:
            return False


class AdductNetwork:

    def __init__(self, peaks, rt_tol, aic, awr, cc, atr, aip, cores=10):
        self.peaks = peaks
        self.rt_tol = rt_tol
        self.cores = cores
        self.aic = aic
        self.awr = awr
        self.cc = cc
        self.atr = atr
        self.aip = aip
        # sorting function
        if self.aip:
            self.sort_function = operator.attrgetter("intensity", "priority")
        else:
            self.sort_function = operator.attrgetter("priority", "intensity")
        # node -> (peak_id, adduct_id)
        self.graph = nx.Graph()
        self.ac_overlap_graph = nx.Graph()
        self.final_adduct_clusters = []

    def call_adduct_groups(self):
        # initial setup
        logging.info("Overlapping all adduct clusters")
        self.find_base_nodes()
        self.compute_adduct_clusters()
        self.update_ac_overlap_graph()
        ac_count = len(self.ac_dict)
        logging.info("Total clusters: {}".format(ac_count))
        permutation = 1
        # constant loop until
        while ac_count > 0:
            self.find_best_adduct_clusters()
            ac_count = len(self.ac_dict)
            logging.info("Permutation: {}".format(permutation))
            logging.info("Remaining Clusters: {}".format(ac_count))
            permutation += 1
        # build adduct map
        self.build_adduct_map()
        final_count = len(self.final_adduct_clusters)
        logging.info("Found {} adduct clusters".format(final_count))

    def find_base_nodes(self):
        self.base_nodes = set()
        for n in self.graph.nodes:
            if adducts_lib[n[1]]["base"]:
                self.base_nodes.add(n)

    def compute_adduct_clusters(self):
        self.ac_dict = {}
        for ac_id, n in enumerate(self.base_nodes):
            # initiate adduct cluster
            ac = AdductCluster(ac_id)
            # find local group in network
            neighbors = list(self.graph.neighbors(n))
            nodes = neighbors + [n]
            for peak_id, adduct in nodes:
                ac.add_adduct(self.peaks[peak_id], adduct)
            # update adduct properties
            ac.update_properties(self.awr)
            # validate group
            if ac.validate(self.aic, self.cc, self.atr):
                self.ac_dict[ac_id] = ac
            else:
                for ne in neighbors:
                    self.graph.remove_edge(n, ne)
        # remove single nodes
        single_nodes = list(nx.isolates(self.graph))
        self.graph.remove_nodes_from(single_nodes)

    def update_ac_overlap_graph(self):
        self.ac_array = [[aid, a.rt] for aid, a in self.ac_dict.items()]
        self.ac_array = np.array(self.ac_array)
        # overlap adduct clusters (multiprocessing)
        pool = Pool(self.cores)
        process = pool.imap_unordered(
            self.overlap_ac, self.ac_array, chunksize=100
        )
        for result in tqdm(process, total=len(self.ac_array)):
            self.ac_overlap_graph.add_edges_from(result)
        pool.close()

    def find_best_adduct_clusters(self):
        # choose best adduct cluster from each overlap group (multiprocessing)
        components = list(nx.connected_components(self.ac_overlap_graph))
        peaks_to_delete = set()
        # multiproecessing
        pool = Pool(self.cores)
        process = pool.imap_unordered(
            self.choose_best_ac, components, chunksize=10
        )
        for ac in tqdm(process, total=len(components)):
            self.final_adduct_clusters.append(ac.nodes)
            peaks_to_delete.update(ac.peaks)
        pool.close()
        # remove nodes from graph
        nodes_to_delete = set(n for n in self.graph if n[0] in peaks_to_delete)
        self.graph.remove_nodes_from(nodes_to_delete)
        single_nodes = list(nx.isolates(self.graph))
        self.graph.remove_nodes_from(single_nodes)
        peaks_to_delete.update([n[0] for n in single_nodes])
        # determine ac to delete
        ac_to_delete = set()
        for ac_id, ac in self.ac_dict.items():
            ac.remove_peaks(peaks_to_delete)
            ac.update_properties(self.awr)
            if len(ac.peaks) <= 1:
                ac_to_delete.add(ac_id)
            else:
                if ac.validate(self.aic, self.cc, self.atr) == False:
                    ac_to_delete.add(ac_id)
        # delete ac from overlap graph
        self.ac_overlap_graph.remove_nodes_from(ac_to_delete)
        # delete ac from dictionary
        for ac_id in ac_to_delete:
            del self.ac_dict[ac_id]

    def overlap_ac(self, ac_row):
        ac_id, rt = ac_row
        edges = []
        rt_max = rt + self.rt_tol
        rt_min = rt - self.rt_tol
        acf = quick_point_comparison(self.ac_array, rt_max, rt_min, pos=1)
        if acf.size > 0:
            ac_peaks = self.ac_dict[ac_id].peaks
            for x in acf[:, 0]:
                if len(self.ac_dict[x].peaks & ac_peaks) > 0:
                    edges.append([ac_id, x])
        return edges

    def choose_best_ac(self, ac_ids):
        ac_list = [self.ac_dict[ac_id] for ac_id in ac_ids]
        return sorted(ac_list, key=self.sort_function, reverse=True)[0]

    def build_adduct_map(self):
        self.adduct_map = {}
        for idx, cluster in enumerate(self.final_adduct_clusters):
            for peak_id, aid in cluster:
                mass = self.peaks[peak_id].adduct_dict[aid]
                adduct_type = adducts_lib[aid]["adduct"]
                base_adduct = True if adducts_lib[aid]["base"] else False
                self.adduct_map[peak_id] = {
                    "mass": mass,
                    "adduct_type": adduct_type,
                    "adduct_cluster_id": idx,
                    "base_adduct": base_adduct,
                }

    def get_adduct_info(self, peak_id):
        adduct = self.adduct_map.get(peak_id)
        if adduct == None:
            mz = self.peaks[peak_id].mz
            charge = int(self.peaks[peak_id].charge)
            adduct_type = "MpH" if charge == 1 else "Mp{}H".format(charge)
            adduct_cluster_id = None
            monoisotopic_mass = mz * charge - charge * 1.007276
            base_adduct = True
        else:
            adduct_type = adduct["adduct_type"]
            adduct_cluster_id = adduct["adduct_cluster_id"]
            monoisotopic_mass = adduct["mass"]
            base_adduct = adduct["base_adduct"]
        return adduct_type, adduct_cluster_id, monoisotopic_mass, base_adduct


class AdductAnalysis:

    def __init__(
        self,
        MASTERbook,
        rt_tol=10,
        ppm_tol=10,
        atr=False,
        awr=0.8,
        aip=True,
        aic=3000,
        cc=0.6,
        cores=10,
    ):
        """
        mandatory arguments
            MASTERbook is list of dictionaries with the following keys
                peak_id,
                mz,
                charge,
                rt,
                intensity,
                scan_count
        parameters
            atr (adduct_top_restrict)
                if its True , then the most intense peak in the adduct group
                must correspond to a base_adduct
            awr (adduct_window_restrict)
                sometimes the most intense peak in the adduct group might not
                be base_adduct but the base_adduct is still relatively high -
                this is used to determine an intensity cutoff for adduct_top_restrict
                by multiplying the adduct_window_restrict  by the highest intensity in the group
            aip (adduct_intensity_priority)
                prioritize adduct_intensity first and then number of adducts
                when selecting the best group
            aic (adduct_intensity_cutoff)
                minimum intensity of peaks in adduct group - this paramter is used
                to count the number of confident peaks
            cc (confidence_cutoff)
                the number of confident peaks from aic calc / total peaks
        """
        self.MASTERbook = MASTERbook
        self.rt_tol = rt_tol
        self.ppm_tol = ppm_tol
        self.atr = atr
        self.awr = awr
        self.aip = aip
        self.aic = aic
        self.cc = cc
        self.cores = cores

    def adduct_networking(self):
        # precalculate the different adduct masses for each peak
        logging.info("Pre-calculate Adduct Masses")
        self.precalculate_adduct_masses()
        # setup adduct network
        self.network = AdductNetwork(
            self.peaks,
            self.rt_tol,
            self.aic,
            self.awr,
            self.cc,
            self.atr,
            self.aip,
            cores=self.cores,
        )
        # find adduct connections between peaks (multiprocessing)
        logging.info("Compute all possible adduct clusters")
        peak_indexes = range(0, len(self.sorted_peaks))
        random.shuffle(list(peak_indexes))
        pool = Pool(self.cores)
        process = pool.imap_unordered(
            self.find_adduct_connections, peak_indexes, chunksize=100
        )
        for connections in tqdm(process, total=len(peak_indexes)):
            self.network.graph.add_edges_from(connections)
        pool.close()
        # call adduct groups
        self.network.call_adduct_groups()
        # update MASTERbook with adducts
        for peak in self.MASTERbook:
            adduct_type, adduct_cluster_id, monoisotopic_mass, base_adduct = (
                self.network.get_adduct_info(peak["peak_id"])
            )
            peak["adduct_type"] = adduct_type
            peak["adduct_cluster_id"] = adduct_cluster_id
            peak["monoisotopic_mass"] = monoisotopic_mass
            peak["base_adduct"] = base_adduct

    def precalculate_adduct_masses(self):
        # procedure
        self.peaks = {}
        for peak in self.MASTERbook:
            peak_id = peak["peak_id"]
            rt = peak["rt"]
            self.peaks[peak_id] = MS1Ion(peak)
        self.sorted_peaks = sorted(
            list(self.peaks.values()), key=operator.attrgetter("rt")
        )

    def find_adduct_connections(self, p1_index):
        # for every MS1 ion, iterate through all potential base adducts
        # iterate through subsequent MS1 ions
        # stop iteration if outside rt tol
        # find adduct relationship and add it to network
        p1 = self.sorted_peaks[p1_index]
        connections = []
        total = len(self.sorted_peaks)
        # forward adduct scan
        connections.extend(
            self.rt_comparison(p1, self.sorted_peaks[p1_index + 1 :])
        )
        # reverse adduct scan
        connections.extend(
            self.rt_comparison(p1, self.sorted_peaks[: p1_index - total])
        )
        return connections

    def rt_comparison(self, p1, p2_list):
        connections = []
        for p2 in p2_list:
            rt_diff = abs(p2.rt - p1.rt)
            if rt_diff <= self.rt_tol:
                connections.extend(self.adduct_comparison(p1, p2))
            else:
                break
        return connections

    def adduct_comparison(self, p1, p2):
        connections = []
        for base, M1 in p1.base_adducts:
            max_M2 = ppm_upper_end(M1, self.ppm_tol)
            min_M2 = ppm_lower_end(M1, self.ppm_tol)
            # match possible adducts of p2 to p1
            aa = quick_point_comparison(p2.all_adducts, max_M2, min_M2, pos=2)
            if aa.size > 0:
                # add adduct edges
                for row in aa:
                    connections.append([(p1.id, base), (row[0], row[1])])
        return connections
