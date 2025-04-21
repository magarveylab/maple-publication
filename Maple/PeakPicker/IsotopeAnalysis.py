import ast
import itertools as it
import os

import networkx as nx
import numpy as np
import pandas as pd

from Maple.PeakPicker.utils import euclidean_similarity, quick_point_comparison

# global variables
mass_neutron = 1.004

# load halogen rules
halogen_rules = {}
curdir = os.path.abspath(os.path.dirname(__file__))
halogen_rules_fp = os.path.join(curdir, "database/halogen_rules.csv")
for rec in pd.read_csv(halogen_rules_fp).to_dict("records"):
    iv = ast.literal_eval(rec["intensity_vector"])
    max_iv = max(iv)
    iv = [e / max_iv for e in iv]
    halogen_rules[rec["halogen_rule"]] = {
        "index": ast.literal_eval(rec["index_vector"]),
        "intensity": iv,
        "min_size": rec["min_size"],
    }


class IsotopeAnalysis:

    def __init__(
        self,
        abs_tol=0.1,
        min_charge=1,
        max_charge=3,
        min_isopeaks=2,
        max_isopeaks=5,
        hal_sim_cutoff=0.9,
        allow_bromine=False,
        noise_intensity_cutoff=1000,
    ):
        # parameters
        self.abs_tol = abs_tol
        self.min_charge = min_charge
        self.max_charge = max_charge
        self.min_isopeaks = min_isopeaks
        self.max_isopeaks = max_isopeaks
        self.allow_bromine = allow_bromine
        self.hal_sim_cutoff = hal_sim_cutoff
        self.noise_intensity_cutoff = noise_intensity_cutoff
        # calculate mass differences to observe from charge range
        self.md_to_observe = {}
        for c in range(self.min_charge, self.max_charge + 1):
            self.md_to_observe[c] = mass_neutron / c

    def build_networks(self, scan):
        # note this function is multiprocessed
        # function resulted overlapping isotopic distributions as networks
        out = []
        # note peaks are organized by increasing mz
        scan_id, scan_peaks = scan
        scan_peaks = scan_peaks[scan_peaks[:, 1].argsort()]
        # working graphs
        isotope_diG = nx.DiGraph()  # stores direction of direction
        isotope_G = nx.Graph()  # necessary to find overlapping ditributions
        # draw isotopic connections between scan peaks
        for idx, p1 in enumerate(scan_peaks):
            for charge in range(self.min_charge, self.max_charge + 1):
                p1_node = (p1[0], charge)
                isotope_diG.add_node(
                    p1_node, mz=p1[1], charge=charge, intensity=p1[2]
                )
                md = self.md_to_observe[charge]
                # target values for p2
                p2_mz = p1[1] + md
                p2_max_mz = p2_mz + self.abs_tol
                p2_min_mz = p2_mz - self.abs_tol
                # find p2
                p2 = find_closest_match_by_error(
                    scan_peaks[idx + 1 :,], p2_mz, p2_max_mz, p2_min_mz
                )
                if p2.size == 3:
                    p2_node = (p2[0], charge)
                    isotope_diG.add_node(
                        p2_node, mz=p2[1], charge=charge, intensity=p2[2]
                    )
                    isotope_diG.add_edge(p1_node, p2_node)
                    isotope_G.add_edge(p1_node, p2_node)
                    break
                else:
                    continue
        # edges will also be made between nodes sharing the same mz
        for n1, n2 in it.combinations(isotope_diG.nodes, 2):
            if n1[0] == n2[0]:
                isotope_G.add_edge(n1, n2)
        # find overlapping distributions
        overlap_components = nx.connected_components(isotope_G)
        for nodes in overlap_components:
            diG = isotope_diG.subgraph(nodes).copy()
            out.append({"scan_id": scan_id, "diG": diG})
        return out

    def dereplicate_isotopes(self, net):
        isotopes = []
        network = net["diG"]
        continue_isotope_cleanup = True
        while continue_isotope_cleanup:
            network, out = self.isotope_calling(network)
            isotopes.extend(out)
            if len(network.nodes) == 0:
                continue_isotope_cleanup = False
        return isotopes

    def isotope_calling(self, network):
        out = []  # return confident called isotopes
        # find all potential isotopic distributions
        invalid_distributions_present = True
        all_paths = []
        while invalid_distributions_present:
            invalid_distributions_present = False
            # find all potential isotopic distributions
            all_paths = get_start_to_end_paths(network)
            all_paths = [
                p[: self.max_isopeaks] for p in all_paths if len(p) > 1
            ]
            # each path represents a potential isotopic distribution
            for path in all_paths:
                most_intens_peak = max(
                    path, key=lambda p: network.nodes[p]["intensity"]
                )
                most_intens_peak_idx = path.index(most_intens_peak)
                # usually the most_intens_peak is the first peak
                first_peak = most_intens_peak
                first_peak_idx = most_intens_peak_idx
                # unless molecule has bromine which results
                # in the third peak being morst intense
                if self.allow_bromine and most_intens_peak_idx > 2:
                    top_intens = network.nodes[most_intens_peak]["intensity"]
                    prev_peak_idx = most_intens_peak_idx - 2
                    prev_peak = path[prev_peak_idx]
                    prev_intens = network.nodes[prev_peak]["intensity"]
                    has_potential_bromine = False
                    # check if intens at 2 peaks away is equal to top for 1 Br
                    if abs(prev_intens / top_intens - 1) <= 0.05:
                        has_potential_bromine = True
                    # or if it is half of top for metabolites containing 2 Br
                    elif abs(prev_intens / top_intens - 0.5) <= 0.05:
                        has_potential_bromine = True
                    # check for bromine
                    if has_potential_bromine:
                        first_peak = prev_peak
                        first_peak_idx = prev_peak_idx
                # check if estimated first peak index is truly the first
                # if not remove edge and repeat cycle
                if first_peak_idx != 0:
                    invalid_distributions_present = True
                    previous_peak = path[first_peak_idx - 1]
                    if network.has_edge(previous_peak, first_peak):
                        network.remove_edge(previous_peak, first_peak)
        # validate paths
        filtered_paths = []
        for path in all_paths:
            mz_patt = [network.nodes[n]["mz"] for n in path]
            intensity_patt = [network.nodes[n]["intensity"] for n in path]
            if has_downstream_pattern(path, intensity_patt):
                filtered_paths.append(path)
            elif has_halogen(path, intensity_patt, self.hal_sim_cutoff):
                filtered_paths.append(path)
            else:
                edges_to_remove = find_invalid_edges_from_path(
                    path, intensity_patt, self.min_isopeaks
                )
                for edge in edges_to_remove:
                    if network.has_edge(*edge):
                        network.remove_edge(*edge)
        # organize it as network
        overlap_graph = nx.Graph()
        for pid, path in enumerate(filtered_paths):
            signal_set = set([p[0] for p in path])
            mz_patt = [network.nodes[n]["mz"] for n in path]
            intensity_patt = [network.nodes[n]["intensity"] for n in path]
            charge = path[0][1]
            overlap_graph.add_node(
                pid,
                path=path,
                signal_set=signal_set,
                mz_patt=mz_patt,
                intensity_patt=intensity_patt,
                intensity=sum(intensity_patt),
                charge=charge,
            )
        # find distributions that use the same peak
        # draw an edge in the overlap graph
        for p1_id, p1 in overlap_graph.nodes(data=True):
            for p2_id, p2 in overlap_graph.nodes(data=True):
                if p1 == p2:
                    continue
                else:
                    if len(p1["signal_set"] & p2["signal_set"]) > 0:
                        overlap_graph.add_edge(p1_id, p2_id)
        # find the best isotopic distribution
        signals_to_remove = set()
        groups = nx.connected_components(overlap_graph)
        for g in groups:
            # select the best path
            if len(g) == 1:
                best_pid = list(g)[0]
            else:
                best_pid = max(
                    g, key=lambda x: overlap_graph.nodes[x]["intensity"]
                )
            # get information on isotope
            path = overlap_graph.nodes[best_pid]["path"]
            signal_id = path[0][0]
            signal_set = overlap_graph.nodes[best_pid]["signal_set"]
            charge = overlap_graph.nodes[best_pid]["charge"]
            mz_patt = overlap_graph.nodes[best_pid]["mz_patt"]
            intensity_patt = overlap_graph.nodes[best_pid]["intensity_patt"]
            max_intens = max(intensity_patt)
            intensity_patt = [p / max_intens for p in intensity_patt]
            iso = {
                "peak_id": signal_id,
                "charge": charge,
                "isotopic_dist_mz": mz_patt,
                "isotopic_dist_intensity": intensity_patt,
            }
            # add path
            if max_intens >= self.noise_intensity_cutoff:
                out.append(iso)
            network.remove_nodes_from(path)
            signals_to_remove.update(signal_set)
        # remove nodes from graph with peaks already used
        remaining_nodes = set(network.nodes)
        nodes_to_remove = set(
            n for n in remaining_nodes if n[0] in signals_to_remove
        )
        network.remove_nodes_from(nodes_to_remove)
        # remove isolated nodes
        isolates = list(nx.isolates(network))
        network.remove_nodes_from(isolates)
        return network, out


########################################################################
# helper functions
########################################################################


def find_closest_match_by_error(a, mz, max_mz, min_mz):
    # note a -> [signal_id, mz, intensity]
    # find matching peaks based on mz
    A = quick_point_comparison(a, max_mz, min_mz, pos=1)
    # choose peak with lowest error
    if A.size == 0:
        return np.array([])
    else:
        return A[np.abs(A[:, 1] - mz).argmin()]


def has_downstream_pattern(path, intensity_patt):
    for idx, p in enumerate(path[:-1]):
        if intensity_patt[idx] < intensity_patt[idx + 1]:
            return False
    else:
        return True


def has_halogen(path, intensity_patt, hal_sim_cutoff):
    for rule, p in halogen_rules.items():
        if len(path) < p["min_size"]:
            continue
        else:
            observed = [intensity_patt[idx] for idx in p["index"]]
            highest_intensity = max(observed)
            observed = [e / highest_intensity for e in observed]
            score = euclidean_similarity(observed, p["intensity"])
            if score >= hal_sim_cutoff:
                return True
    else:
        return False


def find_invalid_edges_from_path(path, intensity_patt, min_isopeaks):
    all_edges = {(p, path[idx + 1]) for idx, p in enumerate(path[:-1])}
    if len(path) < min_isopeaks:
        return all_edges
    else:
        invalid_edges = set()
        for idx, p1 in enumerate(path[:-1]):
            p1_intens = intensity_patt[idx]
            p2 = path[idx + 1]
            p2_intens = intensity_patt[idx + 1]
            if p2_intens > p1_intens:
                invalid_edges.add((p1, p2))
        # contruct new graph with leftover edges
        leftover_edges = all_edges - invalid_edges
        if len(leftover_edges) == 0:
            return all_edges
        else:
            # after removal of edges find valid isotopic distributions
            # keep the edges in valid distributions
            # remove everything else
            subgraph = nx.Graph()
            for edge in leftover_edges:
                subgraph.add_edge(*edge)
                groups = list(nx.connected_components(subgraph))
                keep_nodes = set()
                for g in groups:
                    if len(g) >= min_isopeaks:
                        keep_nodes.update(g)
            if len(keep_nodes) == 0:
                return all_edges
            else:
                for edge in leftover_edges:
                    if set(edge).issubset(keep_nodes) == False:
                        invalid_edges.add(edge)
                return invalid_edges


def get_start_nodes(G):
    return [
        node_id for node_id, inward_edges in G.in_degree() if inward_edges == 0
    ]


def get_end_nodes(G):
    return [
        node_id
        for node_id, outward_edges in G.out_degree()
        if outward_edges == 0
    ]


def get_start_to_end_paths(G):
    start_nodes = get_start_nodes(G)
    end_nodes = get_end_nodes(G)
    single_nodes = [[n] for n in set(start_nodes) & set(end_nodes)]
    paths = [
        p for n in start_nodes for p in nx.all_simple_paths(G, n, end_nodes)
    ]
    paths.extend(single_nodes)
    return paths
