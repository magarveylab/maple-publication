import itertools as it

import networkx as nx
from numba import njit
from scipy.spatial import distance


def calc_ppm(M1, M2):
    return abs(M1 - M2) / ((M1 + M2) / 2) * 1000000


def ppm_upper_end(M, ppm):
    return M * (2000000 + ppm) / (2000000 - ppm)


def ppm_lower_end(M, ppm):
    return M * (2000000 - ppm) / (2000000 + ppm)


def euclidean_similarity(x, y):
    d = distance.euclidean(x, y)
    return 1 / (1 + d)


def overlap_masses(masses, threshold=0.05):
    G = nx.Graph()
    G.add_nodes_from(masses)
    for m1, m2 in it.combinations(masses, 2):
        if abs(m1 - m2) <= threshold:
            G.add_edge(m1, m2)
    return list(nx.connected_components(G))


@njit
def quick_filtering(a, min_v, pos=0):
    return a[(a[:, pos] >= min_v)]


@njit
def quick_point_comparison(a, max_v, min_v, pos=0):
    # this function is used by:
    # MSwizard.eicWizard.find_closest_match_by_intensity
    # MSwizard.isotopeWizard.find_closest_match_by_error
    return a[((a[:, pos] <= max_v) & (a[:, pos] >= min_v))]


@njit
def quick_precursor_search(a, max_mz, min_mz, max_rt, min_rt):
    # a -> (scan_id, mz, rt)
    return a[
        (
            (a[:, 1] <= max_mz)
            & (a[:, 1] >= min_mz)
            & (a[:, 2] <= max_rt)
            & (a[:, 2] >= min_rt)
        )
    ]
