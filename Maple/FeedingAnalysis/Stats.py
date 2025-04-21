import math
import random

import numpy as np
from scipy.stats import shapiro


def subset_by_bins(data, num_bins: int = 3, subset_size: int = 8):
    # Dixon's Q test is only valid for small datasets (n <= 10)
    data = np.array(data)  # Convert input to numpy array
    # Compute bin edges
    bin_edges = np.linspace(np.min(data), np.max(data), num_bins + 1)
    # Assign each data point to a bin
    bin_assignments = np.digitize(data, bins=bin_edges, right=True)
    # organize data by bin
    sorted_data = {i: [] for i in range(0, num_bins + 1)}
    for x, y in zip(data, bin_assignments):
        sorted_data[y].append(x)
    # Select subset from each bin
    subset = []
    for x, y in sorted_data.items():
        random.shuffle(y)
        select = math.ceil(subset_size * len(y) / len(data))
        subset.extend(y[:select])
    # remove random points cannot exceed 9
    to_remove = len(subset) - 9
    if to_remove > 0:
        new_subset = subset[1:-1]
        random.shuffle(new_subset)
        new_subset = new_subset[:7]
        subset = [subset[0]] + new_subset + [subset[-1]]
    return subset


def is_dist_normal(data):
    stat, p = shapiro(data)
    return True if p > 0.05 else False


def dixon_q_test(data, target, alpha=0.05):
    """
    Performs Dixon's Q test for a single outlier in small datasets (n <= 10).
    Returns Q statistic and whether the suspected value is an outlier.
    """
    # Sort the data
    if len(data) > 9:
        data = subset_by_bins(data)
    normality = is_dist_normal(data)
    if target > min(data):
        return None
    if len(data) < 3:
        return None
    data = sorted(data + [target])
    # Compute Q for the target value
    Q_min = (data[1] - data[0]) / (
        data[-1] - data[0]
    )  # Check for small outlier
    # Critical values from Dixon's table for different n and alpha = 0.05
    if alpha == 0.01:
        Q_crit_table = {
            3: 0.994,
            4: 0.926,
            5: 0.821,
            6: 0.740,
            7: 0.680,
            8: 0.634,
            9: 0.598,
            10: 0.568,
        }
    elif alpha == 0.05:
        Q_crit_table = {
            3: 0.970,
            4: 0.829,
            5: 0.710,
            6: 0.625,
            7: 0.568,
            8: 0.526,
            9: 0.493,
            10: 0.466,
        }
    Q_crit = Q_crit_table.get(len(data), None)
    return {
        "normality": normality,
        "Q_min": round(Q_min, 3),
        "Q_crit": round(Q_crit, 3),
        "n": len(data),
        "outlier": True if Q_min > Q_crit else False,
    }


def does_peak_shift(c12_skews, c13_skew, alpha=0.05):
    if c13_skew > min(c12_skews):
        return {"shift": False, "stats": None}
    else:
        stats = dixon_q_test(c12_skews, c13_skew, alpha=alpha)
        if stats == None:
            shift = "Unknown"
        elif stats["normality"] == False:
            shift = "Unknown"
        elif stats["outlier"]:
            shift = True
        else:
            shift = False
        return {"shift": shift, "stats": stats}
