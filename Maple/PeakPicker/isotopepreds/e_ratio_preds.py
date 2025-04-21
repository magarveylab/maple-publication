import json
import os
from operator import itemgetter
from os.path import abspath, join

import numpy as np
from sklearn.externals import joblib

# load models and confidence intervals
curdir = abspath(os.path.dirname(__file__))
with open(join(curdir, "data_headers.json")) as fp:
    header_names = json.load(fp)

with open(join(curdir, "features.txt")) as fp:
    feature_names = [x.strip() for x in fp.readlines()]

with open(join(curdir, "CI_intervals.json")) as fp:
    CI_intervals = json.load(fp)

with open(join(curdir, "nc_model.cp"), "rb") as fp:
    nc_model = joblib.load(fp)

with open(join(curdir, "sc_model.cp"), "rb") as fp:
    sc_model = joblib.load(fp)


# apply confidence interval
def apply_CI(alpha, value, name):
    interval = [
        x for x in CI_intervals if x["name"] == name and x["alpha"] == alpha
    ][0]
    lower = value + interval["lower"]
    if lower < 0:
        lower = 0
    upper = value + interval["upper"]
    return lower, upper


def predict(feat_list, model, model_type, ci):

    header_map = {}
    for i in range(len(header_names)):
        header_map[header_names[i]] = i

    # format numpy array
    new_data = np.empty((len(feat_list), len(header_names)), float)
    for i in range(len(feat_list)):
        feats = feat_list[i]
        for name in header_names:
            new_data[i][header_map[name]] = feats[name]

    # replace infinity to nan
    new_data[np.isinf(new_data)] = np.nan
    # perdict
    Y = model.predict(new_data)
    predictions = []
    for pred in Y.tolist():
        lower, upper = apply_CI(ci, pred, model_type)
        full_pred = {}
        full_pred["pred"] = pred
        full_pred["lower"] = lower
        full_pred["upper"] = upper
        predictions.append(full_pred)
    return predictions


def difference(m1, m2):
    try:
        return m1 - m2
    except TypeError:
        return None


def divide(m1, m2):
    try:
        return m1 / m2
    except (TypeError, ZeroDivisionError):
        return None


def median(list_of_numbers):
    return np.median(np.array(list_of_numbers))


def gsum(mlist):
    return sum([x for x in mlist if x is not None])


def generate_iso_features(peak):

    feature_list = []
    all_intensity = peak["iso_intens"][:5]
    all_mass = peak["iso_mz"][:5]
    # fill in third value
    if len(all_mass) == 2:
        all_mass.append(2 * p["iso_mz"][1] - p["iso_mz"][0])
        all_intensity.append(0)
    # fill in fourth and fifth value
    all_mass += [None] * (5 - len(all_mass))
    all_intensity += [None] * (5 - len(all_intensity))

    # First set
    feature_list += all_intensity
    feature_list.append(min([x for x in all_intensity if x is not None]))
    feature_list.append(max([x for x in all_intensity if x is not None]))
    feature_list.append(median([x for x in all_intensity if x is not None]))

    # Second set
    feature_list.append(
        sum([x for x in itemgetter(0, 2, 4)(all_intensity) if x is not None])
    )
    feature_list.append(
        sum([x for x in itemgetter(1, 3)(all_intensity) if x is not None])
    )
    feature_list.append(
        min([x for x in itemgetter(0, 2, 4)(all_intensity) if x is not None])
    )
    feature_list.append(
        min([x for x in itemgetter(1, 3)(all_intensity) if x is not None])
    )
    feature_list.append(
        max([x for x in itemgetter(0, 2, 4)(all_intensity) if x is not None])
    )
    feature_list.append(
        max([x for x in itemgetter(1, 3)(all_intensity) if x is not None])
    )

    # Third set
    sorted_intensity = sorted([x for x in all_intensity if x is not None])[
        ::-1
    ]
    feature_list.append(all_intensity.index(sorted_intensity[0]))
    feature_list.append(all_intensity.index(sorted_intensity[1]))
    feature_list.append(all_intensity.index(sorted_intensity[2]))

    # Fourth set
    feature_list.append(difference(all_intensity[0], all_intensity[1]))
    feature_list.append(difference(all_intensity[0], all_intensity[2]))
    feature_list.append(difference(all_intensity[0], all_intensity[3]))
    feature_list.append(difference(all_intensity[1], all_intensity[2]))
    feature_list.append(difference(all_intensity[1], all_intensity[3]))
    feature_list.append(difference(all_intensity[2], all_intensity[3]))

    feature_list.append(divide(all_intensity[0], all_intensity[1]))
    feature_list.append(divide(all_intensity[0], all_intensity[2]))
    feature_list.append(divide(all_intensity[0], all_intensity[3]))
    feature_list.append(divide(all_intensity[1], all_intensity[2]))
    feature_list.append(divide(all_intensity[1], all_intensity[3]))
    feature_list.append(divide(all_intensity[2], all_intensity[3]))

    # Fifth set
    feature_list.append(
        difference(
            divide(all_intensity[0], all_intensity[1]),
            divide(all_intensity[1], all_intensity[2]),
        )
    )
    feature_list.append(
        difference(
            divide(all_intensity[1], all_intensity[2]),
            divide(all_intensity[2], all_intensity[3]),
        )
    )

    feature_list.append(
        divide(
            divide(all_intensity[0], all_intensity[1]),
            divide(all_intensity[1], all_intensity[2]),
        )
    )
    feature_list.append(
        divide(
            divide(all_intensity[1], all_intensity[2]),
            divide(all_intensity[2], all_intensity[3]),
        )
    )

    # Sixth set
    feature_list.append(gsum(all_intensity[:1]))
    feature_list.append(gsum(all_intensity[:2]))
    feature_list.append(gsum(all_intensity[:3]))
    feature_list.append(gsum(all_intensity[1:3]))
    feature_list.append(gsum(all_intensity[1:2]))
    feature_list.append(gsum(all_intensity[2:3]))

    # Seventh Set
    feature_list.append(all_mass[0])
    feature_list.append(difference(all_mass[1], all_mass[0]))
    feature_list.append(difference(all_mass[2], all_mass[0]))
    feature_list.append(difference(all_mass[3], all_mass[0]))
    feature_list.append(difference(all_mass[4], all_mass[0]))
    feature_list.append(difference(all_mass[2], all_mass[1]))
    feature_list.append(difference(all_mass[3], all_mass[1]))
    feature_list.append(difference(all_mass[4], all_mass[1]))
    feature_list.append(difference(all_mass[3], all_mass[2]))
    feature_list.append(difference(all_mass[4], all_mass[2]))
    feature_list.append(difference(all_mass[4], all_mass[3]))

    feature_dict = dict()
    for i in range(len(feature_list)):
        feature_dict[feature_names[i]] = feature_list[i]
    return feature_dict


def e_ratio_preds(peaks, ci=0.99):
    # adjust distribution
    for p in peaks:
        if len(p["iso_mz"]) == 2:
            p["iso_mz"].append(2 * p["iso_mz"][1] - p["iso_mz"][0])
            p["iso_intens"].append(0)
    # generate features
    feats = [generate_iso_features(x) for x in peaks]
    # predictions
    sc_preds = predict(feats, sc_model, "sc", ci)
    nc_preds = predict(feats, nc_model, "nc", ci)
    # add predictions
    for idx, peak in enumerate(peaks):
        peak["nc"] = nc_preds[idx]
        peak["sc"] = sc_preds[idx]
    return peaks
