import json
import math
from collections import deque
from glob import glob
from itertools import islice
from typing import List, TypedDict

import numpy as np
import pandas as pd

from Maple.Embedder import curdir
from Maple.Embedder.graphs.HeteroGraph import HeteroGraph


class MS1Peak(TypedDict):
    peak_id: int
    mz: float
    charge: int
    rt: float  # in serconds
    intensity: float


########################################################################
# Vocab
########################################################################


def get_node_vocab(vocab_dir: str = f"{curdir}/models/ms1/vocab/node_vocab/*"):
    vocab_filenames = glob(vocab_dir)
    vocab = {}
    for fp in vocab_filenames:
        node_type = fp.split("/")[-1].split(".")[0]
        vocab[node_type] = json.load(open(fp))
    return vocab


def get_word_vocab(vocab_dir: str = f"{curdir}/models/ms1/vocab/word_vocab/*"):
    vocab_filenames = glob(vocab_dir)
    vocab = {}
    for fp in vocab_filenames:
        word_type = fp.split("/")[-1].split(".")[0]
        vocab[word_type] = json.load(open(fp))
    return vocab


########################################################################
# Helper functions
########################################################################

rt_bins = [i * 30 for i in range(0, 41)] + [100000]
rt_bin_labels = [f"R{i}" for i in range(len(rt_bins) - 1)]


def custom_round(number, bin_rounding: float = 0.005) -> float:
    x = math.floor(number / bin_rounding) * bin_rounding
    return "{:.3f}".format(round(x * 1000) / 1000)


def get_rt_bins():
    bins = [i * 30 for i in range(0, 41)] + [100000]
    labels = [f"R{i}" for i in range(len(rt_bins) - 1)]
    return {"bins": bins, "labels": labels}


def get_intensity_bins(values: list):
    values = sorted(values)
    # create bins based on percentiles
    intensity_bins = [
        values[0] - 1,
        np.percentile(values, 10),
        np.percentile(values, 20),
        np.percentile(values, 30),
        np.percentile(values, 40),
        np.percentile(values, 50),
        np.percentile(values, 60),
        np.percentile(values, 70),
        np.percentile(values, 80),
        np.percentile(values, 90),
        values[-1] + 1,
    ]
    # adjust intensity bins
    adjustments = list(reversed(range(11)))
    for idx, a in enumerate(adjustments):
        intensity_bins[idx] = intensity_bins[idx] - a
    # labels
    labels = [f"I{i}" for i in range(10)]
    return {"bins": intensity_bins, "labels": labels}


def tokenize_ms1(
    ms1_peaks: List[MS1Peak], mz_tol: float = 0.005, rt_tol: int = 30
):
    # tokenize masses
    for p in ms1_peaks:
        # add min rt and max rt
        p["min_rt"] = p["rt"] - rt_tol
        p["max_rt"] = p["rt"] + rt_tol
        min_mz = p["mz"] - mz_tol
        max_mz = p["mz"] + mz_tol
        mz_word = custom_round(p["mz"])
        min_mz_word = custom_round(min_mz)
        max_mz_word = custom_round(max_mz)
        p["min_mz_base"] = min_mz_word.split(".")[0]
        p["min_mz_dec"] = min_mz_word.split(".")[1]
        p["max_mz_base"] = max_mz_word.split(".")[0]
        p["max_mz_dec"] = max_mz_word.split(".")[1]
        p["mz_base"] = mz_word.split(".")[0]
        p["mz_dec"] = mz_word.split(".")[1]
    # bin RT and intensity
    df = pd.DataFrame(ms1_peaks)
    intensity_bins = get_intensity_bins(df["intensity"])
    rt_bins = get_rt_bins()
    df["intensity_bin"] = pd.cut(
        df["intensity"],
        bins=intensity_bins["bins"],
        labels=intensity_bins["labels"],
    )
    df["rt_bin"] = pd.cut(
        df["rt"], bins=rt_bins["bins"], labels=rt_bins["labels"]
    )
    df["min_rt_bin"] = pd.cut(
        df["min_rt"], bins=rt_bins["bins"], labels=rt_bins["labels"]
    )
    df["max_rt_bin"] = pd.cut(
        df["max_rt"], bins=rt_bins["bins"], labels=rt_bins["labels"]
    )
    return df.to_dict(orient="records")


def sliding_window(iterable, size=2, step=1, fillvalue=None):
    if size < 0 or step < 1:
        raise ValueError
    it = iter(iterable)
    q = deque(islice(it, size), maxlen=size)
    if not q:
        return  # empty iterable or size == 0
    q.extend(fillvalue for _ in range(size - len(q)))  # pad to size
    while True:
        yield iter(q)  # iter() to avoid accidental outside modifications
        try:
            q.append(next(it))
        except StopIteration:  # Python 3.5 pep 479 support
            return
        q.extend(next(it, fillvalue) for _ in range(step - 1))


def slice_mzml(ms1_peaks: list, size=1000, step=500) -> List[str]:
    ms1_peaks = sorted(
        ms1_peaks,
        key=lambda x: (x["rt"], x["charge"], x["mz"], x["intensity"]),
    )
    return [
        [x for x in i if x != None]
        for i in sliding_window(ms1_peaks, size=size, step=step)
    ]


########################################################################
# Graph Structure
########################################################################


class MS1Graph(HeteroGraph):

    def __init__(self, graph_id: str):
        self.graph_id = graph_id
        # define schema
        schema = {}
        schema["node_types"] = [
            "Spectra",
            "Peak",
            "RT",
        ]
        schema["edge_types"] = [
            ("Spectra", "spectra_to_rt", "RT"),
            ("RT", "rt_to_rt", "RT"),
            ("Peak", "peak_to_rt", "RT"),
        ]
        schema["node_embedding_dim"] = {}
        schema["edge_embedding_dim"] = {}
        schema["sentence_structure"] = {
            "Peak": [
                "MZbase",
                "MZdec",
                "MZbase",
                "MZdec",
                "MZbase",
                "MZdec",
                "Charge",
                "Intensity",
            ],
        }
        super().__init__(schema=schema)

    @classmethod
    def build_from_ms1_spectra(
        cls,
        mzml_id: str,
        ms1_peaks: List[MS1Peak],
        store_traceback: bool = False,
    ):
        # tokenize ms1 spectra
        ms1_peaks = tokenize_ms1(ms1_peaks)
        # slice ms1 spectra into windows of 200 peaks
        ms1_slices = slice_mzml(ms1_peaks)
        # create graph for each ms1 slices
        graphs = []
        for sid, mzml_window in enumerate(ms1_slices):
            graph_id = f"{mzml_id}_{sid}"
            G = cls(graph_id=graph_id)
            edge_type_lookup = {e[1]: e for e in G.edge_types}
            ms1_traceback = {}
            # add special nodes
            # add spectra node
            spectra_node_id = G.add_node(node_type="Spectra", label="[CLS]")
            # add RT nodes
            rt_traceback = {}
            for rt_bin in rt_bin_labels:
                rt_node_id = G.add_node(node_type="RT", label=rt_bin)
                rt_traceback[rt_bin] = rt_node_id
                G.add_edge(
                    spectra_node_id,
                    rt_node_id,
                    edge_type_lookup["spectra_to_rt"],
                )
            # add RT edges (get chromatogram)
            for rt1, rt2 in zip(rt_bin_labels, rt_bin_labels[1:]):
                G.add_edge(
                    rt_traceback[rt1],
                    rt_traceback[rt2],
                    edge_type_lookup["rt_to_rt"],
                )
            # add peaks
            # sentence structure: MZbase, MZdec, MZbase, MZdec, Charge, Intensity
            for peak in mzml_window:
                label = [
                    str(peak["min_mz_base"]),
                    str(peak["min_mz_dec"]),
                    str(peak["mz_base"]),
                    str(peak["mz_dec"]),
                    str(peak["max_mz_base"]),
                    str(peak["max_mz_dec"]),
                    str(peak["charge"]),
                    str(peak["intensity_bin"]),
                ]
                label = " ".join(label)
                peak_node_id = G.add_node(node_type="Peak", label=label)
                ms1_traceback[peak_node_id] = int(peak["peak_id"])
                # add edges to RT
                if peak["min_rt_bin"] in rt_traceback:
                    G.add_edge(
                        peak_node_id,
                        rt_traceback[peak["min_rt_bin"]],
                        edge_type_lookup["peak_to_rt"],
                    )

                if peak["rt_bin"] in rt_traceback:
                    G.add_edge(
                        peak_node_id,
                        rt_traceback[peak["rt_bin"]],
                        edge_type_lookup["peak_to_rt"],
                    )
                if peak["max_rt_bin"] in rt_traceback:
                    G.add_edge(
                        peak_node_id,
                        rt_traceback[peak["max_rt_bin"]],
                        edge_type_lookup["peak_to_rt"],
                    )
            if store_traceback:
                G.ms1_traceback = ms1_traceback
            graphs.append(G)
        return graphs
