import json
import math
from glob import glob
from typing import List, TypedDict

import pandas as pd

from Maple.Embedder import curdir
from Maple.Embedder.graphs.HeteroGraph import HeteroGraph


class MS2Ion(TypedDict):
    mz: float
    intensity: float


########################################################################
# Vocab
########################################################################


def get_node_vocab(vocab_dir: str = f"{curdir}/models/ms2/vocab/node_vocab/*"):
    vocab_filenames = glob(vocab_dir)
    vocab = {}
    for fp in vocab_filenames:
        node_type = fp.split("/")[-1].split(".")[0]
        vocab[node_type] = json.load(open(fp))
    return vocab


def get_word_vocab(vocab_dir: str = f"{curdir}/models/ms2/vocab/word_vocab/*"):
    vocab_filenames = glob(vocab_dir)
    vocab = {}
    for fp in vocab_filenames:
        word_type = fp.split("/")[-1].split(".")[0]
        vocab[word_type] = json.load(open(fp))
    return vocab


########################################################################
# Helper functions
########################################################################


def custom_round(number, bin_rounding: float = 0.005) -> float:
    x = math.floor(number / bin_rounding) * bin_rounding
    return "{:.3f}".format(round(x * 1000) / 1000)


intens_bins = {
    "bins": [-1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.1],
    "labels": ["I1", "I2", "I3", "I4", "I5", "I6", "I7", "I8", "I9", "I10"],
}


def tokenize_ms2(
    ms2_spectra, precursor_mz: float, threshold: float = 0.005
) -> pd.DataFrame:
    # take top 50 ions
    ms2_spectra = sorted(
        ms2_spectra, key=lambda x: x["intensity"], reverse=True
    )[:50]
    # create difference
    highest_intensity = max([x["intensity"] for x in ms2_spectra])
    # tokenize ms2 ions
    tokenized_spectra = []
    for ion in ms2_spectra:
        # get properties
        mz = ion["mz"]
        intensity = round(ion["intensity"] / highest_intensity, 3)
        nl = precursor_mz - mz
        nl = 0 if nl <= 0 else nl
        min_mz = mz - threshold
        max_mz = mz + threshold
        min_nl = nl - threshold
        max_nl = nl + threshold
        min_nl = 0 if min_nl <= 0 else min_nl
        # create tokens for mz
        min_mz_word = custom_round(min_mz, bin_rounding=threshold)
        mz_word = custom_round(mz, bin_rounding=threshold)
        max_mz_word = custom_round(max_mz, bin_rounding=threshold)
        # create tokens for nl
        min_nl_word = custom_round(min_nl, bin_rounding=threshold)
        nl_word = custom_round(nl, bin_rounding=threshold)
        max_nl_word = custom_round(max_nl, bin_rounding=threshold)
        tokenized_spectra.append(
            {
                "mz": mz,
                "intensity": intensity,
                "min_mz_base": min_mz_word.split(".")[0],
                "min_mz_dec": min_mz_word.split(".")[1],
                "mz_base": mz_word.split(".")[0],
                "mz_dec": mz_word.split(".")[1],
                "max_mz_base": max_mz_word.split(".")[0],
                "max_mz_dec": max_mz_word.split(".")[1],
                "min_nl_base": min_nl_word.split(".")[0],
                "min_nl_dec": min_nl_word.split(".")[1],
                "nl_base": nl_word.split(".")[0],
                "nl_dec": nl_word.split(".")[1],
                "max_nl_base": max_nl_word.split(".")[0],
                "max_nl_dec": max_nl_word.split(".")[1],
            }
        )
    # tokenize intensities
    df = pd.DataFrame(tokenized_spectra)
    df["intensity_bin"] = pd.cut(
        df["intensity"], bins=intens_bins["bins"], labels=intens_bins["labels"]
    )
    return df


########################################################################
# Graph Structure
########################################################################


class MS2Graph(HeteroGraph):

    def __init__(self, graph_id: str):
        self.graph_id = graph_id
        # define schema
        schema = {}
        schema["node_types"] = [
            "Precursor",
            "MZ",
            "NL",
        ]
        schema["edge_types"] = [
            ("Precursor", "precursor_to_nl", "NL"),
            ("NL", "nl_to_mz", "MZ"),
        ]
        schema["node_embedding_dim"] = {}
        schema["edge_embedding_dim"] = {}
        schema["sentence_structure"] = {
            "MZ": [
                "MZbase",
                "MZdec",
                "MZbase",
                "MZdec",
                "MZbase",
                "MZdec",
                "Intensity",
            ],
            "NL": [
                "NLbase",
                "NLdec",
                "NLbase",
                "NLdec",
                "NLbase",
                "NLdec",
            ],
        }
        super().__init__(schema=schema)

    @classmethod
    def build_from_ms2_spectra(
        cls, spectra_id: str, ms2_spectra: List[MS2Ion], precursor_mz: float
    ):
        G = cls(graph_id=spectra_id)
        edge_type_lookup = {e[1]: e for e in G.edge_types}
        # tokenize ms2 spectra
        df = tokenize_ms2(ms2_spectra=ms2_spectra, precursor_mz=precursor_mz)
        # add precursor node
        precursor_node_id = G.add_node(node_type="Precursor", label="[CLS]")
        # add MS2 nodes
        for ion in df.to_dict("records"):
            mz_label = [
                str(ion["min_mz_base"]),
                str(ion["min_mz_dec"]),
                str(ion["mz_base"]),
                str(ion["mz_dec"]),
                str(ion["max_mz_base"]),
                str(ion["max_mz_dec"]),
                str(ion["intensity_bin"]),
            ]
            mz_label = " ".join(mz_label)
            mz_node_id = G.add_node(node_type="MZ", label=mz_label)
            nl_label = [
                str(ion["min_nl_base"]),
                str(ion["min_nl_dec"]),
                str(ion["nl_base"]),
                str(ion["nl_dec"]),
                str(ion["max_nl_base"]),
                str(ion["max_nl_dec"]),
            ]
            nl_label = " ".join(nl_label)
            nl_node_id = G.add_node(node_type="NL", label=nl_label)
            # add edges
            G.add_edge(
                precursor_node_id,
                nl_node_id,
                edge_type_lookup["precursor_to_nl"],
            )
            G.add_edge(
                nl_node_id,
                mz_node_id,
                edge_type_lookup["nl_to_mz"],
            )
        return G
