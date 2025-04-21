from glob import glob
from typing import List, Optional

import numpy as np
import torch
from torch_geometric.data import Batch

from Maple.Embedder import curdir
from Maple.Embedder.graphs.HeteroGraph import (
    batch_to_homogeneous,
    get_lookup_from_hetero,
)
from Maple.Embedder.graphs.MS2Graph import (
    MS2Graph,
    MS2Ion,
    get_node_vocab,
    get_word_vocab,
)


class MS2Pipeline:

    def __init__(
        self,
        model_dir: str = "",
        gpu_id: Optional[int] = None,
    ):
        edge_types = [
            ("Precursor", "precursor_to_nl", "NL"),
            ("NL", "nl_to_mz", "MZ"),
        ]
        self.edge_type_lookup = {e[1]: e for e in edge_types}
        # load vocab
        self.node_vocab = get_node_vocab()
        self.word_vocab = get_word_vocab()
        # load node encoders
        self.node_encoders = {}
        for model_fp in glob(f"{model_dir}/node_encoders/*"):
            node_type = model_fp.split("/")[-1].split(".")[0]
            self.node_encoders[node_type] = torch.jit.load(model_fp)
        # load edge type encoder
        self.edge_type_encoder = torch.jit.load(
            f"{model_dir}/edge_type_encoder.pt"
        )
        # load message passing nn
        self.gnn = torch.jit.load(f"{model_dir}/gnn.pt")
        # load transformer
        self.transformer = torch.jit.load(f"{model_dir}/transformer.pt")
        # move models to gpu (if devide defined)
        self.gpu_id = gpu_id
        if isinstance(self.gpu_id, int):
            for node_type in self.node_encoders:
                self.node_encoders[node_type].to(f"cuda:{self.gpu_id}")
            self.edge_type_encoder.to(f"cuda:{self.gpu_id}")
            self.gnn.to(f"cuda:{self.gpu_id}")
            self.transformer.to(f"cuda:{self.gpu_id}")

    def __call__(
        self,
        G: MS2Graph,
    ):
        spectra_id = G.graph_id
        data = self.preprocess(G=G)
        embedding = self._forward(data=data)
        return {"spectra_id": spectra_id, "embedding": embedding}

    def embed_ms2_spectra_from(
        self, spectra_id: str, ms2_spectra: List[MS2Ion], precursor_mz: float
    ):
        G = MS2Graph.build_from_ms2_spectra(
            spectra_id=spectra_id,
            ms2_spectra=ms2_spectra,
            precursor_mz=precursor_mz,
        )
        return self(G=G)

    def preprocess(self, G: MS2Graph) -> Batch:
        # prepare tensor
        data = G.get_tensor_data(
            node_vocab=self.node_vocab,
            word_vocab=self.word_vocab,
            apply_edge_attr=False,
            apply_multigraph_wrapper=False,
        )
        data = Batch.from_data_list([data])
        return data

    def _forward(self, data: Batch) -> np.array:
        if isinstance(self.gpu_id, int):
            data.to(f"cuda:{self.gpu_id}")
        # preprocess node encoding (all types should be converted to same dimensionality)
        data["Precursor"]["x"] = self.node_encoders["Precursor"](
            data["Precursor"]["x"], None
        )
        data["NL"]["x"] = self.node_encoders["NL"](data["NL"]["x"])
        data["MZ"]["x"] = self.node_encoders["MZ"](data["MZ"]["x"])
        # convert heterogenous to homogenous
        lookup = get_lookup_from_hetero(data)
        data = batch_to_homogeneous(data)
        # edge encode by edge type
        data.edge_attr = self.edge_type_encoder(data.edge_type, None)
        # message passing
        data.x = self.gnn(data.x, data.edge_index, data.edge_attr)
        # transformer (global attention accross nodes)
        data.x = self.transformer(data.x, data.batch)
        # convert homogenous to heterogenous
        data = data.to_heterogeneous()
        precursor_key = lookup["Precursor"]
        return data[precursor_key].x.cpu().detach().numpy()[1]


class ChemotypeMS2Pipeline(MS2Pipeline):

    def __init__(self, gpu_id: Optional[int] = None):
        super().__init__(
            model_dir=f"{curdir}/models/ms2/p1-d5-chemotype", gpu_id=gpu_id
        )


class AnalogMS2Pipeline(MS2Pipeline):

    def __init__(self, gpu_id: Optional[int] = None):
        super().__init__(
            model_dir=f"{curdir}/models/ms2/p1-d5-tanimoto-mlm", gpu_id=gpu_id
        )
