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
from Maple.Embedder.graphs.MS1Graph import (
    MS1Graph,
    MS1Peak,
    get_node_vocab,
    get_word_vocab,
)


class MS1Pipeline:

    def __init__(
        self,
        model_dir: str = f"{curdir}/models/ms1/torchscript",
        gpu_id: Optional[int] = None,
    ):
        edge_types = [
            ("Spectra", "spectra_to_rt", "RT"),
            ("RT", "rt_to_rt", "RT"),
            ("Peak", "peak_to_rt", "RT"),
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
        G: MS1Graph,
    ):
        spectra_id = G.graph_id.split("_")[0]
        ms1_traceback = G.ms1_traceback
        data = self.preprocess(G)
        r = self._forward(data)
        peak_embeddings = {
            ms1_traceback[node_id]: embedding
            for node_id, embedding in r["peak_embeddings"].items()
            if node_id in ms1_traceback
        }
        return {
            "spectra_id": spectra_id,
            "spectra_embedding": r["spectra_embedding"],
            "peak_embeddings": peak_embeddings,
        }

    def embed_ms1_spectra_from(self, ms1_peaks: List[MS1Peak]):
        # compute graph slices
        graph_slices = MS1Graph.build_from_ms1_spectra(
            mzml_id=0, ms1_peaks=ms1_peaks, store_traceback=True
        )
        # embed each graph slice
        out = [self(G) for G in graph_slices]
        # average spectra embeddings
        spectra_embedding = np.mean(
            [o["spectra_embedding"] for o in out], axis=0
        )
        # combine peak embeddings
        all_peak_embeddings = {}
        for o in out:
            for peak_id, peak_embedding in o["peak_embeddings"].items():
                if peak_id not in all_peak_embeddings:
                    all_peak_embeddings[peak_id] = []
                all_peak_embeddings[peak_id].append(peak_embedding)
        for peak_id in all_peak_embeddings:
            all_peak_embeddings[peak_id] = np.mean(
                all_peak_embeddings[peak_id], axis=0
            )
        return {
            "spectra_embedding": spectra_embedding,
            "peak_embeddings": all_peak_embeddings,
        }

    def preprocess(self, G: MS1Graph) -> Batch:
        # prepare tensor
        data = G.get_tensor_data(
            node_vocab=self.node_vocab,
            word_vocab=self.word_vocab,
            apply_edge_attr=False,
            apply_multigraph_wrapper=False,
        )
        data = Batch.from_data_list([data])
        return data

    def _forward(self, data: Batch):
        if isinstance(self.gpu_id, int):
            data.to(f"cuda:{self.gpu_id}")
        # preprocess node encoding (all types should be converted to same dimensionality)
        data["Spectra"]["x"] = self.node_encoders["Spectra"](
            data["Spectra"]["x"], None
        )
        data["RT"]["x"] = self.node_encoders["RT"](data["RT"]["x"], None)
        data["Peak"]["x"] = self.node_encoders["Peak"](data["Peak"]["x"])
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
        # get spectra embedding
        spectra_key = lookup["Spectra"]
        spectra_embedding = data[spectra_key].x.cpu().detach().numpy()[1]
        # get peak embedding
        peak_key = lookup["Peak"]
        peak_x = data[peak_key].x.cpu().detach().numpy()
        peak_node_ids = data[peak_key].node_ids.cpu().detach().numpy()
        peak_embeddings = dict(zip(list(peak_node_ids), peak_x))
        return {
            "spectra_embedding": spectra_embedding,
            "peak_embeddings": peak_embeddings,
        }
