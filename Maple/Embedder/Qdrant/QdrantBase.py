import os
import time
from typing import Any, Dict, List, Literal, TypedDict, Union

import numpy as np
import pandas as pd
from dotenv import find_dotenv, get_key
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import CollectionStatus, Filter, SearchRequest
from tqdm import tqdm


def batchify(l: list, bs: int = 1000):
    return [l[x : x + bs] for x in range(0, len(l), bs)]


default_dist_metric = models.Distance.EUCLID

distance_metrics = {
    "euclidean": models.Distance.EUCLID,
    "cosine": models.Distance.COSINE,
    "manhattan": models.Distance.MANHATTAN,
    "dot_product": models.Distance.DOT,
}

dist_metric_options = Literal[
    "euclidean", "cosine", "manhattan", "dot_product"
]

default_search_params = {
    "exact": False,
    "hnsw_ef": None,
    "indexed_only": False,
    "quantization": models.QuantizationSearchParams(
        ignore=False, rescore=True, oversampling=None
    ),
}


class DataQuery(TypedDict):
    query_id: int  # identifier
    embedding: np.array


class DistHitResponse(TypedDict):
    subject_id: int
    distance: float
    label: str
    data: dict


class SimHitResponse(TypedDict):
    subject_id: int
    similarity: float
    label: str
    data: dict


class SearchResponse(TypedDict):
    query_id: int  # identifier
    hits: List[Union[DistHitResponse, SimHitResponse]]


class QdrantBase:
    def __init__(
        self,
        collection_name: str,
        label_alias: str = None,
        embedding_dim: int = None,
        memory_strategy: Literal["disk", "memory", "hybrid"] = None,
        memmap_threshold: int = None,
        delete_existing: bool = False,
        client: QdrantClient = None,
        **kwargs,
    ):
        if client is None:
            self.client = QdrantClient(
                host=get_key(find_dotenv(), "QDRANT_HOST"),
                port=get_key(find_dotenv(), "QDRANT_PORT"),
                timeout=300,
            )
        else:
            self.client = client
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        self.label_alias = label_alias
        collections = self.client.get_collections()
        collection_names = {x.name for x in collections.collections}
        if delete_existing and collection_name in collection_names:
            self.delete_database()
            collection_names = collection_names - set([collection_name])
        if collection_name not in collection_names:
            assert all(
                [
                    x is not None
                    for x in [
                        embedding_dim,
                        memory_strategy,
                        label_alias,
                    ]
                ]
            )
            print(f"Creating new collection {collection_name}...")
            self.create_collection(
                embedding_dim=embedding_dim,
                memory_strategy=memory_strategy,
                memmap_threshold=memmap_threshold,
                **kwargs,
            )
            print(f"Finished creating collection {collection_name}.")
        else:
            print(f"Loading existing collection {collection_name}.")
            self.check_status()

    def preprocess_meta(self, meta: pd.DataFrame):
        # overwrite in inheritance as needed.
        return meta

    def initial_upload(self, constants):
        meta = pd.read_csv(constants.sample_fp)
        meta = self.preprocess_meta(meta=meta)
        meta_lookup = {
            x["sample_id"]: {
                k: v for k, v in x.items() if k in constants.meta_keys
            }
            for x in meta.to_dict("records")
        }
        # bulk upload annotations to dbs.
        print(f"Uploading reference data for {self.collection_name}...")
        self.upload_from_embed_dir(
            embedding_dir=constants.embedding_dir,
            meta_lookup=meta_lookup,
        )
        print("Upload complete. Indexing...")
        self.index_collection()
        print("Indexing complete.")

    def report_size(self):
        num_vectors = self.collection_status.points_count
        print(
            f"Number of vectors in the collection '{self.collection_name}': {num_vectors}"
        )

    def report_indexed_fields(self):
        # Retrieve indexed fields from the payload schema
        indexed_fields = {
            field: details
            for field, details in self.collection_status.payload_schema.items()
            if details is not None  # Only consider fields with indices
        }

        print("Indexed Fields (Potential Partitions):")
        for field, details in indexed_fields.items():
            print(f"- {field}: {details}")

    def _check_status(self):
        self.collection_status = self.client.get_collection(
            collection_name=self.collection_name
        )

    def check_status(self, timeout: int = 30):
        self._check_status()
        start = time.time()
        while self.collection_status.status.value != "green":
            current = time.time()
            if current - start > timeout:
                break
            time.sleep(5)
            self._check_status()
        if self.collection_status.status.value != "green":
            raise TimeoutError(
                f"""
                There is an issue with the collection as loaded,
                 status checking timed out after {round(current-start, 1)}
                 seconds and the collection never returned green status. 
                 The curent status is {self.collection_status}""".replace(
                    "\n", ""
                )
                .replace("\t", "")
                .replace("    ", "")
            )

    def create_collection(
        self,
        embedding_dim: int,
        memory_strategy: Literal["disk", "memory", "hybrid"],
        memmap_threshold: int,
        distance_metric: dist_metric_options = None,
        **kwargs,
    ):
        if distance_metric is None:
            dist_metric = default_dist_metric
        else:
            assert distance_metric in distance_metrics.keys()
            dist_metric = distance_metrics[distance_metric]
        if memory_strategy == "memory":
            on_disk = False
            always_ram = True
            if memmap_threshold is not None:
                print(
                    "Memmap threshold is not recommended with an in-memory \
                    configuration. Recommend rebuilding with \
                    'memmap_threshold' equal to zero."
                )
            else:
                memmap_threshold = 0
        elif memory_strategy == "hybrid":
            on_disk = True
            always_ram = True
        elif memory_strategy == "disk":
            on_disk = True
            always_ram = False
        else:
            raise ValueError(
                f"memory_strategy expects one of 'disk', \
                    'memory', or 'hybrid'. You passed {memory_strategy}"
            )
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=embedding_dim, distance=dist_metric, on_disk=on_disk
            ),
            # set indexing disabled before bulk insert
            optimizers_config=models.OptimizersConfigDiff(
                memmap_threshold=memmap_threshold, indexing_threshold=0
            ),
            quantization_config=models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(
                    type=models.ScalarType.INT8, always_ram=always_ram
                )
            ),
            **kwargs,
        )

    def index_collection(self, indexing_threshold: int = 20000):
        self.check_status()
        # perform indexing
        self.client.update_collection(
            collection_name=self.collection_name,
            optimizer_config=models.OptimizersConfigDiff(
                indexing_threshold=indexing_threshold
            ),
        )
        collection_info = self.client.get_collection(
            collection_name=self.collection_name
        )
        start = time.time()
        while collection_info.status != CollectionStatus.GREEN:
            current = time.time()
            print(
                f"Waiting for Collection to finish indexing. \
                {round(current-start, 2)} seconds have elapsed..."
            )
            time.sleep(10)
        print("Indexing complete. Waiting 5 seconds for cleanup.")
        time.sleep(5)

    def upload_data_batch(
        self,
        ids: List[int],
        vectors: Union[List[np.array], np.ndarray],
        payloads: List[Dict[str, Any]] = None,
    ):
        self.check_status()
        # convert list to 2d array (if relevant)
        if isinstance(vectors, list):
            vectors = np.array(vectors)
        # Ensure vector is of correct dimensionality for collection
        collection_info = self.client.get_collection(
            collection_name=self.collection_name
        )
        assert vectors.shape[1] == collection_info.config.params.vectors.size
        # do bulk uploading
        self.client.upsert(
            collection_name=self.collection_name,
            points=models.Batch(
                ids=ids,
                # Qdrant python API supports only native python objects - convert to list.
                vectors=vectors.tolist(),
                payloads=payloads,
            ),
        )

    def get_db_data(
        self,
        return_embeds: bool = False,
        return_data: bool = False,
        data_filter: models.Filter = None,
        limit: int = 100,
    ):
        if data_filter is None:
            data = self.client.scroll(
                collection_name=self.collection_name,
                with_vectors=return_embeds,
                with_payload=return_data,
                limit=limit,
            )
        else:
            data = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=data_filter,
                with_vectors=return_embeds,
                with_payload=return_data,
                limit=limit,
            )
        return data

    def delete_database(self):
        print(
            f"Permanently deleting collection {self.collection_name} and all associated data..."
        )
        # get db ids
        data = self.get_db_data(
            return_data=False, return_embeds=False, limit=1000000000
        )
        ids = [x.id for x in data[0]]
        if len(ids) != 0:
            print(f"Deleting {len(ids)} vectors...")
            self.client.delete_vectors(
                collection_name=self.collection_name, points=ids, vectors=[""]
            )
            print(f"Deleting {len(ids)} payloads...")
            for sub_ls in tqdm(batchify(ids, 1000)):
                tmp = self.client.clear_payload(
                    collection_name=self.collection_name,
                    points_selector=models.PointIdsList(
                        points=sub_ls,
                    ),
                )
            print(f"Deleting {len(ids)} points...")
            for sub_ls in tqdm(batchify(ids, 1000)):
                tmp = self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=models.PointIdsList(points=sub_ls),
                )
        self.client.delete_collection(collection_name=self.collection_name)

    def create_payload_index(
        self,
        payload_key_to_idx: str,
        idx_type: Literal[
            "keyword", "integer", "float", "bool", "geo", "text"
        ],
    ):
        self.check_status()
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name=payload_key_to_idx,
            field_schema=idx_type,
        )
        self.check_status()

    def retrieve(
        self,
        ids: List[int],
        return_embeds: bool = False,
        return_data: bool = True,
    ):
        """Retrieve datapoints from Qdrant database when primary key \
            IDs are known."""
        self.check_status()
        dat = self.client.retrieve(
            collection_name=self.collection_name,
            ids=ids,
            with_vectors=return_embeds,
            with_payload=return_data,
        )
        if return_embeds:
            for x in dat:
                # convert list to array.
                x.vector = np.array(x.vector)
        return dat

    def search(
        self,
        query: DataQuery,
        max_results: int,
        return_embeds: bool = False,
        return_data: bool = True,
        query_filter: Filter = None,
        consistency=None,
        distance_cutoff: float = None,  # NOT squared distance!!!
        exact_mode: bool = False,
    ) -> List[SearchResponse]:
        self.check_status()
        if exact_mode is True:
            sps = {k: v for k, v in default_search_params.items()}
            sps["exact"] = True
            search_params = models.SearchParams(**sps)
        else:
            search_params = models.SearchParams(**default_search_params)

        result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query["embedding"].tolist(),
            limit=max_results,
            with_payload=True,
            with_vectors=return_embeds,
            consistency=consistency,
            query_filter=query_filter,
            score_threshold=distance_cutoff,
            search_params=search_params,
        )
        hits = []
        for r in result:
            dist = r.score
            data = r.payload if return_data else {}
            label = (
                r.payload[self.label_alias]
                if self.label_alias in r.payload
                else None
            )
            hits.append(
                {
                    "subject_id": r.id,
                    "distance": dist,
                    "label": label,
                    "data": data,
                }
            )
        return [{"query_id": query["query_id"], "hits": hits}]

    def batch_search(
        self,
        queries: List[DataQuery],
        batch_size: int,
        max_results: int,
        return_embeds: bool = False,
        return_data: bool = True,
        consistency=None,
        distance_cutoff: float = None,  # NOT squared distance!!!
        exact_mode: bool = False,
        ignore_self_matches: bool = True,
    ) -> List[SearchResponse]:
        self.check_status()
        if exact_mode is True:
            sps = {k: v for k, v in default_search_params}
            sps["exact"] = True
            search_params = models.SearchParams(**sps)
        else:
            search_params = models.SearchParams(**default_search_params)
        batches = batchify(queries, bs=batch_size)
        all_results = []
        for batch in tqdm(batches, leave=False):
            batch_qids = []
            batch_reshape = []
            for entry in batch:
                batch_qids.append(entry["query_id"])
                batch_reshape.append(
                    SearchRequest(
                        vector=entry["embedding"].tolist(),
                        limit=max_results,
                        with_payload=True,
                        with_vector=return_embeds,
                        filter=entry.get("query_filter"),
                        score_threshold=distance_cutoff,
                        params=search_params,
                    )
                )
            results = self.client.search_batch(
                collection_name=self.collection_name,
                requests=batch_reshape,
                consistency=consistency,
                timeout=300,
            )
            for qid, result in zip(batch_qids, results):
                hits = []
                for r in result:
                    dist = r.score
                    data = r.payload if return_data else {}
                    if r.id == qid and ignore_self_matches:
                        continue
                    label = (
                        r.payload[self.label_alias]
                        if self.label_alias in r.payload
                        else None
                    )
                    hits.append(
                        {
                            "subject_id": r.id,
                            "distance": dist,
                            "label": label,
                            "data": data,
                        }
                    )
                all_results.append({"query_id": qid, "hits": hits})
        return all_results

    def upload_from_embed_dir(
        self,
        embedding_dir: str,
        meta_lookup: Dict[int, Dict[str, Any]] = None,
    ):
        from glob import glob

        ids = []
        vectors = []
        payloads = []
        embed_fps = glob(os.path.join(embedding_dir, "*.npy"))
        for embed_fp in tqdm(embed_fps):
            sample_id = int(embed_fp.split("/")[-1].split(".")[0])
            if sample_id not in meta_lookup:
                continue
            embedding = np.load(embed_fp)
            ids.append(sample_id)
            vectors.append(embedding)
            payloads.append(meta_lookup[sample_id])

        self.upload_data_batch(
            ids=ids,
            vectors=vectors,
            payloads=payloads,
        )
