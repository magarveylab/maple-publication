import time
from typing import List

import numpy as np


def run_umap(
    matrix: np.array,
    n_components: int = 20,
    n_neighbors: int = 15,
    n_epochs: int = 500,
    min_dist=0.1,
):
    import cuml

    print("Running Dimension Reduction ...")
    start = time.time()
    reducer = cuml.UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        n_epochs=n_epochs,
        min_dist=min_dist,
        random_state=12,
    )
    embedding = reducer.fit_transform(matrix)
    end = time.time()
    timing = round(end - start, 2)
    print(f"Took {timing} seconds")
    return embedding


def run_hdbscan(
    reduced_matrix: np.array,
    matrix_keys: List[dict],
    min_cluster_size: int = 5,
    metric: str = "euclidean",
):
    import cuml

    print("Running Soft Clustering ...")
    start = time.time()
    clusterer = cuml.cluster.hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric=metric,
        prediction_data=True,
    )
    clusterer.fit(reduced_matrix)
    labels = clusterer.labels_
    end = time.time()
    timing = round(end - start, 2)
    print(f"Took {timing} seconds")
    output = []
    for idx, l in enumerate(labels):
        row = dict(matrix_keys[idx])
        row["family_id"] = int(l)
        output.append(row)
    return output


def compute_clustering(
    matrix: np.array,
    matrix_keys: List[dict],
    n_components: int = 20,
    n_neighbors: int = 15,
    n_epochs: int = 500,
    min_dist=0.1,
    min_cluster_size: int = 5,
    metric: str = "euclidean",
):

    # run umap
    reduced_matrix = run_umap(
        matrix=matrix,
        n_components=n_components,
        n_neighbors=n_neighbors,
        n_epochs=n_epochs,
        min_dist=min_dist,
    )
    # run soft clustering with hdbscan
    return run_hdbscan(
        reduced_matrix=reduced_matrix,
        matrix_keys=matrix_keys,
        min_cluster_size=min_cluster_size,
        metric=metric,
    )
