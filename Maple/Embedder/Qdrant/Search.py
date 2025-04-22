from typing import List, TypedDict

import numpy as np
from qdrant_client.http.models import Filter
from tqdm import tqdm

from Maple.Embedder.Qdrant import Databases
from Maple.Embedder.Qdrant.QdrantBase import QdrantBase


class ms1PeakQuery(TypedDict):
    peak_id: str
    embedding: np.array
    mass: float
    rt: float
    adduct_type: str


def ppm_upper_end(M, ppm):
    return M * (2000000 + ppm) / (2000000 - ppm)


def ppm_lower_end(M, ppm):
    return M * (2000000 - ppm) / (2000000 + ppm)


def get_related_peaks_by_ms1(
    peak_queries: List[ms1PeakQuery],
    qdrant_db: QdrantBase = Databases.MS1FullCollection,
    top_n: int = 100,
    batch_size: int = 1000,
    dist_cutoff: float = 500.0,  # arbitrarily large.
    rt_tol: float = 30,  # in seconds
    ppm_tol: float = 10,
):
    peak_lookup = {}
    query_list = []
    for p in peak_queries:
        peak_id = p["peak_id"]
        query_mass = round(p["mass"], 3)
        query_rt = round(p["rt"])
        adduct_type = p["adduct_type"]
        max_mass = round(ppm_upper_end(query_mass, ppm_tol), 3)
        min_mass = round(ppm_lower_end(query_mass, ppm_tol), 3)
        max_rt = round(query_rt + rt_tol)
        min_rt = round(query_rt - rt_tol)
        query_filter = Filter(
            must=[
                {
                    "key": "mass",
                    "range": {"gte": min_mass, "lte": max_mass},
                },
                {
                    "key": "rt",
                    "range": {"gte": min_rt, "lte": max_rt},
                },
                {"key": "adduct_type", "match": {"value": adduct_type}},
            ]
        )
        peak_lookup[peak_id] = {
            "mass": query_mass,
            "rt": query_rt,
            "adduct_type": adduct_type,
        }
        query_list.append(
            {
                "query_id": peak_id,
                "embedding": p["embedding"],
                "query_filter": query_filter,
            }
        )
    # Initialize Qdrant Database
    db = qdrant_db()
    predictions = db.batch_search(
        queries=query_list,
        batch_size=batch_size,
        max_results=top_n,
        return_embeds=False,
        return_data=True,
        distance_cutoff=dist_cutoff,
        exact_mode=False,  # best to disable for prod.
    )
    # terminate connection
    del db
    # find related peaks
    result = []
    for p in tqdm(predictions):
        query, hits = p["query_id"], p["hits"]
        query_mass = peak_lookup[query]["mass"]
        query_rt = peak_lookup[query]["rt"]
        adduct_type = peak_lookup[query]["adduct_type"]
        if len(hits) == 0:
            related_peaks = []
        else:
            related_peaks = []
            for h in hits:
                subject = h["subject_id"]
                subject_intensity = h["data"]["intensity"]
                subect_strain_id = h["data"]["strain_id"]
                subject_mzml_id = h["data"]["mzml_id"]
                related_peaks.append(
                    {
                        "peak_id": subject,
                        "distance": round(h["distance"], 3),
                        "intensity": subject_intensity,
                        "strain_id": subect_strain_id,
                        "mzml_id": subject_mzml_id,
                    }
                )
        # cache results
        result.append(
            {
                "peak_id": query,
                "mass": query_mass,
                "rt": query_rt,
                "adduct_type": adduct_type,
                "related_peaks": related_peaks,
            }
        )
    return result
