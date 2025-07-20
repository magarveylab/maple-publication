from functools import partial
from typing import Callable, Dict, List, Optional, TypedDict

from Maple.Embedder.Qdrant import Databases
from Maple.Embedder.Qdrant.QdrantBase import (
    DataQuery,
    DistHitResponse,
    QdrantBase,
)


class KnnOutput(TypedDict):
    hash_id: str
    label: str
    distance: float
    homology: float
    reference_id: int  # peak id


def neighborhood_classification(
    hits: List[DistHitResponse],
    top_n: int = 5,
    dist_cutoff: Optional[float] = None,
    apply_cutoff_before_homology: bool = True,
    homology_cutoff: float = 1.0,
    apply_homology_cutoff: bool = False,
    apply_cutoff_after_homology: bool = False,
):
    # only consider top n hits
    hits = sorted(hits, key=lambda x: x["distance"])[:top_n]
    all_labels = []
    lookup = {}
    distance_lookup = {}
    if dist_cutoff == None:
        apply_cutoff_before_homology = False
    for h in hits:
        if (
            apply_cutoff_before_homology == True
            and h["distance"] > dist_cutoff
        ):
            continue
        label = h["label"]
        distance = h["distance"]
        all_labels.append(label)
        if label not in lookup:
            lookup[label] = []
            distance_lookup[label] = {}
        lookup[label].append(h["distance"])
        distance_lookup[label][distance] = h["subject_id"]
    if len(lookup) == 0:
        return None
    # return most frequent, if tied return the closest in terms of distance
    label = max(lookup, key=lambda x: (len(lookup[x]), -min(lookup[x])))
    distance = min(lookup[label])
    c = all_labels.count(label)
    homology_score = round(c / top_n, 2)
    # filters based on homology score and final distance
    if apply_homology_cutoff == True:
        if homology_score < homology_cutoff:
            return None
        if apply_cutoff_after_homology == True and distance > dist_cutoff:
            return None
    # output
    return {
        "label": label,
        "reference_id": distance_lookup[label][distance],
        "homology": round(homology_score, 3),
        "distance": round(distance, 3),
    }


def KNNClassification(
    query_list: List[DataQuery],
    qdrant_db: QdrantBase = None,
    use_cloud_service: bool = True,
    classification_method: Callable = None,
    top_n: int = 1,
    dist_cutoff: float = 500.0,  # arbitrarily large.
    apply_cutoff_before_homology: bool = True,
    homology_cutoff: float = 1.0,
    apply_homology_cutoff: bool = False,
    apply_cutoff_after_homology: bool = False,
    batch_size: int = 1000,
) -> Dict[int, KnnOutput]:
    # Initialize Qdrant Database
    db = qdrant_db(use_cloud_service=use_cloud_service)
    # run KNN
    predictions = db.batch_search(
        queries=query_list,
        batch_size=batch_size,
        max_results=top_n,
        return_embeds=False,
        return_data=True,
        distance_cutoff=dist_cutoff,
        exact_mode=False,  # best to disable for prod.
    )
    # classification
    result = {}
    for p in predictions:
        query, hits = p["query_id"], p["hits"]
        if len(hits) == 0:
            result[query] = None
        else:
            result[query] = classification_method(
                hits,
                top_n=top_n,
                dist_cutoff=dist_cutoff,
                apply_cutoff_before_homology=apply_cutoff_before_homology,
                homology_cutoff=homology_cutoff,
                apply_homology_cutoff=apply_homology_cutoff,
                apply_cutoff_after_homology=apply_cutoff_after_homology,
            )
    # terminate connection
    del db
    return result


# Define KNN functions
ms2_chemotype_classification = partial(
    KNNClassification,
    qdrant_db=Databases.MS2Reference,
    use_cloud_service=True,
    classification_method=neighborhood_classification,
    top_n=10,
    dist_cutoff=None,
    apply_cutoff_before_homology=False,
    apply_homology_cutoff=False,
    apply_cutoff_after_homology=False,
)
