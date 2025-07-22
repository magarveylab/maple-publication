import os
from typing import List, Literal, Optional

from dotenv import load_dotenv

curdir = os.path.abspath(os.path.dirname(__file__))
dotenv_path = os.path.join(curdir, ".env")
load_dotenv(dotenv_path)


def run_MS1Former_on_mzXML(peaks_fp: str, output_fp: str, gpu_id: int = 0):
    import json
    import pickle

    from Maple.Embedder.inference.MS1Pipeline import MS1Pipeline

    pipe = MS1Pipeline(gpu_id=gpu_id)
    ms1_peaks = json.load(open(peaks_fp))
    out = pipe.embed_ms1_spectra_from(ms1_peaks=ms1_peaks)
    pickle.dump(out, open(output_fp, "wb"))


def annotate_mzXML_with_tax_scores(
    peaks_fp: str,  # from peak picking module
    ms1_emb_fp: str,  # from MS1Pipeline
    output_fp: str,
    peak_ids: Optional[List[int]] = None,
    query_phylum: Optional[str] = None,
    query_class: Optional[str] = None,
    query_order: Optional[str] = None,
    query_family: Optional[str] = None,
    query_genus: Optional[str] = None,
    use_cloud_service: bool = True,
):
    import json
    import pickle

    import pandas as pd

    from Maple.Embedder.Qdrant.Search import get_related_peaks_by_ms1
    from Maple.Embedder.Qdrant.TaxScore import get_tax_score_from_search_result

    # load peaks
    peaks = json.load(open(peaks_fp, "r"))
    # load ms1 embedding
    emb_lookup = pickle.load(open(ms1_emb_fp, "rb"))["peak_embeddings"]
    # prepare input data
    input_data = []
    for peak in peaks:
        peak_id = int(peak["peak_id"])
        adduct_type = peak["adduct_type"]
        if adduct_type in ["MpH", "Mp2H", "Mp3H"]:
            if isinstance(peak_ids, list) and peak_id not in peak_ids:
                continue
            input_data.append(
                {
                    "peak_id": peak_id,
                    "embedding": emb_lookup[peak_id],
                    "mass": peak["monoisotopic_mass"],
                    "rt": peak["rt"],
                    "adduct_type": adduct_type,
                }
            )
    # get related peaksmit to 1000 for performance
    all_search_results = get_related_peaks_by_ms1(
        peak_queries=input_data,
        batch_size=1,
        use_cloud_service=use_cloud_service,
    )
    out = []
    for search_result in all_search_results:
        # get tax scores
        response = get_tax_score_from_search_result(
            search_result=search_result["related_peaks"],
            query_phylum=query_phylum,
            query_class=query_class,
            query_order=query_order,
            query_family=query_family,
            query_genus=query_genus,
        )
        r = {
            "peak_id": search_result["peak_id"],
            "phylum_score": response.get("phylum_score"),
            "class_score": response.get("class_score"),
            "order_score": response.get("order_score"),
            "family_score": response.get("family_score"),
            "genus_score": response.get("genus_score"),
        }
        out.append(r)
    pd.DataFrame(out).to_csv(output_fp, index=False)


def run_MS2Former_on_mzXML(
    peaks_fp: str,
    output_fp: Optional[str] = None,
    embedding_type: str = Literal["chemotype", "analog"],
    gpu_id: int = 0,
    min_ms2: int = 5,
):
    import json
    import pickle

    from tqdm import tqdm

    from Maple.Embedder.inference.MS2Pipeline import (
        AnalogMS2Pipeline,
        ChemotypeMS2Pipeline,
    )

    # load appropriate ingerence pipeline
    if embedding_type == "chemotype":
        pipe = ChemotypeMS2Pipeline(gpu_id=gpu_id)
    elif embedding_type == "analog":
        pipe = AnalogMS2Pipeline(gpu_id=gpu_id)
    else:
        raise ValueError(
            f"embedding_type must be one of ['chemotype', 'analog'], got {embedding_type}"
        )
    # load peaks
    peaks = json.load(open(peaks_fp, "r"))
    # reformat input data
    input_data = []
    for p in peaks:
        if len(p["ms2"]) >= min_ms2:
            input_data.append(
                {
                    "peak_id": int(p["peak_id"]),
                    "precursor_mz": p["mz"],
                    "ms2_spectra": p["ms2"],
                }
            )
    # get embeddings
    out = [pipe.embed_ms2_spectra_from(**p) for p in tqdm(input_data)]
    # save output
    if output_fp is None:
        return out
    else:
        pickle.dump(out, open(output_fp, "wb"))


def annotate_mzXML_with_chemotypes(
    peaks_fp: Optional[str] = None,
    ms2_emb_fp: Optional[str] = None,
    output_fp: Optional[str] = None,
    gpu_id: int = 0,
    min_ms2: int = 5,
):
    import pickle

    import pandas as pd

    from Maple.Embedder.Qdrant.Classification import (
        ms2_chemotype_classification,
    )

    if peaks_fp is not None:
        emb_result = run_MS2Former_on_mzXML(
            peaks_fp=peaks_fp,
            output_fp=None,
            embedding_type="chemotype",
            gpu_id=gpu_id,
            min_ms2=min_ms2,
        )
    else:
        emb_result = pickle.load(open(ms2_emb_fp, "rb"))
    query_list = [
        {"query_id": r["peak_id"], "embedding": r["embedding"]}
        for r in emb_result
    ]
    result = ms2_chemotype_classification(query_list=query_list)
    out = []
    for peak_id, prop in result.items():
        out.append(
            {
                "peak_id": peak_id,
                "label": prop["label"],
                "homology": prop["homology"],
                "distance": prop["distance"],
            }
        )
    pd.DataFrame(out).to_csv(output_fp, index=False)


def compute_ms2_networks_from_mzXMLs(
    ms2_emb_fps: List[str],
    output_fp: str,
    n_neighbors: int = 15,
    min_cluster_size: int = 5,
    use_rapidsai: bool = False,
):
    import pickle

    import numpy as np
    import pandas as pd

    from Maple.Embedder.clustering.clustering import compute_clustering

    # prepare input data
    keys = []
    matrix = []
    for ms2_emb_fp in ms2_emb_fps:
        source = ms2_emb_fp.split("/")[-1]
        emb_result = pickle.load(open(ms2_emb_fp, "rb"))
        for r in emb_result:
            keys.append({"source": source, "peak_id": r["peak_id"]})
            matrix.append(r["embedding"])
    matrix = np.array(matrix)
    # density based clustering
    out = compute_clustering(
        matrix=matrix,
        matrix_keys=keys,
        min_cluster_size=min_cluster_size,
        n_neighbors=n_neighbors,
        use_rapidsai=use_rapidsai,
    )
    # remove -1 family id
    next_family_id = max([i["family_id"] for i in out]) + 1
    for i in out:
        if i["family_id"] == -1:
            i["family_id"] = next_family_id
            next_family_id += 1
    # save output
    pd.DataFrame(out).to_csv(output_fp, index=False)
