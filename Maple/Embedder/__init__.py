import json
import os
import pickle
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

curdir = os.path.abspath(os.path.dirname(__file__))
dotenv_path = os.path.join(curdir, ".env")
load_dotenv(dotenv_path)

# inference functions


def run_MS1Former_on_mzXML(peaks_fp: str, output_fp: str):
    from Maple.Embedder.inference.MS1Pipeline import MS1Pipeline

    pipe = MS1Pipeline(gpu_id=0)
    ms1_peaks = json.load(open(peaks_fp))
    out = pipe.embed_ms1_spectra_from(ms1_peaks=ms1_peaks)
    pickle.dump(out, open(output_fp, "wb"))


def annotate_mzXML_with_tax_scores(
    peaks_fp: str,  # from peak picking module
    ms1_emb_fp: str,  # from MS1Pipeline
    output_fp: str,
    query_phylum: Optional[str] = None,
    query_class: Optional[str] = None,
    query_order: Optional[str] = None,
    query_family: Optional[str] = None,
    query_genus: Optional[str] = None,
):
    from Maple.Embedder.Qdrant.Search import get_related_peaks_by_ms1
    from Maple.Embedder.Qdrant.TaxScore import get_tax_score_from_search_result

    # load peaks
    peaks = json.load(open(peaks_fp, "r"))
    # load ms1 embedding
    emb_lookup = pickle.load(open(ms1_emb_fp, "rb"))["peak_embeddings"]
    # prepare input data
    input_data = []
    for peak in peaks:
        peak_id = peak["peak_id"]
        adduct_type = peak["adduct_type"]
        if adduct_type in ["MpH", "Mp2H", "Mp3H"]:
            input_data.append(
                {
                    "peak_id": int(peak["peak_id"]),
                    "embedding": emb_lookup[peak_id],
                    "mass": peak["monoisotopic_mass"],
                    "rt": peak["rt"],
                    "adduct_type": adduct_type,
                }
            )
    # get related peaks
    all_search_results = get_related_peaks_by_ms1(peak_queries=input_data)
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
