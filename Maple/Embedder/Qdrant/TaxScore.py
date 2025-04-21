from typing import List, Optional, TypedDict

import pandas as pd

from Maple.Embedder import curdir


class TaxKNNOutput(TypedDict):
    peak_id: int
    strain_id: int
    distance: float


def load_strain_to_tax_lookups():
    tax_levels = ["phylum", "class", "order", "family", "genus"]
    strain_to_tax = {t: {} for t in tax_levels}
    for r in pd.read_csv(f"{curdir}/dat/tax_rels.csv").to_dict("records"):
        strain_id = r["strain_id"]
        for tax in strain_to_tax:
            tax_id = r[f"{tax}_id"]
            if pd.isna(tax_id) == False:
                strain_to_tax[tax][strain_id] = tax_id
    return strain_to_tax


def load_tax_lookup(tax_level: str):
    df = pd.read_csv(f"{curdir}/dat/{tax_level}.csv")
    tax_lookup = dict(zip(df.name, df.taxonomy_id))
    return tax_lookup


strain_to_tax = load_strain_to_tax_lookups()
phylum_lookup = load_tax_lookup("phylum")
class_lookup = load_tax_lookup("class")
order_lookup = load_tax_lookup("order")
family_lookup = load_tax_lookup("family")
genus_lookup = load_tax_lookup("genus")


def get_tax_score_from_knn_result(
    knn_result: List[TaxKNNOutput],
    query_phylum: Optional[str] = None,
    query_class: Optional[str] = None,
    query_order: Optional[str] = None,
    query_family: Optional[str] = None,
    query_genus: Optional[str] = None,
    top_n: int = 100,
):
    out = {}
    query_tax = {
        "phylum": phylum_lookup.get(query_phylum),
        "class": class_lookup.get(query_class),
        "order": order_lookup.get(query_order),
        "family": family_lookup.get(query_family),
        "genus": genus_lookup.get(query_genus),
    }
    # sort knn (by distance and base it of top n similar peaks)
    knn_result = sorted(knn_result, key=lambda x: x["distance"])[:top_n]
    # convert to unique strain ids
    unique_strain_ids = set(r["strain_id"] for r in knn_result)
    # trace to orgnaisms at different levels and output homology scores
    tax_levels = ["phylum", "class", "order", "family", "genus"]
    for t in tax_levels:
        if query_tax[t] is None:
            continue
        signature = [
            strain_to_tax[t][s]
            for s in unique_strain_ids
            if s in strain_to_tax[t]
        ]
        if len(signature) == 0:
            continue
        score = round(signature.count(query_tax[t]) / len(signature), 2)
        out[f"{t}_score"] = score
    return out
