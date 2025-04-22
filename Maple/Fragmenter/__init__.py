import json


def run_ms2_fragmenter(
    smiles: str,
    output_fp: str,
    cpu: int = 1,
):
    from Maple.Fragmenter.Fragmenter import compute_fragments

    # wrapper function
    r = compute_fragments(smiles, max_rounds=6, cores=cpu)
    out = {
        "smiles": smiles,
        "nodes": r["nodes"],
        "edges": r["edges"],
    }
    json.dump(out, open(output_fp, "w"))
