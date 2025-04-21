import pandas as pd

from Maple.Fragmenter.FragmentNetwork import FragmentNetwork
from Maple.Fragmenter.Logger import logging


def compute_fragments(
    smiles,
    cores=1,
    min_mass=100,
    max_rounds=6,
    consider_sic=True,
    consider_cm=True,
    consider_cc=True,
    consider_cr=True,
    consider_re=True,
    apply_stability_filter=True,
    debug=False,
):
    # create network to cache fragments
    fragment_network = FragmentNetwork(
        min_mass=min_mass,
        cores=cores,
        consider_sic=consider_sic,
        consider_cm=consider_cm,
        consider_cc=consider_cc,
        consider_cr=consider_cr,
        consider_re=consider_re,
        apply_stability_filter=apply_stability_filter,
        debug=debug,
    )
    # ionization of parent molecule
    fragment_network.load_molecule(smiles)
    fragment_network.ionization()

    # fragmentation of ionized molecules
    round_id = 1
    while round_id <= max_rounds:
        # determine if round is final
        if round_id == max_rounds:
            is_final_round = True
        else:
            is_final_round = False
        # fetch nodes to fragment
        logging.info("Round {}".format(round_id))
        nodes_to_fragment = fragment_network.fetch_nodes_to_fragment()
        # fragmentation process
        logging.info("{} fragments to break".format(len(nodes_to_fragment)))
        if len(nodes_to_fragment) > 0:
            fragment_network.fragmentation(nodes_to_fragment, is_final_round)
        round_id += 1

    # tabulate data
    output = tabulate_graphs(fragment_network.graph)
    return output


def tabulate_graphs(graph):
    node_table = []
    edge_table = []
    # organize nodes
    for n in graph.nodes:
        charged = graph.nodes[n]["charged"]
        rxn_type = graph.nodes[n]["rxn_type"]
        node_table.append(
            {
                "hash_id": charged.hash_id,
                "smiles": charged.smiles,
                "mass": charged.mass,
                "formula": charged.formula,
                "rxn_type": rxn_type,
            }
        )
    # organize edges
    for n1, n2, e_attr in graph.edges(data=True):
        neutral_loss = ".".join([m.smiles for m in e_attr["neutral"]])
        edge_table.append(
            {
                "n1": n1,
                "n2": n2,
                "rxn_id": e_attr["rxn_id"],
                "neutral_loss": neutral_loss,
            }
        )
    # cast as dataframes
    node_table = pd.DataFrame(node_table)
    edge_table = pd.DataFrame(edge_table)
    return {"nodes": node_table, "edges": edge_table}
