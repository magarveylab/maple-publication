from multiprocessing import Pool

import networkx as nx
from rdkit import Chem
from tqdm import tqdm

from Maple.Fragmenter.Logger import logging
from Maple.Fragmenter.Molecule import Molecule
from Maple.Fragmenter.Reaction import (
    cc_lib,
    cm_lib,
    cr_lib,
    ion_lib,
    re_lib,
    sic_lib,
)


class FragmentNodeFunc:

    def __init__(self, min_mass, apply_stability_filter):
        self.min_mass = min_mass
        self.apply_stability_filter = apply_stability_filter
        self.formula_cache = set()

    def overlap_nodes(self, nodes):
        overlap = {}
        for n in nodes:
            mass = n["charged"].mass
            formula = n["charged"].formula
            # mass filtering
            if mass >= self.min_mass:
                # for charge migration always add the node
                if n["rxn_type"] == "charge_migration":
                    add_formula = True
                # for rearrangement always add the node
                elif n["rxn_type"] == "rearrangement":
                    add_formula = True
                # for ring break always add the node
                elif n["ring_break"]:
                    add_formula = True
                # formula filtering unless ring break
                elif formula not in self.formula_cache:
                    add_formula = True
                else:
                    add_formula = False
                # add formula
                if add_formula:
                    if formula not in overlap:
                        overlap[formula] = []
                    overlap[formula].append(n)
        return list(overlap.values())

    def update_reaction_routes(self, nodes, rxn_type):
        # note nodes here are overlapping
        if rxn_type == "ionization":
            return self.branch_propagation_1(nodes)
        elif rxn_type == "simple_inductive_cleavage":
            return self.branch_propagation_1(nodes)
        elif rxn_type == "compounded_cleavage":
            return self.branch_propagation_1(nodes)
        elif rxn_type == "charge_migration":
            return self.branch_propagation_2(nodes)
        elif rxn_type == "charge_retention":
            return self.branch_propagation_3(nodes)
        elif rxn_type == "rearrangement":
            return self.branch_propagation_4(nodes)

    def branch_propagation_1(self, nodes):
        # filter nodes by stability
        if self.apply_stability_filter:
            nodes = self.filter_by_stability(nodes)
        # only run cr for node with most future branches
        nodes = self.sort_nodes_by_future_branches(nodes, cascade_check=False)
        for n in nodes[1:]:
            n["reaction_routes"]["run_cr"] = False
            n["reaction_routes"]["run_re"] = False
        return nodes

    def branch_propagation_2(self, nodes):
        # filter nodes by stability
        if self.apply_stability_filter:
            nodes = self.filter_by_stability(nodes)
        # disable all reactions except for simple_inductive_cleavage
        # and compounded cleavage
        for n in nodes:
            n["reaction_routes"]["run_cm"] = False
            n["reaction_routes"]["run_cr"] = False
            n["reaction_routes"]["run_re"] = False
        return nodes

    def branch_propagation_3(self, nodes):
        # next reaction must act on the reaction centre
        # keep the node with most future branches
        nodes = self.sort_nodes_by_future_branches(nodes)
        return [nodes[0]]

    def branch_propagation_4(self, nodes):
        # next reaction must act on the reaction centre
        # disable all reactions except for charge retention
        nodes = self.sort_nodes_by_future_branches(nodes)
        filtered_nodes = []
        for n in nodes:
            if n["future_branches"] > 0:
                n["reaction_routes"]["run_sic"] = False
                n["reaction_routes"]["run_cm"] = False
                n["reaction_routes"]["run_cc"] = False
                n["reaction_routes"]["run_re"] = False
                filtered_nodes.append(n)
        return filtered_nodes

    def sort_nodes_by_future_branches(self, nodes, cascade_check=True):
        # this function is used to priortize nodes with the same molecular
        # formula based on future branches (based on charge retention)
        for n in nodes:
            n["future_branches"] = cr_lib.estimate_future_branches(
                n["charged"], cascade_check=cascade_check
            )
        # sort nodes by future nodes and hash_id for consistency
        nodes = sorted(
            nodes,
            key=lambda k: (k["future_branches"], k["hash_id"]),
            reverse=True,
        )
        return nodes

    def filter_by_stability(self, nodes):
        # filter by resonance
        max_rs_count = max([n["charged"].rs_count for n in nodes])
        nodes = [n for n in nodes if n["charged"].rs_count == max_rs_count]
        # filter by electrongetivity
        min_en_score = min([n["charged"].en_score for n in nodes])
        nodes = [n for n in nodes if n["charged"].en_score == min_en_score]
        return nodes


class FragmentNetwork:

    def __init__(
        self,
        cores=1,
        consider_sic=True,
        consider_cm=True,
        consider_cc=True,
        consider_cr=True,
        consider_re=True,
        apply_stability_filter=True,
        min_mass=100,
        debug=False,
    ):
        self.cores = cores
        self.consider_sic = consider_sic
        self.consider_cm = consider_cm
        self.consider_cc = consider_cc
        self.consider_cr = consider_cr
        self.consider_re = consider_re
        if cores == 1:
            if debug:
                self.debug = True
            else:
                self.debug = False
        else:
            self.debug = False
        self.graph = nx.DiGraph()
        self.NodeFunc = FragmentNodeFunc(min_mass, apply_stability_filter)
        self.ring_bonds = set()
        self.is_final_round = False

    ##################################################################
    # Node manipulation
    ##################################################################

    def add_node(self, result):
        # get properties
        hash_id = result["hash_id"]
        reactant_id = result["reactant_id"]
        rxn_id = result["rxn_id"]
        rxn_type = result["rxn_type"]
        charged = result["charged"]
        neutral = result["neutral"]
        has_neutral = True if len(result["neutral"]) > 0 else False
        reaction_routes = result["reaction_routes"]
        cascade_check = result["cascade_check"]
        ring_break = result["ring_break"]
        # add node
        if hash_id not in self.graph.nodes:
            # cache node
            self.graph.add_node(
                hash_id,
                charged=charged,
                rxn_type=rxn_type,
                reaction_routes=reaction_routes,
                cascade_check=cascade_check,
                ring_break=ring_break,
                stop=False,
            )
            # cache molecular formula
            self.NodeFunc.formula_cache.add(charged.formula)
        # add connections
        if rxn_type != "ionization":
            if self.graph.has_edge(reactant_id, hash_id) == False:
                self.graph.add_edge(
                    reactant_id,
                    hash_id,
                    rxn_id=rxn_id,
                    neutral=neutral,
                    has_neutral=has_neutral,
                )

    def remove_nodes(self, nodes):
        downstream_nodes = set()
        for n in nodes:
            downstream_nodes.update(self.graph.neighbors(n))
        self.graph.remove_nodes_from(nodes)
        self.graph.remove_nodes_from(downstream_nodes)

    def fetch_nodes_to_fragment(self):
        nodes_to_fragment = []
        for n in self.graph.nodes:
            if self.graph.nodes[n]["stop"] == False:
                nodes_to_fragment.append(n)
        return nodes_to_fragment

    ##################################################################
    # Reaction Process
    ##################################################################

    def load_molecule(self, parent_smiles):
        parent_mol = Chem.MolFromSmiles(parent_smiles)
        self.parent_molecule = Molecule(parent_mol)
        self.parent_molecule.initialize_atom_maps()
        self.ring_bonds = self.parent_molecule.get_ring_bonds()

    def ionization(self):
        # run ionization reactions
        nodes = ion_lib.run_reaction_set(self.parent_molecule, do_res=False)
        # filter nodes
        nodes_to_add = self._filter_nodes(nodes, "ionization")
        # add nodes to graph
        for n in nodes_to_add:
            self.add_node(n)

    def _filter_nodes(self, nodes, rxn_type):
        filtered_nodes = []
        # for final round, all nodes must have neutral loss
        if self.is_final_round:
            nodes = [n for n in nodes if len(n["neutral"]) > 0]
        # overlap nodes
        overlap = self.NodeFunc.overlap_nodes(nodes)
        # filter nodes and update reaction routes
        # note nodes are only filtered for CRF produced
        for group in overlap:
            if len(group) == 1:
                filtered_nodes.append(group[0])
            else:
                group = self.NodeFunc.update_reaction_routes(group, rxn_type)
                filtered_nodes.extend(group)
        return filtered_nodes

    def _apply_fragmentation_reaction(self, n):
        nodes_to_add = []
        reactant = self.graph.nodes[n]["charged"]
        do_cascade_check = self.graph.nodes[n]["cascade_check"]
        reaction_route = self.graph.nodes[n]["reaction_routes"]
        # apply simple inductive cleavage
        if self.consider_sic and reaction_route["run_sic"]:
            nodes = sic_lib.run_reaction_set(
                reactant,
                do_res=True,
                ring_bonds=self.ring_bonds,
                do_cascade_check=do_cascade_check,
                debug=self.debug,
            )
            nodes_to_add.extend(nodes)
        # apply charge migration reactions
        # no change in mass - so ignored in last round of fragmentation
        if self.is_final_round == False:
            if self.consider_cm and reaction_route["run_cm"]:
                nodes = cm_lib.run_reaction_set(
                    reactant,
                    ring_bonds=self.ring_bonds,
                    do_res=True,
                    do_cascade_check=do_cascade_check,
                    debug=self.debug,
                )
                nodes_to_add.extend(nodes)
        # apply componded cleavage
        if self.consider_cc and reaction_route["run_cc"]:
            nodes = cc_lib.run_reaction_set(
                reactant,
                ring_bonds=self.ring_bonds,
                do_res=True,
                do_cascade_check=do_cascade_check,
                debug=self.debug,
            )
            nodes_to_add.extend(nodes)
        # apply charge retention
        if self.consider_cr and reaction_route["run_cr"]:
            nodes = cr_lib.run_reaction_set(
                reactant,
                do_res=False,
                ring_bonds=self.ring_bonds,
                do_cascade_check=do_cascade_check,
                debug=self.debug,
            )
            nodes_to_add.extend(nodes)
        # apply rearrangement
        if self.is_final_round == False:
            if self.consider_re and reaction_route["run_re"]:
                nodes = re_lib.run_reaction_set(
                    reactant,
                    do_res=False,
                    ring_bonds=self.ring_bonds,
                    do_cascade_check=do_cascade_check,
                    debug=self.debug,
                )
                nodes_to_add.extend(nodes)
        return nodes_to_add

    def fragmentation(self, nodes_to_fragment, is_final_round):
        self.is_final_round = is_final_round
        resulting_nodes = []
        # single processor (useful for debugging)
        if self.cores == 1:
            for n in tqdm(nodes_to_fragment):
                nodes = self._apply_fragmentation_reaction(n)
                resulting_nodes.extend(nodes)
        # multiprocessing fragmentation (for production)
        else:
            pool = Pool(self.cores)
            process = pool.imap_unordered(
                self._apply_fragmentation_reaction, nodes_to_fragment
            )
            resulting_nodes = []
            for r in tqdm(process, total=len(nodes_to_fragment)):
                resulting_nodes.extend(r)
            pool.close()
        # sort nodes by node types
        sort_nodes = {
            "simple_inductive_cleavage": [],
            "charge_migration": [],
            "compounded_cleavage": [],
            "charge_retention": [],
            "rearrangement": [],
        }
        for n in resulting_nodes:
            sort_nodes[n["rxn_type"]].append(n)
        # filter nodes
        nodes_to_add = []
        for rxn_type, nodes in sort_nodes.items():
            if len(nodes) > 0:
                nodes_to_add.extend(self._filter_nodes(nodes, rxn_type))
        # add nodes to graph
        for n in nodes_to_add:
            self.add_node(n)
        # remove nodes based on edge restrictions
        logging.info(
            "New potential nodes to add: {}".format(len(nodes_to_add))
        )
        nodes_to_delete = set()
        for n in nodes_to_fragment:
            # skip edge restriction for ionization nodes
            if self.graph.nodes[n]["rxn_type"] == "ionization":
                continue
            # neutral loss check
            if self.nl_check(n) == False:
                nodes_to_delete.add(n)
            # ring check
            elif self.ring_check(n) == False:
                nodes_to_delete.add(n)
            # downstream check
            elif self.downstream_check(n) == False:
                nodes_to_delete.add(n)
            # node is approved and complete
            else:
                self.graph.nodes[n]["stop"] = True
        logging.info("New nodes to delete: {}".format(len(nodes_to_delete)))
        self.remove_nodes(nodes_to_delete)

    ##################################################################
    # Node checks
    ##################################################################

    def nl_check(self, node):
        # check if consecutive reactions produce a neutral fragment
        in_nl = any(
            [
                self.graph[e1][e2]["has_neutral"]
                for e1, e2 in self.graph.in_edges(node)
            ]
        )
        if in_nl == True:
            return True
        else:
            out_nl = any(
                [
                    self.graph[e1][e2]["has_neutral"]
                    for e1, e2 in self.graph.out_edges(node)
                ]
            )
            if out_nl == True:
                return True
            else:
                return False

    def ring_check(self, node):
        # check if ring break is followed by another ring break
        ring_break = self.graph.nodes[node]["ring_break"]
        rxn_type = self.graph.nodes[node]["rxn_type"]
        if rxn_type == "charge_migration":
            predecessor_check = any(
                [
                    self.graph.nodes[n]["ring_break"]
                    for n in self.graph.predecessors(node)
                ]
            )
            successor_check = any(
                [
                    self.graph.nodes[n]["ring_break"]
                    for n in self.graph.successors(node)
                ]
            )
            # the previous simple inductive cleavage causes ring break
            if predecessor_check:
                # the subsequent simple inductive cleavage causes ring break
                if successor_check:
                    return True
                else:
                    return False
        elif ring_break:
            successor_check = any(
                [
                    self.graph.nodes[n]["ring_break"]
                    for n in self.graph.successors(node)
                ]
            )
            if successor_check:
                return True
            else:
                return False
        else:
            # there is no ring breaks (so autmatically passes filter)
            return True

    def downstream_check(self, node):
        rxn_type = self.graph.nodes[node]["rxn_type"]
        if rxn_type in ["charge_migration", "rearrangement"]:
            # must be followed by a reaction
            if len(self.graph.neighbors(node)) > 0:
                return True
            else:
                return False
        else:
            return True
