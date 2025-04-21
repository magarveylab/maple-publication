import pandas as pd
import rdkit
from rdkit import Chem
from rdkit.Chem import rdChemReactions

from Maple.Fragmenter.Logger import logging
from Maple.Fragmenter.Molecule import Molecule, validate_mol
from Maple.Fragmenter.Resonance import resonance_enum
from Maple.Fragmenter.utils import reactions_df, terms

# ignore rdkit warnings
# these warnings correspond to mapped indexes in SMARTS
rdkit.RDLogger.DisableLog("rdApp.*")


class Reaction:

    def __init__(self, rxn_row):
        self.rxn_id = rxn_row["reaction_id"]
        self.description = rxn_row["description"]
        self.rxn_type = rxn_row["reaction_type"]
        self.allow_no_nl = rxn_row["allow_no_nl"]
        self.cascade_check = rxn_row["cascade_check"]
        self.add_reaction(rxn_row["reactant"], rxn_row["product"])
        self.add_broken_bonds(rxn_row["bonken_bond_index"])

    ############################################################################
    # functions to parse reaction meta data
    ############################################################################

    def add_reaction(self, reactant, product):
        # add in extra smarts terms
        reactant = reactant.format(**terms)
        product = product.format(**terms)
        self.reactant_smarts = Chem.MolFromSmarts(reactant)
        smarts = "({})>>({})".format(reactant, product)
        self.rxn = rdChemReactions.ReactionFromSmarts(smarts)

    def add_broken_bonds(self, broken_bonds_str):
        self.broken_bonds = []
        if pd.isna(broken_bonds_str) == False:
            for b in broken_bonds_str.split(" | "):
                s, e = b.split("-")
                self.broken_bonds.append([int(s), int(e)])

    ############################################################################
    # apply reactions
    ############################################################################

    def run_reaction(
        self, reactant, ring_bonds, do_res, do_cascade_check, debug
    ):
        fragment_list = []
        # setup reactants
        if do_res:
            reactant_mixture_list = [(mol,) for mol in reactant.rs]
        else:
            reactant_mixture_list = [(reactant.unmapped_mol,)]
        # run reaction
        for reactant_mixture in reactant_mixture_list:
            ps = self.rxn.RunReactants(reactant_mixture)
            # parse products
            for pset in ps:
                mol = pset[0]
                if validate_mol(mol):
                    node = self.process_product_set(
                        mol, reactant, ring_bonds, do_cascade_check, debug
                    )
                    if node != None:
                        node["reactant_id"] = reactant.hash_id
                        fragment_list.append(node)
        return fragment_list

    def process_product_set(
        self, mol, reactant, ring_bonds, do_cascade_check, debug
    ):
        # parse mol
        charged = None
        neutral = []
        rxn_broken_bonds = set()
        product_cache = reactant.pass_atom_maps(mol)
        product_mol = product_cache["mol"]
        rxn_to_map_dict = product_cache["rxn_to_map_dict"].forward
        reaction_centre = set(rxn_to_map_dict.values())
        try:
            for mol in Chem.GetMolFrags(product_mol, asMols=True):
                # render molecule
                fragment = Molecule(mol)
                # validate mol
                if fragment.mol == None:
                    continue
                # standardize molecule
                fragment = resonance_enum.standardize_resonance(fragment)
                # update properties
                fragment.update_properties()
                fragment.add_reaction_centre(reaction_centre)
                # parse charged and neutral fragments
                if fragment.is_charged():
                    # calculate resonance
                    fragment.add_resonance_structures()
                    fragment.add_electronegativity_score()
                    # note in a reaction there is only one charged fragment
                    charged = fragment
                else:
                    neutral.append(fragment)
        except:
            print(f"Error in mol creation: {Chem.MolToSmiles(mol)}")
        # logs
        if debug:
            logging.info("Reaction: {}".format(self.description))
            logging.info("Reaction Centre: {}".format(reaction_centre))
            logging.info("Reactant: {}".format(reactant.unmapped_smiles))
            logging.info("Charged: {}".format(charged.unmapped_smiles))
            for n in neutral:
                logging.info("Neutral: {}".format(n.unmapped_smiles))
            logging.info("------")
        # calculate broken bonds - represent them by map numbers
        # these map numbers to original parent molecule indexes
        # easy for ring bond comparison
        for b in self.broken_bonds:
            bond = [rxn_to_map_dict[b[0]], rxn_to_map_dict[b[1]]]
            bond = tuple(sorted(bond))
            rxn_broken_bonds.add(bond)
        # determine if fragment should be returned
        ring_break = False
        return_fragment = False
        if charged != None:
            # ionization always return fragment
            if self.rxn_type == "ionization":
                return_fragment = True
            # check for neutral loss
            if len(neutral) > 0:
                return_fragment = True
            else:
                if self.allow_no_nl:
                    if len(ring_bonds & rxn_broken_bonds) > 0:
                        ring_break = True
                        return_fragment = True
            # cascade check
            if return_fragment and do_cascade_check:
                prior_reaction_centre = reactant.reaction_centre
                after_reaction_centre = charged.reaction_centre
                if len(prior_reaction_centre & after_reaction_centre) > 0:
                    return_fragment = True
                else:
                    return_fragment = False
        else:
            return_fragment = False
        # return fragment
        if return_fragment:
            return {
                "hash_id": charged.hash_id,
                "rxn_id": self.rxn_id,
                "rxn_type": self.rxn_type,
                "cascade_check": self.cascade_check,
                "charged": charged,
                "neutral": neutral,
                "broken_bonds": rxn_broken_bonds,
                "ring_break": ring_break,
                "reaction_routes": {
                    "run_sic": True,
                    "run_cm": True,
                    "run_cc": True,
                    "run_cr": True,
                    "run_re": True,
                },
            }
        else:
            return None


class ReactionLibrary:

    def __init__(self, name):
        self.name = name
        self.reactions = []
        df = reactions_df[reactions_df.reaction_type == self.name]
        for row in df.to_dict("records"):
            if row["disable"] == False:
                self.reactions.append(Reaction(row))

    def run_reaction_set(
        self,
        molecule,
        ring_bonds=set(),
        do_cascade_check=False,
        do_res=False,
        debug=False,
    ):
        nodes_to_add = []
        for rxn in self.reactions:
            nodes = rxn.run_reaction(
                molecule, ring_bonds, do_res, do_cascade_check, debug
            )
            nodes_to_add.extend(nodes)
        return nodes_to_add

    def estimate_future_branches(self, molecule, cascade_check=True):
        count = 0
        mol = molecule.mol
        atom_to_map_dict = molecule.atom_to_map_dict.forward
        reaction_centre = molecule.reaction_centre
        for rxn in self.reactions:
            patts = mol.GetSubstructMatches(rxn.reactant_smarts)
            if cascade_check:
                map_patts = [
                    set(atom_to_map_dict[a] for a in p) for p in patts
                ]
                count += len(
                    [p for p in map_patts if len(p & reaction_centre) > 0]
                )
            else:
                count += len(patts)
        return count


# load all the libraries
ion_lib = ReactionLibrary("ionization")
sic_lib = ReactionLibrary("simple_inductive_cleavage")
cm_lib = ReactionLibrary("charge_migration")
cc_lib = ReactionLibrary("compounded_cleavage")
cr_lib = ReactionLibrary("charge_retention")
re_lib = ReactionLibrary("rearrangement")
