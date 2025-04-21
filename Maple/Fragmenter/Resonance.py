from rdkit import Chem
from rdkit.Chem import rdChemReactions

from Maple.Fragmenter.Molecule import Molecule
from Maple.Fragmenter.utils import reactions_df, terms


class ResonanceEnumerator:

    def __init__(self):
        # load reactions
        df = reactions_df[reactions_df.reaction_type == "resonance"]
        self.reactions = []
        for row in df.to_dict("records"):
            reactant = row["reactant"].format(**terms)
            product = row["product"].format(**terms)
            smarts = "{}>>{}".format(reactant, product)
            rxn = rdChemReactions.ReactionFromSmarts(smarts)
            self.reactions.append(rxn)

    def standardize_resonance(self, reactant):
        # loop through each reaction
        # map new atom indexes to map numbers
        # adopt old map numbers to atom indexes
        for rxn in self.reactions:
            resonance = self.apply_reaction(reactant, rxn)
            if resonance != None:
                return resonance
        else:
            return reactant

    def apply_reaction(self, reactant, rxn):
        reactant_mixture = (reactant.unmapped_mol,)
        ps = rxn.RunReactants(reactant_mixture)
        if len(ps) == 0:
            return None
        else:
            res_mol = ps[0][0]
            cache = reactant.pass_atom_maps(res_mol)
            product = Molecule(cache["mol"])
            if product.mol == None:
                return None
            else:
                return product


resonance_enum = ResonanceEnumerator()
