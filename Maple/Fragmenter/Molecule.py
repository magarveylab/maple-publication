import numpy as np
import pandas as pd
import xxhash
from rdkit import Chem
from rdkit.Chem.rdchem import ResonanceMolSupplier
from rdkit.Chem.rdMolDescriptors import CalcExactMolWt, CalcMolFormula

from Maple.Fragmenter.utils import InverseDict, en_chart

# smart pattern for charged molecule
charged_patt = Chem.MolFromSmarts("[*+1]")
flags = Chem.KEKULE_ALL | Chem.ALLOW_INCOMPLETE_OCTETS


class Molecule:

    def __init__(self, mol):
        self.mol = mol
        self.smiles = Chem.MolToSmiles(self.mol)
        # render atom to map dict
        self.atom_to_map_dict = InverseDict()
        for atom in self.mol.GetAtoms():
            self.atom_to_map_dict.add(atom.GetIdx(), atom.GetAtomMapNum())
        # clean mol
        self.unmapped_mol = Chem.Mol(self.mol)
        for atom in self.unmapped_mol.GetAtoms():
            atom.SetAtomMapNum(0)
        # get hash id
        self.unmapped_smiles = Chem.MolToSmiles(self.unmapped_mol)
        self.hash_id = xxhash.xxh32(self.unmapped_smiles).intdigest()

    def initialize_atom_maps(self):
        # update atom maps by indexes in smiles
        self.atom_to_map_dict = InverseDict()
        for atom in self.mol.GetAtoms():
            atom_idx = atom.GetIdx()
            map_num = atom.GetIdx() + 1
            atom.SetAtomMapNum(map_num)
            self.atom_to_map_dict.add(atom_idx, map_num)

    def get_ring_bonds(self):
        # update bond dict
        ring_bonds = set()
        for bond in self.mol.GetBonds():
            if bond.IsInRing():
                start = bond.GetBeginAtom().GetAtomMapNum()
                end = bond.GetEndAtom().GetAtomMapNum()
                bond_id = tuple(sorted([start, end]))
                ring_bonds.add(bond_id)
        return ring_bonds

    def pass_atom_maps(self, product_mol):
        # this function is used to process mols from reactions
        # atom map numbers correspond to reactant indexes
        product_rxn_to_map_dict = InverseDict()
        for atom in product_mol.GetAtoms():
            # determine corresponding map num of reactant
            react_atom_idx = int(atom.GetProp("react_atom_idx"))
            map_num = self.atom_to_map_dict.forward[react_atom_idx]
            # determine atoms invovled in reactions
            if atom.HasProp("old_mapno"):
                rxn_idx = atom.GetIntProp("old_mapno")
                product_rxn_to_map_dict.add(rxn_idx, map_num)
            # update cache
            atom.SetAtomMapNum(map_num)
        # fix atom properties
        product_mol.UpdatePropertyCache()
        return {"mol": product_mol, "rxn_to_map_dict": product_rxn_to_map_dict}

    def update_properties(self):
        # find atom map numbers to delete - ones not found in the structure
        all_map_nums = set(self.atom_to_map_dict.reverse.keys())
        self.reactant_atoms = {a.GetAtomMapNum() for a in self.mol.GetAtoms()}
        map_nums_to_delete = all_map_nums - self.reactant_atoms
        # clean dictionaries
        for map_num in map_nums_to_delete:
            self.atom_to_map_dict.delete("reverse", map_num)
        # other mol properties
        self.mass = CalcExactMolWt(self.mol)
        self.formula = CalcMolFormula(self.mol)

    def add_reaction_centre(self, reaction_centre):
        map_nums = set(self.atom_to_map_dict.reverse.keys())
        self.reaction_centre = reaction_centre & map_nums

    def is_charged(self):
        return self.mol.HasSubstructMatch(charged_patt)

    def add_resonance_structures(self):
        self.rs = list(ResonanceMolSupplier(self.unmapped_mol, flags=flags))
        self.rs_count = len(self.rs)

    def add_electronegativity_score(self):
        scores = []
        for resMol in self.rs:
            match = self.mol.GetSubstructMatch(charged_patt)
            atomic_num = resMol.GetAtomWithIdx(match[0]).GetAtomicNum()
            if atomic_num in en_chart:
                scores.append(en_chart[atomic_num])
        self.en_score = np.mean(scores)


def validate_mol(mol):
    # use rdkit definitions of valid mol
    mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
    if mol == None:
        return False
    else:
        return True
