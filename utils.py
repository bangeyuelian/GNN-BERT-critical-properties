import rdkit
from rdkit import Chem
from rdkit.Chem import rdmolfiles, rdmolops
import numpy as np
import openbabel as ob


def obsmitosmile(smi):
    conv = ob.OBConversion()
    conv.SetInAndOutFormats("smi", "can")
    conv.SetOptions("K", conv.OUTOPTIONS)
    mol = ob.OBMol()
    conv.ReadString(mol, smi)
    smile = conv.WriteString(mol)
    smile = smile.replace('\t\n', '')
    return smile

def smiles2adjoin(smiles,explicit_hydrogens=True,canonical_atom_order=False):

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print('error')
        mol = Chem.MolFromSmiles(obsmitosmile(smiles))
        assert mol is not None, smiles + ' is not valid '

    if explicit_hydrogens:
        mol = Chem.AddHs(mol)
    else:
        mol = Chem.RemoveHs(mol)

    if canonical_atom_order:
        new_order = rdmolfiles.CanonicalRankAtoms(mol)
        mol = rdmolops.RenumberAtoms(mol, new_order)
    num_atoms = mol.GetNumAtoms()
    atoms_list = []
    for i in range(num_atoms):
        atom = mol.GetAtomWithIdx(i)
        atoms_list.append(atom.GetSymbol())

    adjoin_matrix = np.eye(num_atoms)
    # Add edges
    num_bonds = mol.GetNumBonds()
    for i in range(num_bonds):
        bond = mol.GetBondWithIdx(i)
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        adjoin_matrix[u,v] = 1.0
        adjoin_matrix[v,u] = 1.0
    return atoms_list,adjoin_matrix


def smiles2coords(smiles, addH=True):
    """从SMILES生成3D坐标"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if addH:
            mol = Chem.AddHs(mol)

        # 生成3D坐标
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)

        # 获取坐标
        conf = mol.GetConformer()
        coords = []
        for i in range(mol.GetNumAtoms()):
            pos = conf.GetAtomPosition(i)
            coords.append([pos.x, pos.y, pos.z])

        return np.array(coords)
    except:
        # 如果生成失败，返回零坐标
        mol = Chem.MolFromSmiles(smiles)
        if addH:
            mol = Chem.AddHs(mol)
        num_atoms = mol.GetNumAtoms()
        return np.zeros((num_atoms, 3))
