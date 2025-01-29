import fileinput
import networkx as nx
from rdkit import Chem
import sys

organic_subset = {l: (i+1) for i, l in enumerate('* H B C N O P S F Cl Br I b c n o s p'.split())}

def molToGraph(mol):
  graph = nx.Graph()
  for atom in mol.GetAtoms():
    graph.add_node(atom.GetIdx(),
                   atomic_num=atom.GetAtomicNum(),
                   is_aromatic=atom.GetIsAromatic(),
                   atom_symbol=atom.GetSymbol())
  for bond in mol.GetBonds():
    graph.add_edge(bond.GetBeginAtomIdx(),
                   bond.GetEndAtomIdx(),
                   bond_type=bond.GetBondType())
  return graph

def smartsToGraph(smarts: str) -> nx.DiGraph:
  """Convert a SMARTS string to a networkx graph.
  
  Args:
    smarts: A SMARTS string.
  
  Returns:
    A networkx graph.
  """
  mol_smart = Chem.MolFromSmarts(smarts)
  if mol_smart is None:
    return None
  return molToGraph(mol_smart)

def getLabel(g: dict, digit: bool = True):
  return organic_subset.get(g.get('element', '*'), 0) if digit else str(g.get('element', '*'))

NUM_LABELS = len(organic_subset) + 1