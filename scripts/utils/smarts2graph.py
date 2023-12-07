import fileinput
import networkx as nx
import pysmiles
from rdkit import Chem
import sys

organic_subset = {l: (i+1) for i, l in enumerate('* H B C N O P S F Cl Br I b c n o s p'.split())}

def smartsToGraph(smarts: str) -> nx.DiGraph:
  """Convert a SMARTS string to a networkx graph.
  
  Args:
    smarts: A SMARTS string.
  
  Returns:
    A networkx graph.
  """
  mol_smart = Chem.MolFromSmarts(smarts)
  mol_str = Chem.MolToSmiles(mol_smart)
  g: nx.Graph = pysmiles.read_smiles(mol_str, explicit_hydrogen=True)
  g = nx.DiGraph(g)
  return g

def getLabel(g: dict, digit: bool = True):
  return organic_subset.get(g.get('element', '*'), 0) if digit else str(g.get('element', '*'))
  