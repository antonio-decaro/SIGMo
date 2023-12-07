import fileinput
import networkx as nx
import pysmiles
from rdkit import Chem
import sys

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
