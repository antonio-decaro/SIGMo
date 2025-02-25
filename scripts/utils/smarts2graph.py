import fileinput
import networkx as nx
from rdkit import Chem
import sys
from tqdm import tqdm

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
  if digit:
    if g['atom_symbol'] in organic_subset:
      return organic_subset[g['atom_symbol']]
    else:
      return 0
  return g['atom_symbol']

NUM_LABELS = len(organic_subset) + 1

def process_single_graph(mol):
  g = smartsToGraph(mol)
  g = g.to_directed()
  return g

def process_graph(lines):
  g = nx.DiGraph()
  # for each line in input
  for el in tqdm(lines, desc="Processing graph", file=sys.stderr):
    el = el.strip()
    
    if not el:
      continue

    tmp = smartsToGraph(el)
    tmp = tmp.to_directed()
    g = nx.disjoint_union(g, tmp)
  return g