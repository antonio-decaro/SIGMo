#!/usr/bin/env python3

import fileinput
import networkx as nx
import pysmiles
from rdkit import Chem
from utils import organic_subset
import sys

if __name__ == '__main__':
  if len(sys.argv) < 2:
    print('Usage: python3 smarts2vf2.py <smarts>')
    sys.exit(1)
  mol = sys.argv[1]
  
  # Read the row in SMARTS format
  mol_smart = Chem.MolFromSmarts(mol)
  mol_str = Chem.MolToSmiles(mol_smart)
  # Convert to SMILES format
  g: nx.Graph = pysmiles.read_smiles(mol_str, explicit_hydrogen=True)
  g = nx.DiGraph(g)

  nodes = []

  for node, node_value in g.nodes.data():
    label = node_value.get('element', '*')
    nodes.append(label)
  
  print(len(nodes), len(g.edges))
  print(*nodes, sep='\n')
  for a, b in g.edges:
    print(a, b)
