# Copyright (c) 2025 University of Salerno
# SPDX-License-Identifier: Apache-2.0

from abc import abstractmethod
import argparse
from typing import List
import networkx as nx
from rdkit import Chem
import sys
from tqdm import tqdm

organic_subset = {l: (i) for i, l in enumerate('N Cl * Br I P H O C S F'.split())}
NUM_LABELS = len(organic_subset)

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

def process_single_graph(mol):
  g = smartsToGraph(mol)
  g = g.to_directed()
  return g

def get_graph_list(lines):
  graphs = []
  # for each line in input
  for el in tqdm(lines, desc="[*] Processing graph", file=sys.stderr):
    el = el.strip()
    
    if not el:
      continue

    tmp = smartsToGraph(el)
    tmp = tmp.to_directed()
    graphs.append(tmp)

  return graphs


class Parser:
  def __init__(self, graphs: List[nx.DiGraph]):
    self.graphs = graphs
  
  @abstractmethod
  def parse(self, file):
    pass

  def join_graphs(self):
    return nx.disjoint_union_all(self.graphs)

  @classmethod
  def get_parsers(cls):
    return [subclass.__name__.replace('Parser', '') for subclass in cls.__subclasses__()]
  
  @classmethod
  def create_parser(cls, framework: str, g: nx.DiGraph):
    for subclass in cls.__subclasses__():
      if subclass.__name__ == framework + 'Parser':
        return subclass(g)
    raise ValueError(f"Parser for framework '{framework}' not found")
  
class CuTSParser(Parser):
  def parse(self, file):
    g = self.join_graphs()
    print(g.number_of_nodes(), file=file)
    for a, b in tqdm(g.edges(), desc="Printing edges", file=sys.stderr):
      print(a, b, file=file)

class GSIParser(Parser):
  def parse(self, file):
    g = self.join_graphs()
    print("t # 0", file=file)
    print(len(g.nodes), len(g.edges), NUM_LABELS, 1, file=file)
    for i in tqdm(range(len(g.nodes)), desc="Printing nodes", file=sys.stderr):
      print('v', i, getLabel(g.nodes[i]), file=file)
    
    for u, v in tqdm(g.edges, desc="Printing edges", file=sys.stderr):
      print('e', u, v, 1, file=file)
    
    print('t # -1', file=file)

class VF3Parser(Parser):
  def parse(self, file):
    g = self.join_graphs()
    print(len(g.nodes), file=file)
    for n in g.nodes:
      print(n, getLabel(g.nodes[n], True), file=file)
    
    for node in g.nodes:
      neighbors = list(g.neighbors(node))
      print(len(neighbors), file=file)
      for neighbor in neighbors:
        print(node, neighbor, file=file)

class MBSMParser(Parser):
  def parse(self, file):
    for g in self.graphs:
      print(f'n#{g.number_of_nodes()} l#{NUM_LABELS}',  end=' ', file=file)
      for i, n in enumerate(g.nodes):
        print(i, getLabel(g.nodes[i]), end=' ', file=file)
      print(f'e#{g.number_of_edges()}', end=' ', file=file)
      for a, b in g.edges:
        print(a, b, end=' ', file=file)
      print(file=file)



if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Convert SMILES to graph')
  parser.add_argument('--output', '-o', type=str, help='Output file, if None, print to stdout')
  parser.add_argument('--framework', '-f', choices=Parser.get_parsers(), type=str, help='Framework to use', default='cuts')
  args = parser.parse_args()
  
  if sys.stdin.isatty():
    print(f'Usage: python {sys.argv[0]} < smarts_file')
    sys.exit(1)
  
  lines = sys.stdin.readlines()
  graphs = get_graph_list(lines)
  
  if args.output:
    with open(args.output, 'w') as f:
      Parser.create_parser(args.framework, graphs).parse(f)
  else:
    Parser.create_parser(args.framework, graphs).parse(sys.stdout)