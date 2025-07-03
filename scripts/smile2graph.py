# Copyright (c) 2025 University of Salerno
# SPDX-License-Identifier: Apache-2.0

from abc import abstractmethod
import argparse
import os
from typing import List
import networkx as nx
from rdkit import Chem
import sys
from tqdm import tqdm

organic_subset = {l: (i) for i, l in enumerate('N Cl * Br I P H O C S F'.split())}
bond_types = {l: (i) for i, l in enumerate([Chem.rdchem.BondType.UNSPECIFIED, Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC])}
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

def getNodeLabel(g: dict, digit: bool = True):
  if digit:
    if g['atom_symbol'] in organic_subset:
      return organic_subset[g['atom_symbol']]
    else:
      return 0
  return g['atom_symbol']

def getEdgeLabel(e: dict):
  if e['bond_type'] not in bond_types:
    bond_types[e['bond_type']] = len(bond_types)
  return bond_types[e['bond_type']]

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
  def __init__(self, graphs: List[nx.DiGraph], no_wildcards: bool = False, group_diameter: bool = False):
    if no_wildcards:
      self.graphs = [g for g in graphs if self.check_wildcards(g)]
    else:
      self.graphs = graphs
    self.group_diameter = group_diameter
    
  
  @abstractmethod
  def parse(self, file):
    pass

  def join_graphs(self):
    return nx.disjoint_union_all(self.graphs)

  def check_wildcards(self, g):
    for n in g.nodes:
      if g.nodes[n]['atom_symbol'] == '*':
        return False
    return True

  @classmethod
  def get_parsers(cls):
    return [subclass.__name__.replace('Parser', '') for subclass in cls.__subclasses__()]
  
  @classmethod
  def create_parser(cls, framework: str, g: nx.DiGraph, no_wildcards: bool = False, group_diameter: bool = False):
    for subclass in cls.__subclasses__():
      if subclass.__name__ == framework + 'Parser':
        return subclass(g, no_wildcards, group_diameter)
    raise ValueError(f"Parser for framework '{framework}' not found")
  
class CuTSParser(Parser):
  def parse(self, file):
    g = self.join_graphs()
    print(g.number_of_nodes(), file=file)
    for a, b in tqdm(g.edges(), desc="Printing edges", file=sys.stderr):
      print(a, b, file=file, sep='\t')

class GSIParser(Parser):
  def parse(self, file):
    g = self.join_graphs()
    print("t # 0", file=file)
    print(len(g.nodes), len(g.edges), NUM_LABELS, 1, file=file)
    for i in tqdm(range(len(g.nodes)), desc="Printing nodes", file=sys.stderr):
      print('v', i, getNodeLabel(g.nodes[i]), file=file)
    
    for u, v in tqdm(g.edges, desc="Printing edges", file=sys.stderr):
      print('e', u, v, 1, file=file)
    
    print('t # -1', file=file)

class VF3Parser(Parser):
  def parse(self, file):
    g = self.join_graphs()
    print(len(g.nodes), file=file)
    for n in g.nodes:
      print(n, getNodeLabel(g.nodes[n], True), file=file)
    
    for node in g.nodes:
      neighbors = list(g.neighbors(node))
      print(len(neighbors), file=file)
      for neighbor in neighbors:
        print(node, neighbor, file=file)

class SIGMOParser(Parser):
  def parse(self, file):
    if self.group_diameter:
      diameter_groups = {}
      for g in self.graphs:
        d = nx.diameter(g)
        if d not in diameter_groups:
          diameter_groups[d] = []
        diameter_groups[d].append(g)
      
      max_num_graphs = max([len(diameter_groups[d]) for d in diameter_groups])
      
      for d in diameter_groups:
        to_add = max_num_graphs - len(diameter_groups[d])
        while to_add > 0:
          curr_len = len(diameter_groups[d])
          if to_add < curr_len:
            diameter_groups[d].extend(diameter_groups[d][:to_add])
          else:
            diameter_groups[d].extend(diameter_groups[d])
          to_add = max_num_graphs - len(diameter_groups[d])

      for d, graphs in diameter_groups.items():
        with open(f'{file}/query_d{d}.dat', 'w') as dfile:
          for g in graphs:
            g: nx.DiGraph
            g = g.to_undirected()
            print(f'n#{g.number_of_nodes()} l#{NUM_LABELS}',  end=' ', file=dfile)
            for i, n in enumerate(g.nodes):
              print(i, getNodeLabel(g.nodes[i]), end=' ', file=dfile)
            print(f'e#{g.number_of_edges()}', end=' ', file=dfile)
            for a, b, d in g.edges(data=True):
              print(a, b, getEdgeLabel(d), end=' ', file=dfile)
            print(file=dfile)
    else:
      for g in self.graphs:
        g: nx.DiGraph
        g = g.to_undirected()
        print(f'n#{g.number_of_nodes()} l#{NUM_LABELS}',  end=' ', file=file)
        for i, n in enumerate(g.nodes):
          print(i, getNodeLabel(g.nodes[i]), end=' ', file=file)
        print(f'e#{g.number_of_edges()}', end=' ', file=file)
        for a, b, d in g.edges(data=True):
          print(a, b, getEdgeLabel(d), end=' ', file=file)
        print(file=file)



if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Convert SMILES to graph')
  parser.add_argument('--output', '-o', type=str, help='Output file, if None, print to stdout')
  parser.add_argument('--framework', '-f', choices=Parser.get_parsers(), type=str, help='Framework to use', default='cuts')
  parser.add_argument('--no-wildcards', action='store_true', help='Do not use wildcards')
  parser.add_argument('--group-diameter', action='store_true', help='Group graphs by diameter and creates a file for each graph diameter')
  args = parser.parse_args()
  if sys.stdin.isatty():
    print(f'Usage: python {sys.argv[0]} < smarts_file')
    sys.exit(1)
  
  lines = sys.stdin.readlines()
  graphs = get_graph_list(lines)
  
  if args.output:
    if os.path.isdir(args.output):
      Parser.create_parser(args.framework, graphs, args.no_wildcards, args.group_diameter).parse(args.output)
    else:
      with open(args.output, 'w') as f:
        Parser.create_parser(args.framework, graphs, args.no_wildcards, args.group_diameter).parse(f)
  else:
    Parser.create_parser(args.framework, graphs, args.no_wildcards, args.group_diameter).parse(sys.stdout)