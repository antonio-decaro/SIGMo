#!/usr/bin/env python3

import sys
from utils import smartsToGraph, getLabel

if __name__ == '__main__':
  if len(sys.argv) != 2:
    print('Usage: python3 smarts2vf2.py <smarts>')
    sys.exit(1)
  mol = sys.argv[1]
  
  g = smartsToGraph(mol)
  
  print(len(g.nodes))
  for n in g.nodes:
    print(n, getLabel(g.nodes[n], True))
  
  for node in g.nodes:
    neighbors = list(g.neighbors(node))
    print(len(neighbors))
    for neighbor in neighbors:
      print(node, neighbor)
