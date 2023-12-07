#!/usr/bin/env python3

from utils import smartsToGraph, getLabel
import sys

if __name__ == '__main__':
  if len(sys.argv) < 2:
    print('Usage: python3 smarts2vf2.py <smarts>')
    sys.exit(1)
  mol = sys.argv[1]
  
  g = smartsToGraph(mol)

  nodes = []

  for node, node_value in g.nodes.data():
    label = getLabel(node_value)
    nodes.append(label)
  
  print(len(nodes), len(g.edges))
  print(*nodes, sep='\n')
  for a, b in g.edges:
    print(a, b)
