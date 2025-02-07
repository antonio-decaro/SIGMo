#!/usr/bin/env python3

from utils import smartsToGraph, getLabel, NUM_LABELS
import sys

if __name__ == '__main__':
  if len(sys.argv) < 2:
    print(f'Usage: python3 {sys.argv[0]} <smarts>')
    sys.exit(1)
  file = sys.argv[1]
  
  with open(file) as f:
    smarts = f.readlines()

  for s in smarts:
    s = s.strip()
    g = smartsToGraph(s)

    print(f'n#{g.number_of_nodes()} l#{NUM_LABELS}',  end=' ')
    for i, n in enumerate(g.nodes):
      print(i, getLabel(g.nodes[i]), end=' ')
    print(f'e#{g.number_of_edges()}', end=' ')
    for a, b in g.edges:
      print(a, b, end=' ')
    print()

