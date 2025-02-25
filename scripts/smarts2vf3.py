#!/usr/bin/env python3

import sys
from utils import *

if __name__ == '__main__':
  if sys.stdin.isatty():
    print(f'Usage: python {sys.argv[0]} < smarts_file')
    sys.exit(1)

  lines = sys.stdin.readlines()
  g = process_graph(lines)
  
  print(len(g.nodes))
  for n in g.nodes:
    print(n, getLabel(g.nodes[n], True))
  
  for node in g.nodes:
    neighbors = list(g.neighbors(node))
    print(len(neighbors))
    for neighbor in neighbors:
      print(node, neighbor)
