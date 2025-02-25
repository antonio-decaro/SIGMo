#!/usr/bin/env python3
import sys
import networkx as nx
from utils import *

if __name__ == '__main__':
  if sys.stdin.isatty():
    print(f'Usage: python {sys.argv[0]} < smarts_file')
    sys.exit(1)

  lines = sys.stdin.readlines()
  g = process_graph(lines)
  
  print(g.number_of_nodes())
  for a, b in g.edges():
    print(a, b)