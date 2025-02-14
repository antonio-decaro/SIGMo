#!/usr/bin/env python3
import sys
from utils import *

if __name__ == '__main__':
  if len(sys.argv) < 2:
    print('Usage: python smarts2cuTS.py <smarts>')
    sys.exit(1)
    
  mol = sys.argv[1]
  g = smartsToGraph(mol)
  g = g.to_directed()
  
  print(g.number_of_nodes())
  for a, b in g.edges():
    print(a, b)

