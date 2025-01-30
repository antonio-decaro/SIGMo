#!/usr/bin/env python3
import sys
from utils import *

if __name__ == '__main__':
  if len(sys.argv) < 2:
    print('Usage: python smarts2GSI.py <smarts>')
    sys.exit(1)
    
  mol = sys.argv[1]
  g = smartsToGraph(mol)
  
  print("t # 0")
  print(len(g.nodes), len(g.edges), NUM_LABELS, 1)
  for i in range(len(g.nodes)):
    print('v', i, getLabel(g.nodes[i]))
  
  for u, v in g.edges:
    print('e', u, v, 1)
  
  print('t # -1')