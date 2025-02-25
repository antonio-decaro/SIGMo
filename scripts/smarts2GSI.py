#!/usr/bin/env python3
import sys
from utils import *
from tqdm import tqdm

if __name__ == '__main__':
  if sys.stdin.isatty():
    print(f'Usage: python {sys.argv[0]} < smarts_file')
    sys.exit(1)

  lines = sys.stdin.readlines()
  g = process_graph(lines)
  
  print("t # 0")
  print(len(g.nodes), len(g.edges), NUM_LABELS, 1)
  for i in tqdm(range(len(g.nodes)), desc="Printing nodes", file=sys.stderr):
    print('v', i, getLabel(g.nodes[i]))
  
  for u, v in tqdm(g.edges, desc="Printing edges", file=sys.stderr):
    print('e', u, v, 1)
  
  print('t # -1')