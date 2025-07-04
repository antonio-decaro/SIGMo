#!/usr/bin/env python3

import argparse
import os
import networkx as nx
import pysmiles
import sys
import glob
from tqdm import tqdm

organic_subset = {l: (i) for i, l in enumerate('N Cl * Br I P H O C S F B Sn'.split())}

def smileToGraph(smarts: str) -> nx.DiGraph:
  return pysmiles.read_smiles(smarts, explicit_hydrogen=True, zero_order_bonds=True)

def getLabel(g: dict):
  if g['element'] in organic_subset:
    return organic_subset[g['element']]
  else:
    print(f"Element {g['element']} not in organic subset, adding it", file=sys.stderr)
    organic_subset[g['element']] = len(organic_subset)
    return organic_subset[g['element']]

def printGraph(g: nx.DiGraph, file):
  g = g.to_undirected()
  if g.number_of_nodes() > 64:
    raise ValueError(f"Graph has more than 64 nodes: {g.number_of_nodes()}")
  print(f'n#{g.number_of_nodes()} l#{len(organic_subset)}',  end=' ', file=file)
  for i, n in enumerate(g.nodes):
    print(i, getLabel(g.nodes[i]), end=' ', file=file)
  print(f'e#{g.number_of_edges()}', end=' ', file=file)
  for a, b in g.edges:
    print(a, b, end=' ', file=file)
  print(file=file)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Converts a SMILES file to a graph file')
  parser.add_argument('id', help='id', type=int, default=0)
  parser.add_argument('folder', help='ZINC dataset folder', type=str)
  args = parser.parse_args()

  files = glob.glob(os.path.join(args.folder, "*.smi"))

  file = files[args.id]
  fname = file.split("/")[-1].split(".")[0]
  out_file = os.path.join(args.folder, f"{args.id}_{fname}.dat")
  
  with open(out_file, "w") as f:
    with open(file) as f_in:
      for line in f_in.readlines():
        try:
          line = line.strip().split()[0]
          g = smileToGraph(line)
          if g is not None:
            printGraph(g, f)
        except Exception as e:
          print(f"Error on {line}: {e}")
  print(f"Done with {file}")
