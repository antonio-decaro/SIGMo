#!/usr/bin/env python

import sys

if __name__ == '__main__':
  if len(sys.argv) != 3:
    print("Usage: python parseVF3.py <vf3_file>")
    sys.exit(1)
  
  vf3_file = sys.argv[1]
  output_file = sys.argv[2]
  
  with open(vf3_file, 'r') as f:
    solutions, time = f.readlines()[-1].split()
  
  with open(output_file, 'a') as f:
    f.write(f"{time} s", file=f)
  


