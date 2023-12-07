#!/usr/bin/env python

import sys

if __name__ == '__main__':
  if len(sys.argv) != 3:
    print("Usage: ./parseVF3.py <vf3_file> <out_file>")
    sys.exit(1)
  
  vf3_file = sys.argv[1]
  output_file = sys.argv[2]
  
  with open(vf3_file, 'r') as f:
    lines = f.readlines()
  
  with open(output_file, 'w') as f:
    print('query,data,solutions,time[us]', file=f)
    for i in range(0, len(lines), 2):
      data, query = map(lambda x: x.strip().replace('data_', '').replace('query_', '').replace('.dat', ''), lines[i].split('-'))
      solutions, time = map(lambda x: x.strip(), lines[i+1].split(' '))
      print(','.join([query, data, solutions, str(int(float(time) * 1000000))]), file=f)
