#!/usr/bin/env python3

import sys
import numpy
import pandas as pd

if __name__ == '__main__':
  if len(sys.argv) != 2:
    print("Usage: ./process.py <csv_file>")
    sys.exit(1)
  
  csv_file = sys.argv[1]
  df = pd.read_csv(csv_file)
  df['solutions'] = df['solutions'].astype(int)
  df['time[us]'] = df['time[us]'].astype(int)
  
  total_queries = df['query'].nunique()
  total_data = df['data'].nunique()
  
  print('Algorithm:', csv_file.split('/')[-1].replace('.csv', '').upper())
  print('# query graphs:', total_queries)
  print('# data graphs:', total_data)
  print('# found solutions:', df['solutions'].sum())
  print('Total time:', df['time[us]'].sum(), 'us')