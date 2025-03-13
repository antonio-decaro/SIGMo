# Copyright (c) 2025 University of Salerno
# SPDX-License-Identifier: Apache-2.0

from abc import abstractmethod
import argparse
import re
import pandas as pd

COLUMNS = ['query', 'first_match(s)', 'time(s)', 'solutions']

class Parser:
  def __init__(self, data):
    self.contents = []
    self.dataframe = pd.DataFrame(columns=COLUMNS)
    self.dataframe['query'] = self.dataframe['query'].astype(int)
    self.dataframe['solutions'] = self.dataframe['solutions'].astype(int)
    
    current_content = []
    for line in data.split('\n'):
      if line.startswith('query_') and line.endswith('.dat'):
        if current_content:
          self.contents.append('\n'.join(current_content))
          current_content = []
      current_content.append(line)
    if current_content:
      self.contents.append('\n'.join(current_content))
  
  @abstractmethod
  def parse(self, file) -> pd.DataFrame:
    pass

  @classmethod
  def get_parsers(cls):
    return [subclass.__name__.replace('Parser', '') for subclass in cls.__subclasses__()]
  
  @classmethod
  def create_parser(cls, framework: str, data: str):
    for subclass in cls.__subclasses__():
      if subclass.__name__ == framework + 'Parser':
        return subclass(data)
    raise ValueError(f"Parser for framework '{framework}' not found")
  
  
class CuTSParser(Parser):
  def parse(self):
    for content in self.contents:
      content = content.split('\n')
      if len(content) < 6:
        continue
      content = content[5].split(',')
      query_idx = content[1].replace('query_', '').replace('.dat', '')
      matching = first_solution = float(content[2].replace('ms', '')) / 1000
      self.dataframe.loc[len(self.dataframe)] = [int(query_idx), first_solution, matching, int(content[3])]
    return self.dataframe

class GSIParser(Parser):
  def parse(self):
    for content in self.contents:
      lines = [line.strip() for line in content.split('\n')]
      query_idx = int(lines[0].replace('query_', '').replace('.dat', ''))
      time = 0
      solutions = 0
      for line in lines[1:]:
        if line.startswith('total time used:'):
          tmp = line.split(' ')[-1]
          time = float(tmp.replace('us', '')) / 1e6
        if line.startswith('result:'):
          row = line.split(' ')[1]
          col = line.split(' ')[2]
          solutions = int(row) * int(col)
      self.dataframe.loc[len(self.dataframe)] = [query_idx, time, time, solutions]
    return self.dataframe

class VF3Parser(Parser):
  def parse(self):
    query_idx_patt = re.compile(r'query_(\d+)\.dat')
    
    for query in self.contents:
      query_idx = int(query_idx_patt.search(query).group(1))
      lines = [l.strip() for l in query.split('\n')]
      first_solution = float(lines[4].split(' ')[-1])
      matching = float(lines[5].split(' ')[-1])
      solutions = int(lines[6].split(' ')[-1])
      if solutions == 0:
        first_solution = matching
      self.dataframe.loc[len(self.dataframe)] = [query_idx, first_solution, matching, solutions]
    return self.dataframe


class MBSMParser(Parser):
  def parse(self):
    pass



if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Parse framework output')
  parser.add_argument('file', type=str, help='File to parse')
  parser.add_argument('--output', '-o', type=str, help='Output file, if None, print to stdout')
  parser.add_argument('--framework', '-f', choices=Parser.get_parsers(), type=str, help='Framework to use')
  args = parser.parse_args()
  
  data = ""
  with open(args.file) as f:
    data = f.read()
  
  out: pd.DataFrame = Parser.create_parser(args.framework, data).parse()
  if args.output:
    out.to_csv(args.output, index=False)
  else:
    # print(out.to_string())
    print("Total time (s):", out['time(s)'].sum())
    print(f"Total matches: {int(out['solutions'].sum()):,}")
    print("First match time (s):", out['first_match(s)'].sum())
    print("Number of matched queries:", len(out[out['solutions'] > 0]))