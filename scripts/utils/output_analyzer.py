import glob
import pandas as pd

# time in ms, memory in bytes
class SIGMOOutputAnalyzer:
  def __init__(self, out, err = ''):
    if err:
      self.candidates_sizes = []
      self._parseNumCandidates(err)
    self.n_refinement_steps = 0
    self._lines = [line.strip() for line in out.split('\n')]
    self.parse()
  
  def _parseMemory(self, value, unit):
    value = float(value)
    if unit == 'B':
      return value
    elif unit == 'KB':
      return value * 1024
    elif unit == 'MB':
      return value * 1024 * 1024
    elif unit == 'GB':
      return value * 1024 * 1024 * 1024
  
  def _parseGPUStats(self, start):
    for i in range(1, 6):
      line = self._lines[start + i]
      if line.startswith('Data signature time'):
        self.data_signature_gpu_time = int(line.split(' ')[-2])
      elif line.startswith('Query signature time'):
        self.query_signature_gpu_time = int(line.split(' ')[-2])
      elif line.startswith('Filter time'):
        self.filter_gpu_time = int(line.split(' ')[-2])
      elif line.startswith('Join time'):
        self.join_gpu_time = int(line.split(' ')[-2])
      elif line.startswith('Total time:'):
        self.total_gpu_time = int(line.split(' ')[-2])
      
  def _parseConfigs(self, start):
    for i in range(1, 5):
      line = self._lines[start + i]
      if line.startswith('Filter domain'):
        self.filter_domain = line.split(' ')[-1]
      elif line.startswith('Filter Work Group Size'):
        self.filter_work_group_size = int(line.split(' ')[-1])
      elif line.startswith('Join Work Group Size'):
        self.join_work_group_size = int(line.split(' ')[-1])
      elif line.startswith('Find all'):
        self.find_all = line.split(' ')[-1] == 'Yes'

  def _parseHostStats(self, start):
    for i in range(1, 6):
      line = self._lines[start + i]
      if line.startswith('Setup Data time'):
        self.setup_data_host_time = int(line.split(' ')[3])
      elif line.startswith('Filter time'):
        self.filter_host_time = int(line.split(' ')[-2])
      elif line.startswith('Mapping time'):
        self.mapping_host_time = int(line.split(' ')[-2])
      elif line.startswith('Join time'):
        self.join_host_time = int(line.split(' ')[-2])
      elif line.startswith('Total time:'):
        self.total_time = int(line.split(' ')[-2])
  
  def _parseNumCandidates(self, err):
    for line in err.split('\n'):
      if line.startswith('Node'):
        self.candidates_sizes.append(int(line.split(' ')[-1]))
  
  def parse(self):
    for i, line in enumerate(self._lines):
      if line.startswith('# Query Nodes'):
        self.n_query_nodes = int(line.split(' ')[-1])
      elif line.startswith('# Query Graphs'):
        self.n_query_graphs = int(line.split(' ')[-1])
      elif line.startswith('# Data Nodes'):
        self.n_data_nodes = int(line.split(' ')[-1])
      elif line.startswith('# Data Graphs'):
        self.n_data_graphs = int(line.split(' ')[-1])
      elif "Configs" in line:
        self._parseConfigs(i)
      elif line.endswith('B for graph data'):
        self.graph_data_memory = self._parseMemory(line.split(' ')[1], line.split(' ')[2])
      elif line.endswith('B for query data'):
        self.query_data_memory = self._parseMemory(line.split(' ')[1], line.split(' ')[2])
      elif line.endswith('B for candidates'):
        self.candidates_memory = self._parseMemory(line.split(' ')[1], line.split(' ')[2])
      elif line.endswith('B for data signatures'):
        self.data_signatures_memory = self._parseMemory(line.split(' ')[1], line.split(' ')[2])
      elif line.endswith('B for query signatures'):
        self.query_signatures_memory = self._parseMemory(line.split(' ')[1], line.split(' ')[2])
      elif line.startswith('Total allocated memory'):
        self.total_memory = self._parseMemory(line.split(' ')[3], line.split(' ')[4])
      elif line.startswith('# Total candidates:'):
        self.n_total_candidates = int(line.split(' ')[-1].replace('.', ''))
      elif line.startswith('# Average candidates:'):
        self.n_avg_candidates = int(line.split(' ')[-1].replace('.', ''))
      elif line.startswith('# Median candidates:'):
        self.n_median_candidates = int(line.split(' ')[-1].replace('.', ''))
      elif line.startswith('# Zero candidates:'):
        self.n_zero_candidates = int(line.split(' ')[-1].replace('.', ''))
      elif line.startswith('# Matches:'):
        self.n_matches = int(line.split(' ')[-1].replace('.', ''))
      elif line.startswith('[*] Refinement step'):
        self.n_refinement_steps = int(line.split(' ')[-1].replace(':', ''))
      elif "Overall GPU Stats" in line:
        self._parseGPUStats(i)
      elif "Overall Host Stats" in line:
        self._parseHostStats(i)
      elif line.startswith("Node 0:"):
        self._parseNumCandidates(i)
  
  def get_df_str(self):
    header = []
    vals = []
    for key, value in sorted(self.__dict__.items(), key=lambda x: x[0]):
      if key.startswith('_'):
        continue
      header.append(key)
      if key == 'candidates_sizes':
        v = '-'.join([str(x) for x in value])
        vals.append(v)
      else:
        vals.append(str(value))
    return header, vals

  def get_df_headers(self):
    return list(sorted(self.__dict__.keys()))
      

if __name__ == '__main__':
  import argparse
  
  argparser = argparse.ArgumentParser(description='SIGMO Output Analyzer')
  argparser.add_argument('input', type=str, help='Directory of logs')
  argparser.add_argument('output', type=str, help='Output CSV file, if None, print to stdout')
  args = argparser.parse_args()
  
  headers = []
  vals = []

  output_files = sorted(glob.glob(f'{args.input}/sigmo*.log'))
  error_files = sorted(glob.glob(f'{args.input}/err*.log'))
  
  print(f'Found {len(output_files)} output files and {len(error_files)} error files.')

  for (o, e) in zip(output_files, error_files):
    with open(o, 'r') as f:
      out = f.read()
    with open(e, 'r') as f:
      err = f.read()
    out = SIGMOOutputAnalyzer(out, err)
    if len(headers) == 0:
      headers = out.get_df_str()[0]
    vals.append(out.get_df_str()[1])

  df = pd.DataFrame(columns=headers)
  for val in vals:
    df.loc[len(df)] = val
  df = df.sort_values(by='n_refinement_steps')

  df.to_csv(args.output, index=False)
  del df, headers, vals

