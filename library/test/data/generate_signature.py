import networkx as nx
import numpy as np
import sys
import matplotlib.pyplot as plt

def generate_signature(graph):
    # Initialize a 32-bit integer
    signatures = []
    
    # Modify the signature according to the labels
    for node in graph.nodes(data=True):
        signature = np.int32(0)
        label_counts = [0] * 16  # Assuming there are 16 possible labels
        
        # Count the number of adjacent nodes with each label
        for neighbor in graph.neighbors(node[0]):
            neighbor_label = graph.nodes[neighbor].get('label', 0)
            if neighbor_label < 16:
                label_counts[neighbor_label] += 1
        
        # Map the counts to the signature
        for i, count in enumerate(label_counts):
            if count > 3:
                count = 3  # Limit the count to 2 bits (0-3)
            signature |= (count << (i * 2))
        
        signatures.append(signature)
    
    return signatures

def read_graph(line):
    parts = line.split()
    num_nodes = int(parts[0][2:])
    num_labels = int(parts[1][2:])
    graph = nx.Graph(directed=False)
    
    index = 2
    for _ in range(num_nodes):
        node = int(parts[index])
        label = int(parts[index + 1])
        graph.add_node(node, label=label)
        index += 2
    
    num_edges = int(parts[index][2:])
    index += 1
    for _ in range(num_edges):
        u = int(parts[index])
        v = int(parts[index + 1])
        graph.add_edge(u, v)
        index += 2
    
    return graph

def print_signature(signature):
    details = []
    for i in range(16):
        count = (signature >> (i * 2)) & 0b11
        details.append(f"{i}: [{count}]")
    # print("Signature details:", ", ".join(details))
    print('{', f"0b{format(signature, '032b')}", '},', sep='')

if __name__ == '__main__':
    fname = sys.argv[1]
    
    # Load the graph
    with open(fname, 'r') as f:
        lines = f.readlines()
    
    graphs = []
    for line in lines:
        graph = read_graph(line.strip())
        graphs.append(graph)

    signatures = {}
    for i, graph in enumerate(graphs):
        signature = generate_signature(graph)
        signatures[i] = signature
    
    # Print the signatures in binary with details
    for i, signature in signatures.items():
        print(f"//Graph {i+1}")
        for n, sig in enumerate(signature):
            # print(f"Node {n}:")
            print_signature(sig)




