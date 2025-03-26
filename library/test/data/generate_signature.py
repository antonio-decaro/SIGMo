import networkx as nx
import numpy as np
import sys
import matplotlib.pyplot as plt

def generate_signature(graph, bits=4, refinement_steps=1):
    # Initialize a 64-bit integer
    signatures = []
    max_labels = 64 // bits
    max_count = (1 << bits) - 1
    
    # Modify the signature according to the node_labels
    for node in graph.nodes(data=True):
        signature = np.int64(0)
        label_counts = [0] * max_labels  # Adjust based on the number of bits
        
        # Count the number of adjacent nodes with each label
        for neighbor in graph.neighbors(node[0]):
            neighbor_label = graph.nodes[neighbor].get('label', 0)
            if neighbor_label < max_labels:
                label_counts[neighbor_label] += 1
        
        # Map the counts to the signature
        for i, count in enumerate(label_counts):
            if count > max_count:
                count = max_count  # Limit the count to the maximum value based on bits
            signature |= (count << (i * bits))
        
        signatures.append(signature)

    # Refine the signatures based on neighbor node_labels
    for _ in range(refinement_steps):
        new_signatures = []
        for node in graph.nodes(data=True):
            signature = signatures[node[0]]
            label_counts = [0] * max_labels  # Adjust based on the number of bits
            
            # Count the number of adjacent nodes with each label
            for neighbor in graph.neighbors(node[0]):
                neighbor_signature = signatures[neighbor]
                for i in range(max_labels):
                    count = (neighbor_signature >> (i * bits)) & ((1 << bits) - 1)
                    label_counts[i] += count
            
            # Map the counts to the signature
            for i, count in enumerate(label_counts):
                if count > max_count:
                    count = max_count  # Limit the count to the maximum value based on bits
                signature |= (count << (i * bits))
            
            new_signatures.append(signature)
        signatures = new_signatures
    
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

def print_signature(signature, bits=4):
    max_labels = 64 // bits
    details = []
    for i in range(max_labels):
        count = (signature >> (i * bits)) & ((1 << bits) - 1)
        details.append(f"{i}: [{count}]")
    # print("Signature details:", ", ".join(details))
    print('{', f"0b{format(signature, '064b')}", '},', sep='')

if __name__ == '__main__':
    fname = sys.argv[1]
    bits = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    refinement_steps = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    
    # Load the graph
    with open(fname, 'r') as f:
        lines = f.readlines()
    
    graphs = []
    for line in lines:
        graph = read_graph(line.strip())
        graphs.append(graph)

    signatures = {}
    for i, graph in enumerate(graphs):
        signature = generate_signature(graph, bits, refinement_steps)
        signatures[i] = signature
    
    # Print the signatures in binary with details
    for i, signature in signatures.items():
        print(f"//Graph {i+1}")
        for n, sig in enumerate(signature):
            # print(f"Node {n}:")
            print_signature(sig, bits)




