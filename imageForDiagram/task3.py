import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os

# Configuration based on your paper
n_nodes = 92  # Mean nodes per 2-slice graph [cite: 623]
hidden_dim = 256  # Your hidden layer dimension [cite: 728]
output_dir = "diagram_stage3"
os.makedirs(output_dir, exist_ok=True)

# 1. Create a synthetic "Activation Map"
# This simulates the 256-dim hidden state of the 92 nodes
np.random.seed(42)
hidden_activations = np.random.rand(n_nodes, hidden_dim)

# 2. Visualize the "Neural Latent Space"
# A high-density heatmap represents the complexity of the GraphSAGE backbone
plt.figure(figsize=(10, 4))
plt.imshow(hidden_activations.T, aspect='auto', cmap='magma')
plt.colorbar(label="Activation Intensity")
plt.xlabel("Graph Nodes (v_i)")
plt.ylabel("Hidden Channels (d=256)")
plt.title("Latent Feature Representation (Stage 3)")
plt.savefig(f"{output_dir}/gnn_hidden_space.png", bbox_inches='tight', dpi=300)

# 3. Graph Message Passing Visual
# Show the graph topology with "activated" (bright) tumor nodes
G = nx.erdos_renyi_graph(n_nodes, 0.05) # Sparse graph matching your 3.9 degree [cite: 655]
pos = nx.spring_layout(G)
node_colors = np.random.choice(['#ff0000', '#444444'], size=n_nodes, p=[0.1, 0.9])

plt.figure(figsize=(6, 6))
nx.draw(G, pos, node_size=50, node_color=node_colors, edge_color='gray', alpha=0.6)
plt.savefig(f"{output_dir}/message_passing_flow.png", transparent=True)
print(f"Generated Stage 3 visuals in {output_dir}")