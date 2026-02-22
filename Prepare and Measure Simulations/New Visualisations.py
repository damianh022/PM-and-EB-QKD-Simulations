import os
from pathlib import Path
import logging

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.lines import Line2D


def visualize_network(network, topology="fully_connected", output_dir=None, protocol="PM"):
    """
    Standalone network visualiser. `protocol` must be 'PM' or 'EB'.
    Titles and EB-star special labels follow user's specification.
    """
    if output_dir is None:
        output_dir = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)

    plt.clf()
    plt.figure(figsize=(12, 10))
    G = nx.MultiGraph()

    # Add nodes
    for node in network.nodes:
        G.add_node(str(node))

    # Add edges with attributes for quantum and classical channels
    for connection in network.connections:
        connection_str = str(connection)
        if "conn|" in connection_str and "<->" in connection_str:
            parts = connection_str.split("|")[1]
            node1_name, node2_name = parts.split("<->")

            if 'quantum' in connection_str:
                G.add_edge(node1_name,
                           node2_name,
                           label="Quantum Channel",
                           color='blue',
                           style='solid',
                           weight=2)
            elif 'classical' in connection_str:
                G.add_edge(node1_name,
                           node2_name,
                           label="Classical Channel",
                           color='green',
                           style='dashed',
                           weight=2)

    # Layout selection
    if topology == "ring":
        pos = nx.circular_layout(G)
    elif topology == "bus":
        nodes = list(G.nodes())
        try:
            nodes.sort(key=lambda x: int(x.split('_')[1]))
        except Exception:
            nodes.sort()
        node_count = len(nodes)
        x_positions = np.linspace(0, 1, node_count)
        y_position = 0.5
        pos = {nodes[i]: (x_positions[i], y_position) for i in range(node_count)}
        plt.xlim(-0.1, 1.1)
        plt.ylim(-0.1, 1.1)
    elif topology == "star":
        pos = nx.kamada_kawai_layout(G)
        if "Node_0" in pos:
            pos["Node_0"] = (0, 0)
    else:
        pos = nx.spring_layout(G, seed=42)

    # Prepare display labels
    display_labels = {n: str(n) for n in G.nodes()}

    # EB-star special labels
    if protocol.strip().upper() == 'EB' and topology == 'star':
        for n in G.nodes():
            if n == "Node_0":
                display_labels[n] = "Central source"
            else:
                display_labels[n] = "Node in Centre"

    nx.draw_networkx_nodes(G, pos, node_size=3000, node_color='skyblue')
    nx.draw_networkx_labels(G, pos, labels=display_labels, font_size=16, font_weight='bold')

    for u, v, key, data in G.edges(keys=True, data=True):
        style = data.get('style', 'solid')
        color = data.get('color', 'black')
        weight = data.get('weight', 1)
        label = data.get('label', '')
        curve = "arc3,rad=0.3" if label == "Quantum Channel" else "arc3,rad=-0.3"
        nx.draw_networkx_edges(G,
                               pos,
                               edgelist=[(u, v)],
                               edge_color=color,
                               style=style,
                               width=weight,
                               connectionstyle=curve)

    # Legend
    legend_elements = [
        Line2D([0], [0], color='blue', lw=2, label='Quantum Channel'),
        Line2D([0], [0], color='green', lw=2, linestyle='dashed', label='Classical Channel')
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=20)

    # Title rules
    prot = protocol.strip().upper()
    if prot == 'PM':
        title = f"PM-QKD Network ({topology}-topology)"
    elif prot == 'EB':
        title = f"EB-QKD Network ({topology.capitalize()}-Topology)"
    else:
        title = f"QKD Network ({topology} topology)"

    plt.title(title, fontsize=28, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()

    out_name = f"network_visualisation_{prot}_{topology}.png"
    output_path = os.path.join(output_dir, out_name)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    logging.info(f"Network visualisation saved as '{output_path}'")


# Minimal mock network class for running standalone
class MockNetwork:
    def __init__(self, nodes, connections):
        self.nodes = nodes
        self.connections = connections


def make_sample_network(num_nodes=5, topology='star'):
    nodes = [f"Node_{i}" for i in range(num_nodes)]
    connections = []

    if topology == 'star':
        for i in range(1, num_nodes):
            connections.append(f"conn|Node_0<->Node_{i}|quantum")
            connections.append(f"conn|Node_0<->Node_{i}|classical")
    elif topology == 'ring':
        for i in range(num_nodes):
            a = f"Node_{i}"
            b = f"Node_{(i+1) % num_nodes}"
            connections.append(f"conn|{a}<->{b}|quantum")
            connections.append(f"conn|{a}<->{b}|classical")
    elif topology == 'bus':
        for i in range(num_nodes - 1):
            a = f"Node_{i}"
            b = f"Node_{i+1}"
            connections.append(f"conn|{a}<->{b}|quantum")
            connections.append(f"conn|{a}<->{b}|classical")
    else:  # fully_connected fallback
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                connections.append(f"conn|Node_{i}<->Node_{j}|quantum")
                connections.append(f"conn|Node_{i}<->Node_{j}|classical")

    return MockNetwork(nodes, connections)


def main():
    out = Path.cwd() / "viz_outputs"
    out.mkdir(parents=True, exist_ok=True)

    # PM visualisations
    pm_star = make_sample_network(num_nodes=6, topology='star')
    visualize_network(pm_star, topology='star', output_dir=str(out), protocol='PM')

    pm_ring = make_sample_network(num_nodes=6, topology='ring')
    visualize_network(pm_ring, topology='ring', output_dir=str(out), protocol='PM')

    pm_bus = make_sample_network(num_nodes=6, topology='bus')
    visualize_network(pm_bus, topology='bus', output_dir=str(out), protocol='PM')

    # EB visualisations (special star labels applied)
    eb_star = make_sample_network(num_nodes=6, topology='star')
    visualize_network(eb_star, topology='star', output_dir=str(out), protocol='EB')

    eb_ring = make_sample_network(num_nodes=6, topology='ring')
    visualize_network(eb_ring, topology='ring', output_dir=str(out), protocol='EB')

    print(f"Visualisations saved to {out.resolve()}")


if __name__ == "__main__":
    main()