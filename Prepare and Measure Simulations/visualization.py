import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D
import logging
import os
import numpy as np

def visualize_network(network, topology="fully_connected", output_dir=None):
    if output_dir is None:
        output_dir = os.getcwd()

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
                G.add_edge(node1_name, node2_name, label="Quantum Channel", color='blue', style='solid', weight=2)
            elif 'classical' in connection_str:
                G.add_edge(node1_name, node2_name, label="Classical Channel", color='green', style='dashed', weight=2)

    # Draw the graph with appropriate layout for each topology
    if topology == "ring":
        pos = nx.circular_layout(G)
    elif topology == "bus":
         # For bus topology, create a horizontal layout manually positioned in the center
        nodes = list(G.nodes())
        nodes.sort(key=lambda x: int(x.split('_')[1]))  # Sort by node number
        
        # Calculate figure dimensions to center properly
        fig_width, fig_height = plt.gcf().get_size_inches()
        aspect_ratio = fig_width / fig_height
        
        # Position nodes in a horizontal line in the center of the figure
        node_count = len(nodes)
        x_positions = np.linspace(0, 1, node_count)
        y_position = 0.5  # Center vertically

        pos = {nodes[i]: (x_positions[i], y_position) for i in range(node_count)}

        plt.xlim(-0.1, 1.1)
        plt.ylim(-0.1, 1.1) 
    elif topology == "star":
        # For star topology, central node in center, others around it
        pos = nx.kamada_kawai_layout(G)
        # Ensure Node_0 is at the center
        if "Node_0" in pos:
            pos["Node_0"] = (0, 0)
    else:
        pos = nx.spring_layout(G, seed=42)
        
    nx.draw_networkx_nodes(G, pos, node_size=3000, node_color='skyblue')
    nx.draw_networkx_labels(G, pos, font_size=16, font_weight='bold')

    for u, v, key, data in G.edges(keys=True, data=True):
        style = data.get('style', 'solid')
        color = data.get('color', 'black')
        weight = data.get('weight', 1)
        curve = "arc3,rad=0.3" if data['label'] == "Quantum Channel" else "arc3,rad=-0.3"
        nx.draw_networkx_edges(
            G, pos, edgelist=[(u, v)],
            edge_color=color,
            style=style,
            width=weight,
            connectionstyle=curve
        )

    # Add legend
    legend_elements = [
        Line2D([0], [0], color='blue', lw=2, label='Quantum Channel'),
        Line2D([0], [0], color='green', lw=2, linestyle='dashed', label='Classical Channel')
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=20)
    plt.title(f"QKD Network Visualisation ({topology} topology)", fontsize=28, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    output_path = os.path.join(output_dir, f"network_visualisation_{topology}.png")
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    logging.info(f"Network visualisation saved as '{output_path}'")

# Create a function to create plots based on total distance
def create_total_distance_plots(all_results, topologies, output_dir, node_pairs, num_nodes, max_distance=150, topology_colors=None):
    """Plot key rate and QBER vs total distance, grouped by topology and hop count."""
    # Use consistent colors for topologies
    if topology_colors is None:
        topology_colors = {
            'ring': '#1f77b4',    # Blue
            'star': '#ff7f0e',    # Orange
            'bus': '#2ca02c',     # Green
            'baseline': '#d62728'  # Red
        }
    
    # Define markers for hop counts (not topologies)
    hop_markers = {
        0: 'o',     # Circle for 0 hops (direct connection)
        1: 's',     # Square for 1 hop
        2: '^',     # Triangle for 2 hops
        3: 'D',     # Diamond for 3 hops
        4: 'p',     # Pentagon for 4 hops
        5: 'h',     # Hexagon for 5 hops
        'baseline': 'X'  # X for baseline
    }
    
    # Gather data for all topologies and connection types
    topology_data = {}
    
    for topology in topologies:
        topology_data[topology] = {'distances': [], 'rates': [], 'qbers': [], 'hop_counts': [], 'total_distances': []}
        
        for channel_length in sorted(all_results.keys()):
            if topology in all_results[channel_length]:
                for i, rate in enumerate(all_results[channel_length][topology]["rates"]):
                    try:
                        total_dist = all_results[channel_length][topology]["total_distances"][i]
                        
                        # Skip if beyond our max distance
                        if total_dist > max_distance:
                            continue
                            
                        qber = all_results[channel_length][topology]["qbers"][i]
                        # Calculate path length (number of hops)
                        if i < len(node_pairs):
                            node_pair = node_pairs[topology][i] if isinstance(node_pairs, dict) else node_pairs[i]
                            from network import calculate_path
                            path = calculate_path(node_pair[0], node_pair[1], num_nodes, topology)
                            hop_count = len(path) - 1
                        else:
                            hop_count = 1  # Default if node_pair info is not available
                        
                        topology_data[topology]['distances'].append(channel_length)
                        topology_data[topology]['rates'].append(rate)
                        topology_data[topology]['qbers'].append(qber)
                        topology_data[topology]['hop_counts'].append(hop_count)
                        topology_data[topology]['total_distances'].append(total_dist)
                    except (IndexError, KeyError) as e:
                        logging.error(f"Error processing results: {e}")
                        continue
    
    # Create key rate vs total distance plot
    plt.figure(figsize=(12, 8))
    
    # Plot baseline point-to-point for reference
    baseline_distances = [d for d in sorted(all_results.keys()) if d <= max_distance]
    baseline_rates = [all_results[d]["baseline"]["rates"][0] for d in baseline_distances]
    plt.plot(baseline_distances, baseline_rates, 
             marker=hop_markers['baseline'], linestyle='-',
             linewidth=2, markersize=10, color=topology_colors['baseline'],
             label='Point-to-Point (No Relays)')
    
    # Plot data points for each topology and hop count
    for topology in topologies:
        data = topology_data[topology]
        
        # Group by hop count
        by_hops = {}
        for dist, rate, hops, total_dist in zip(data['distances'], data['rates'], data['hop_counts'], data['total_distances']):
            if hops not in by_hops:
                by_hops[hops] = {'distances': [], 'rates': [], 'total_distances': []}
            by_hops[hops]['distances'].append(dist)
            by_hops[hops]['rates'].append(rate)
            by_hops[hops]['total_distances'].append(total_dist)
        
        # Plot each hop count separately but with same topology color
        for hops, hop_data in sorted(by_hops.items()):
            # Sort by distance
            sorted_indices = np.argsort(hop_data['total_distances'])
            sorted_distances = [hop_data['total_distances'][i] for i in sorted_indices]
            sorted_rates = [hop_data['rates'][i] for i in sorted_indices]
            
            
            marker = hop_markers.get(hops, 'o')  
            plt.plot(sorted_distances, sorted_rates, 
                    marker=marker, linestyle='-', 
                    color=topology_colors[topology],
                    linewidth=2, markersize=10,
                    label=f'{topology.capitalize()} ({hops} hops)')
    
    plt.xlabel('Total End-to-End Distance (km)', fontsize=14)
    plt.ylabel('Key Exchange Rate (bits/unit time)', fontsize=14)
    plt.title('Key Rate vs Total Distance', fontsize=28, fontweight='bold')
    plt.xlim(0, max_distance)  
    plt.grid(True)
    
    
    handles, labels = plt.gca().get_legend_handles_labels()
    
    def sort_key(item):
        label = item[1]
        if 'Point-to-Point' in label:
            return (0, '')
        topology = label.split(' (')[0].lower()
        hop_str = label.split('(')[1].split(' ')[0]
        try:
            hops = int(hop_str)
        except:
            hops = 99  # Put unknown hop counts at the end
        return (1, topology, hops)
    
    sorted_pairs = sorted(zip(handles, labels), key=sort_key)
    sorted_handles, sorted_labels = zip(*sorted_pairs)
    
    plt.legend(sorted_handles, sorted_labels, fontsize=20, loc='upper right')
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "key_rate_vs_total_distance.png")
    plt.savefig(output_path, dpi=300)
    logging.info(f"Key rate vs total distance plot saved to {output_path}")
    
    # Create QBER vs total distance plot with the same color/marker scheme
    plt.figure(figsize=(12, 8))
    
    # Plot baseline QBER
    baseline_qbers = [all_results[d]["baseline"]["qbers"][0] for d in baseline_distances]
    plt.plot(baseline_distances, baseline_qbers, 
             marker=hop_markers['baseline'], linestyle='-',
             linewidth=2, markersize=10, color=topology_colors['baseline'],
             label='Point-to-Point (No Relays)')
    
    # Plot QBER by total distance and hops, with same styling approach
    for topology in topologies:
        data = topology_data[topology]
        
        # Group by hop count
        by_hops = {}
        for dist, qber, hops, total_dist in zip(data['distances'], data['qbers'], data['hop_counts'], data['total_distances']):
            if hops not in by_hops:
                by_hops[hops] = {'distances': [], 'qbers': [], 'total_distances': []}
            by_hops[hops]['distances'].append(dist)
            by_hops[hops]['qbers'].append(qber)
            by_hops[hops]['total_distances'].append(total_dist)
        
        # Plot each hop count separately but with same topology color
        for hops, hop_data in sorted(by_hops.items()):
            # Sort by distance
            sorted_indices = np.argsort(hop_data['total_distances'])
            sorted_distances = [hop_data['total_distances'][i] for i in sorted_indices]
            sorted_qbers = [hop_data['qbers'][i] for i in sorted_indices]
            
            
            marker = hop_markers.get(hops, 'o')  
            plt.plot(sorted_distances, sorted_qbers, 
                    marker=marker, linestyle='-', 
                    color=topology_colors[topology],
                    linewidth=2, markersize=10,
                    label=f'{topology.capitalize()} ({hops} hops)')
    
    plt.xlabel('Total End-to-End Distance (km)', fontsize=14)
    plt.ylabel('Quantum Bit Error Rate (QBER)', fontsize=14)
    plt.title('QBER vs Total Distance', fontsize=28, fontweight='bold')
    plt.xlim(0, max_distance)  
    plt.grid(True)
    
    # Sort legend by topology and then hop count (same as above)
    handles, labels = plt.gca().get_legend_handles_labels()
    sorted_pairs = sorted(zip(handles, labels), key=sort_key)
    sorted_handles, sorted_labels = zip(*sorted_pairs)
    
    plt.legend(sorted_handles, sorted_labels, fontsize=20, loc='upper right')
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "qber_vs_total_distance.png")
    plt.savefig(output_path, dpi=300)
    logging.info(f"QBER vs total distance plot saved to {output_path}")