import matplotlib.pyplot as plt
import numpy as np
import logging
import os

def upper_limit(max_value):
    """Pick a clean y-axis upper bound slightly above max_value."""
    if max_value == 0:
        return 1
    target = max_value * 1.1
    power = np.floor(np.log10(target))
    scale = 10 ** power
    scaled_value = target / scale
    if scaled_value <= 1:
        nice_value = 1
    elif scaled_value <= 2:
        nice_value = 2
    elif scaled_value <= 5:
        nice_value = 5
    else:
        nice_value = 10
    upper = nice_value * scale
    if upper <= max_value:
        if nice_value == 1:
            upper = 2 * scale
        elif nice_value == 2:
            upper = 5 * scale
        else:
            upper = 10 * scale
    return upper

def create_baseline_plots(distances_list, baseline_rates, baseline_qbers, output_dir, topology_colors, hop_markers):
    """Plot P2P baseline key rate and QBER vs distance (linear and log scales)."""
    plt.figure(figsize=(7, 5))
    rates_list = [baseline_rates[dist] for dist in distances_list]
    plt.plot(distances_list, rates_list, marker=hop_markers['baseline'], linestyle='-',
             linewidth=1.5, markersize=6, color=topology_colors['baseline'])
    plt.xlabel('Total End-to-End Distance (km)', fontsize=13)
    plt.ylabel('Key Exchange Rate (bits/second)', fontsize=13)
    plt.title('Point-to-Point Key Exchange Rate vs. Distance\n(No Relay Nodes)', fontsize=16, fontweight='bold')
    plt.grid(True)
    plt.xlim(10, 200)
    plt.xticks(np.arange(10, 201, 10), fontsize=9)
    plt.yticks(fontsize=9)
    y_max = max(rates_list)
    plt.ylim(0, upper_limit(y_max))
    plt.tight_layout()
    plt.legend(fontsize=11, loc='upper right')
    output_path = os.path.join(output_dir, "p2p_baseline_only.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    logging.info(f"Point-to-point baseline graph saved to {output_path}")

    plt.figure(figsize=(7, 5))
    qbers_list = [baseline_qbers[dist] for dist in distances_list]
    plt.plot(distances_list, qbers_list, marker=hop_markers['baseline'], linestyle='-',
             linewidth=1.5, markersize=6, color=topology_colors['baseline'])
    plt.xlabel('Total End-to-End Distance (km)', fontsize=13)
    plt.ylabel('Quantum Bit Error Rate (QBER)', fontsize=13)
    plt.title('Point-to-Point QBER vs. Distance\n(No Relay Nodes)', fontsize=16, fontweight='bold')
    plt.grid(True)
    plt.xlim(10, 200)
    plt.xticks(np.arange(10, 201, 10), fontsize=9)
    plt.yticks(fontsize=9)
    y_max = max(qbers_list)
    plt.ylim(0, upper_limit(y_max))
    plt.tight_layout()
    plt.legend(fontsize=11, loc='upper right')
    output_path = os.path.join(output_dir, "p2p_qber_baseline_only.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    logging.info(f"Point-to-point baseline QBER graph saved to {output_path}")

    plt.figure(figsize=(7, 5))
    rates_list = [baseline_rates[dist] for dist in distances_list]
    plt.plot(distances_list, rates_list, marker=hop_markers['baseline'], linestyle='-',
             linewidth=1.5, markersize=6, color=topology_colors['baseline'])
    plt.xlabel('Total End-to-End Distance (km)', fontsize=13)
    plt.ylabel('Key Exchange Rate (bits/second)', fontsize=13)
    plt.title('Point-to-Point Key Exchange Rate vs. Distance (Log Scale)\n(No Relay Nodes)', fontsize=16, fontweight='bold')
    plt.grid(True, which='both', axis='y')
    plt.xlim(10, 200)
    plt.xticks(np.arange(10, 201, 10), fontsize=9)
    plt.yticks(fontsize=9)
    plt.yscale('log')
    plt.tight_layout()
    plt.legend(fontsize=11, loc='upper right')
    output_path = os.path.join(output_dir, "p2p_baseline_only_log.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    logging.info(f"Point-to-point baseline log-scale graph saved to {output_path}")

def create_distance_comparison_plots(channel_length_km, topology_rates, topology_qbers, topology_distances,
                                    topology_total_distances, output_dir, topology_colors):
    """Create comparison plots for a fixed per-hop channel length."""
    distance_dir = os.path.join(output_dir, f"{channel_length_km}km")
    os.makedirs(distance_dir, exist_ok=True)
    bar_width = 0.13
    max_connections = max(len(distances) for distances in topology_distances.values())

    plt.figure(figsize=(8, 6))
    plt.bar([0], topology_rates["baseline"], width=bar_width,
            label="Point-to-Point (No Relays)", color=topology_colors["baseline"],
            hatch='//')
    topologies = [t for t in topology_rates.keys() if t != "baseline"]
    for i, topology in enumerate(topologies):
        x = np.arange(len(topology_distances[topology]))
        rates = topology_rates[topology]
        plt.bar(x + (i-1)*bar_width + bar_width/2, rates, width=bar_width,
                label=f"{topology.capitalize()}", color=topology_colors[topology])
    plt.xlabel('Connection Type', fontsize=10)
    plt.ylabel('Key Exchange Rate (bits/second)', fontsize=10)
    plt.title(f'Key Exchange Rate Comparison (Including Baseline)\nChannel Length: {channel_length_km} km', fontsize=13)
    most_detailed_topology = max(topology_distances.items(), key=lambda x: len(x[1]))[0]
    plt.xticks(np.arange(len(topology_distances[most_detailed_topology])),
               ["P2P (No Relays)"] + topology_distances[most_detailed_topology][1:])
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path = os.path.join(distance_dir, f"combined_key_rate_comparison_with_baseline.png")
    plt.savefig(output_path, dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.bar([0], topology_qbers["baseline"], width=bar_width,
            label="Point-to-Point (No Relays)", color=topology_colors["baseline"],
            hatch='//')
    for i, topology in enumerate(topologies):
        x = np.arange(len(topology_distances[topology]))
        qbers = topology_qbers[topology]
        plt.bar(x + (i-1)*bar_width + bar_width/2, qbers, width=bar_width,
                label=f"{topology.capitalize()}", color=topology_colors[topology])
    plt.xlabel('Connection Type', fontsize=10)
    plt.ylabel('Quantum Bit Error Rate (QBER)', fontsize=10)
    plt.title(f'QBER Comparison (Including Baseline)\nChannel Length: {channel_length_km} km', fontsize=13)
    plt.xticks(np.arange(len(topology_distances[most_detailed_topology])),
               ["P2P (No Relays)"] + topology_distances[most_detailed_topology][1:])
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path = os.path.join(distance_dir, f"combined_qber_comparison_with_baseline.png")
    plt.savefig(output_path, dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.bar([0], topology_total_distances["baseline"], width=bar_width,
            label="Point-to-Point (No Relays)", color=topology_colors["baseline"],
            hatch='//')
    for i, topology in enumerate(topologies):
        x = np.arange(len(topology_distances[topology]))
        total_dists = topology_total_distances[topology]
        plt.bar(x + (i-1)*bar_width + bar_width/2, total_dists, width=bar_width,
                label=f"{topology.capitalize()}", color=topology_colors[topology])
    plt.xlabel('Connection Type', fontsize=10)
    plt.ylabel('Total End-to-End Distance (km)', fontsize=10)
    plt.title(f'Total Distance Comparison\nChannel Length: {channel_length_km} km per hop', fontsize=13)
    plt.xticks(np.arange(len(topology_distances[most_detailed_topology])),
               ["P2P (No Relays)"] + topology_distances[most_detailed_topology][1:])
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path = os.path.join(distance_dir, f"total_distance_comparison.png")
    plt.savefig(output_path, dpi=300)
    plt.close()

def create_adjacent_nodes_comparison(all_results, topologies, channel_distances,
                                     output_dir, topology_colors, hop_markers):
    """Compare adjacent-node key rates vs total distance across topologies."""
    plt.figure(figsize=(8, 6))
    distances_list = sorted(list(d for d in all_results.keys() if "baseline" in all_results[d]))
    rates_list = [all_results[dist]["baseline"]["rates"][0] for dist in distances_list]
    plt.plot(distances_list, rates_list, marker=hop_markers['baseline'], linestyle='-',
             linewidth=1.5, markersize=6, label='Point-to-Point (No Relays)',
             color=topology_colors['baseline'])

    for topology in topologies:
        total_distances_list = []
        rates_list = []
        for dist in channel_distances:
            try:
                if dist > 150:
                    continue
                total_dist = all_results[dist][topology]["total_distances"][0]
                total_distances_list.append(total_dist)
                rates_list.append(all_results[dist][topology]["rates"][0])
            except (IndexError, KeyError):
                continue
        plt.plot(total_distances_list, rates_list, marker=hop_markers[0], linestyle='-',
                 linewidth=1.2, markersize=6, label=f'{topology.capitalize()} (Adjacent nodes)',
                 color=topology_colors[topology])

    plt.xlabel('Total End-to-End Distance (km)', fontsize=11)
    plt.ylabel('Key Exchange Rate (bits/second)', fontsize=11)
    plt.title('Key Rate vs. Total Distance\nAdjacent Nodes Comparison', fontsize=13)
    plt.xlim(0, 150)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    output_path = os.path.join(output_dir, "adjacent_nodes_total_distance_comparison.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    logging.info(f"Adjacent nodes comparison with total distance plot saved to {output_path}")

def create_topology_vs_baseline_plots(all_results, topology, channel_distances,
                                     output_dir, topology_colors, hop_markers, max_distance=150):
    """Compare a given topology vs baseline across hop counts and total distance."""
    plt.figure(figsize=(8, 6))
    distances_list = sorted(list(d for d in all_results.keys() if "baseline" in all_results[d] and d <= max_distance))
    rates_list = [all_results[dist]["baseline"]["rates"][0] for dist in distances_list]
    plt.plot(distances_list, rates_list, marker=hop_markers['baseline'], linestyle='-',
             linewidth=1.5, markersize=6, label='Point-to-Point (No Relays)',
             color=topology_colors['baseline'])

    first_dist = next(d for d in sorted(all_results.keys()) if topology in all_results[d])
    if first_dist in all_results and topology in all_results[first_dist]:
        connection_types = all_results[first_dist][topology]["distances"]
        if topology == "ring":
            hop_colors = ['#1f77b4', '#aec7e8', '#2ca02c', '#98df8a']
        elif topology == "star":
            hop_colors = ['#ff7f0e', '#ffbb78', '#d62728', '#ff9896']
        else:
            hop_colors = ['#2ca02c', '#98df8a', '#17becf', '#9467bd']

        for conn_idx, conn_type in enumerate(connection_types):
            total_distances = []
            rates_list = []
            if "relays" in conn_type:
                try:
                    hop_count = int(conn_type.split("(")[1].split(" ")[0])
                except:
                    hop_count = conn_idx
            else:
                hop_count = conn_idx

            for dist in channel_distances:
                try:
                    if dist > max_distance:
                        continue
                    if dist in all_results and topology in all_results[dist]:
                        if conn_idx < len(all_results[dist][topology]["total_distances"]):
                            total_dist = all_results[dist][topology]["total_distances"][conn_idx]
                            total_distances.append(total_dist)
                            rates_list.append(all_results[dist][topology]["rates"][conn_idx])
                except (IndexError, KeyError):
                    continue

            hop_color = hop_colors[conn_idx % len(hop_colors)]
            marker = hop_markers.get(hop_count, 'o')
            plt.plot(total_distances, rates_list, marker=marker, linestyle='-',
                     linewidth=1.2, markersize=6, label=f"{conn_type}",
                     color=hop_color)

    plt.xlabel('Total End-to-End Distance (km)', fontsize=11)
    plt.ylabel('Key Exchange Rate (bits/second)', fontsize=11)
    plt.title(f'{topology.capitalize()} Topology Key Rate vs Total Distance', fontsize=13)
    plt.xlim(0, max_distance)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    output_path = os.path.join(output_dir, f"{topology}_vs_baseline_total_distance_{max_distance}km.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    logging.info(f"{topology.capitalize()} vs baseline total distance comparison saved to {output_path}")

def visualize_hop_performance(hop_statistics, output_dir, topology_colors):
    """Show end-to-end advantage from trusted nodes across distances."""
    plt.figure(figsize=(8, 6))
    distance_groups = {}
    for stats in hop_statistics:
        total_dist = stats['total_distance']
        if total_dist not in distance_groups:
            distance_groups[total_dist] = []
        distance_groups[total_dist].append(stats)

    x_positions = []
    x_labels = []
    bar_width = 0.10
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for i, (distance, group) in enumerate(sorted(distance_groups.items())):
        x_pos = i * 1.0
        x_positions.append(x_pos)
        x_labels.append(f"{distance}km")
        direct_rate = np.exp(-0.2 * distance)
        plt.bar(x_pos - 0.3, direct_rate, width=bar_width, color='gray', label='Direct (theoretical)' if i == 0 else "")
        for j, stats in enumerate(group):
            hop_rates = stats['hop_rates']
            path = stats['path']
            n_hops = len(path) - 1
            end_to_end_rate = min(hop_rates)
            label = f"{n_hops} hop" + ("s" if n_hops > 1 else "")
            plt.bar(x_pos + j*bar_width, end_to_end_rate, width=bar_width,
                    color=colors[j % len(colors)],
                    label=label if i == 0 else "")

    plt.xlabel('Total End-to-End Distance (km)', fontsize=11)
    plt.ylabel('Key Exchange Rate (bits/second)', fontsize=11)
    plt.title('Trusted Node Advantage in Key Exchange Rate vs Distance', fontsize=13)
    plt.grid(True, axis='y')
    plt.xticks(x_positions, x_labels)
    plt.yscale('log')
    plt.legend(fontsize=10)
    plt.tight_layout()
    output_path = os.path.join(output_dir, "trusted_node_advantage.png")
    plt.savefig(output_path, dpi=300)
    plt.close()