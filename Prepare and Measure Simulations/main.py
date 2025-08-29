import netsquid as ns
import logging
import os
import numpy as np
from protocols import TransmitterProtocol, ReceiverProtocol
from network import create_qkd_network, calculate_path
from simulation import calculate_qber, run_simulation, TIME_UNIT_SECONDS
from visualization import visualize_network, create_total_distance_plots
from analysis import (create_baseline_plots, create_distance_comparison_plots,
                    create_adjacent_nodes_comparison, create_topology_vs_baseline_plots)
from Parameter_Sweep import parameter_sweep_study

def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DIR = os.path.join(BASE_DIR, "PM QKD Results")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logging.info(f"Results will be saved to {OUTPUT_DIR}")

    channel_distances = (
        list(range(5, 101, 5)) +
        list(range(110, 151, 10)) +
        list(range(160, 251, 10))
    )
    detector_efficiency = 0.9
    loss_per_km = 0.2
    simulation_duration = 1000

    num_nodes = 8
    topologies = ["ring", "star", "bus"]

    topology_colors = {
        'ring': '#1f77b4',
        'star': '#ff7f0e',
        'bus': '#2ca02c',
        'baseline': '#d62728'
    }

    hop_markers = {
        0: 'o',
        1: 's',
        2: '^',
        3: 'D',
        4: 'p',
        5: 'h',
        'baseline': 'X'
    }

    all_results = {}
    baseline_rates = {}
    baseline_qbers = {}
    baseline_total_distances = {}

    for channel_length_km in channel_distances:
        logging.info(f"\n===== Running baseline perfect photons point-to-point simulation with {channel_length_km}km channel =====")
        baseline_network, baseline_nodes = create_qkd_network(2, channel_length_km, detector_efficiency, loss_per_km, "bus")

        ns.sim_reset()
        final_key = []
        sent_bits = []
        measured_bits = []
        sent_bases = []
        recv_bases = []
        tx_ids = []
        rx_ids = []

        protocols = [
            TransmitterProtocol(baseline_nodes[0], final_key, sent_bits, 1,
                                sent_bases=sent_bases, pulse_ids=tx_ids),
            ReceiverProtocol(baseline_nodes[1], final_key, sent_bits, measured_bits, 0,
                             detection_efficiency=detector_efficiency,
                             recv_bases=recv_bases, recv_pulse_ids=rx_ids)
        ]

        for protocol in protocols:
            protocol.start()
        ns.sim_run(duration=simulation_duration)

        key_exchange_rate = len(final_key) / (simulation_duration * TIME_UNIT_SECONDS)
        qber = calculate_qber(sent_bits, measured_bits, sent_bases, recv_bases, tx_ids, rx_ids)

        baseline_rates[channel_length_km] = key_exchange_rate
        baseline_qbers[channel_length_km] = qber
        baseline_total_distances[channel_length_km] = channel_length_km
        logging.info(f"Baseline P2P simulation completed. Total distance: {channel_length_km}km, Key rate: {key_exchange_rate:.4f}, QBER: {qber:.4f}")

    create_baseline_plots(
        sorted(list(baseline_rates.keys())),
        baseline_rates,
        baseline_qbers,
        OUTPUT_DIR,
        topology_colors,
        hop_markers
    )

    topology_node_pairs = {}

    for topology in topologies:
        logging.info(f"\n===== Visualizing {topology} topology =====")
        vis_network, _ = create_qkd_network(num_nodes, 20, detector_efficiency, loss_per_km, topology)
        visualize_network(vis_network, topology=topology, output_dir=OUTPUT_DIR)

    for channel_length_km in channel_distances:
        if channel_length_km > 250:
            continue
        logging.info(f"\n===== Testing with channel distance: {channel_length_km} km =====")

        topology_rates = {}
        topology_qbers = {}
        topology_distances = {}
        topology_total_distances = {}
        topology_rates["baseline"] = [baseline_rates[channel_length_km]]
        topology_qbers["baseline"] = [baseline_qbers[channel_length_km]]
        topology_distances["baseline"] = ["Direct P2P"]
        topology_total_distances["baseline"] = [channel_length_km]

        for topology in topologies:
            logging.info(f"\n===== Running {topology} topology with {channel_length_km}km channels =====")
            if topology == "star":
                node_pairs = [(1, 0), (1, 2)]
                distances = ['Adjacent nodes (0 relays)', '1 relay node']
            elif topology == "bus":
                node_pairs = [(0, j) for j in range(1, num_nodes)]
                distances = []
                for j in range(1, num_nodes):
                    hops = j
                    relays = hops - 1
                    if relays == 0:
                        distances.append('Adjacent nodes (0 relays)')
                    elif relays == 1:
                        distances.append('1 relay node')
                    else:
                        distances.append(f'{relays} relay nodes')
            else:
                max_hops = num_nodes // 2
                node_pairs = [(0, k) for k in range(1, max_hops + 1)]
                distances = []
                for k in range(1, max_hops + 1):
                    hops = k
                    relays = hops - 1
                    if relays == 0:
                        distances.append('Adjacent nodes (0 relays)')
                    elif relays == 1:
                        distances.append('1 relay node')
                    else:
                        distances.append(f'{relays} relay nodes')

            topology_node_pairs[topology] = node_pairs

            # Always run perfect simulation
            key_exchange_rates, network, keys, qbers, sent_bits, measured_bits, total_distances, hop_statistics = run_simulation(
                num_nodes=num_nodes,
                channel_length_km=channel_length_km,
                detector_efficiency=detector_efficiency,
                loss_per_km=loss_per_km,
                simulation_duration=simulation_duration,
                topology=topology,
                node_pairs=node_pairs
            )

            if channel_length_km not in all_results:
                all_results[channel_length_km] = {}
                all_results[channel_length_km]["baseline"] = {
                    "rates": [baseline_rates[channel_length_km]],
                    "qbers": [baseline_qbers[channel_length_km]],
                    "distances": ["Direct P2P (No Relays)"],
                    "total_distances": [channel_length_km]
                }
            all_results[channel_length_km][topology] = {
                "rates": key_exchange_rates,
                "qbers": qbers,
                "keys": keys,
                "distances": distances,
                "total_distances": total_distances
            }

            topology_rates[topology] = key_exchange_rates
            topology_qbers[topology] = qbers
            topology_distances[topology] = distances
            topology_total_distances[topology] = total_distances

            for i, (node_pair, key, qber, total_dist) in enumerate(zip(node_pairs, keys, qbers, total_distances)):
                distance = distances[i]
                key_preview = key[:10] if key else []
                logging.info(f"[{channel_length_km}km per hop, {total_dist}km total] Final Key for nodes {node_pair} ({distance}): {key_preview}... (length: {len(key)}, QBER: {qber:.4f})")

        create_distance_comparison_plots(
            channel_length_km,
            topology_rates,
            topology_qbers,
            topology_distances,
            topology_total_distances,
            OUTPUT_DIR,
            topology_colors
        )

    create_adjacent_nodes_comparison(
        all_results,
        topologies,
        channel_distances,
        OUTPUT_DIR,
        topology_colors,
        hop_markers
    )

    for topology in topologies:
        create_topology_vs_baseline_plots(
            all_results,
            topology,
            [d for d in channel_distances if d <= 150],
            OUTPUT_DIR,
            topology_colors,
            hop_markers,
            max_distance=150
        )
        create_topology_vs_baseline_plots(
            all_results,
            topology,
            [d for d in channel_distances if d <= 250],
            OUTPUT_DIR,
            topology_colors,
            hop_markers,
            max_distance=250
        )
        for max_d in [350, 450, 550, 650]:
            create_topology_vs_baseline_plots(
                all_results,
                topology,
                [d for d in channel_distances if d <= max_d],
                OUTPUT_DIR,
                topology_colors,
                hop_markers,
                max_distance=max_d
            )

    create_total_distance_plots(
        all_results,
        topologies,
        OUTPUT_DIR,
        topology_node_pairs,
        num_nodes,
        max_distance=150,
        topology_colors=topology_colors
    )

if __name__ == "__main__":
    main()