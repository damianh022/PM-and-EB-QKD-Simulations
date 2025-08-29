import logging
import os

import matplotlib.pyplot as plt
import netsquid as ns
import numpy as np
from network import create_qkd_network
from protocols import ReceiverProtocol, TransmitterProtocol
from simulation import calculate_qber, run_simulation


def parameter_sweep_study(output_dir):
    """Sweep loss and detector efficiency and summarize performance."""
    loss_per_km_values = [0.1, 0.2, 0.3, 0.4]
    detector_efficiency_values = [0.7, 0.8, 0.9, 0.99]
    selected_distances = [20, 50, 100]
    simulation_duration = 500
    num_nodes = 7
    topologies = ["ring", "star", "bus"]

    parameter_results = {}
    param_sweep_dir = os.path.join(output_dir, "param_sweep")
    os.makedirs(param_sweep_dir, exist_ok=True)

    for loss_per_km in loss_per_km_values:
        for detector_efficiency in detector_efficiency_values:
            param_id = f"loss_{loss_per_km}_eff_{detector_efficiency}"
            param_dir = os.path.join(param_sweep_dir, param_id)
            os.makedirs(param_dir, exist_ok=True)
            logging.info(
                f"\n===== PARAMETER SET: loss={loss_per_km}/km, detector efficiency={detector_efficiency} ====="
            )

            p2p_results = {}
            topology_results = {t: {} for t in topologies}

            for distance_km in selected_distances:
                logging.info(f"Running P2P simulation at {distance_km}km")
                baseline_network, baseline_nodes = create_qkd_network(
                    2, distance_km, detector_efficiency, loss_per_km, "bus")

                ns.sim_reset()
                final_key = []
                sent_bits = []
                measured_bits = []
                sent_bases = []
                recv_bases = []
                tx_ids = []
                rx_ids = []

                protocols = [
                    TransmitterProtocol(baseline_nodes[0],
                                        final_key,
                                        sent_bits,
                                        1,
                                        sent_bases=sent_bases,
                                        pulse_ids=tx_ids),
                    ReceiverProtocol(baseline_nodes[1],
                                     final_key,
                                     sent_bits,
                                     measured_bits,
                                     0,
                                     detection_efficiency=detector_efficiency,
                                     recv_bases=recv_bases,
                                     recv_pulse_ids=rx_ids)
                ]
                for protocol in protocols:
                    protocol.start()
                ns.sim_run(duration=simulation_duration)

                key_rate = len(final_key) / simulation_duration
                qber = calculate_qber(sent_bits, measured_bits, sent_bases,
                                      recv_bases, tx_ids, rx_ids)
                p2p_results[distance_km] = {
                    "rate": key_rate,
                    "qber": qber,
                    "key_length": len(final_key)
                }
                logging.info(
                    f"P2P at {distance_km}km: key_rate={key_rate:.4f}, QBER={qber:.4f}"
                )

                for topology in topologies:
                    logging.info(
                        f"Running {topology} topology at {distance_km}km")
                    if topology == "star":
                        node_pairs = [(1, 0), (1, 2)]
                    elif topology == "bus":
                        node_pairs = [(0, 1), (0, 2)]
                    else:
                        node_pairs = [(0, 1), (0, 2)]

                    key_rates, _, _, qbers, _, _, total_dists = run_simulation(
                        num_nodes, distance_km, detector_efficiency,
                        loss_per_km, simulation_duration, topology, node_pairs)

                    topology_results[topology][distance_km] = {
                        "rates": key_rates,
                        "qbers": qbers,
                        "total_distances": total_dists
                    }

            parameter_results[param_id] = {
                "loss_per_km": loss_per_km,
                "detector_efficiency": detector_efficiency,
                "p2p": p2p_results,
                "topologies": topology_results
            }
            create_parameter_visualization(param_id,
                                           parameter_results[param_id],
                                           param_dir)

    create_comparative_parameter_plots(parameter_results, param_sweep_dir)
    return parameter_results


def create_parameter_visualization(param_id, results, output_dir):
    """Visualize key rate and QBER vs distance for a parameter set."""
    loss = results["loss_per_km"]
    efficiency = results["detector_efficiency"]
    p2p_results = results["p2p"]
    topology_results = results["topologies"]
    distances = sorted(p2p_results.keys())

    plt.figure(figsize=(12, 8))
    p2p_rates = [p2p_results[d]["rate"] for d in distances]
    plt.plot(distances,
             p2p_rates,
             marker='X',
             linestyle='-',
             linewidth=2.5,
             markersize=10,
             color='#d62728',
             label='Point-to-Point')
    topology_colors = {'ring': '#1f77b4', 'star': '#ff7f0e', 'bus': '#2ca02c'}
    for topology, color in topology_colors.items():
        rates = []
        topo_distances = []
        for d in distances:
            if d in topology_results[topology]:
                if topology_results[topology][d]["rates"]:
                    rates.append(topology_results[topology][d]["rates"][0])
                    topo_distances.append(d)
        if rates:
            plt.plot(topo_distances,
                     rates,
                     marker='o',
                     linestyle='-',
                     linewidth=2,
                     markersize=8,
                     color=color,
                     label=f'{topology.capitalize()} (Adjacent nodes)')
    plt.xlabel('Distance (km)', fontsize=14)
    plt.ylabel('Key Exchange Rate (bits/time unit)', fontsize=14)
    plt.title(
        f'Key Rate vs Distance\nLoss: {loss}/km, Detector Efficiency: {efficiency}',
        fontsize=16)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    output_path = os.path.join(output_dir, "key_rate_vs_distance.png")
    plt.savefig(output_path, dpi=300)
    plt.close()

    plt.figure(figsize=(12, 8))
    p2p_qbers = [p2p_results[d]["qber"] for d in distances]
    plt.plot(distances,
             p2p_qbers,
             marker='X',
             linestyle='-',
             linewidth=2.5,
             markersize=10,
             color='#d62728',
             label='Point-to-Point')
    for topology, color in topology_colors.items():
        qbers = []
        topo_distances = []
        for d in distances:
            if d in topology_results[topology]:
                if topology_results[topology][d]["qbers"]:
                    qbers.append(topology_results[topology][d]["qbers"][0])
                    topo_distances.append(d)
        if qbers:
            plt.plot(topo_distances,
                     qbers,
                     marker='o',
                     linestyle='-',
                     linewidth=2,
                     markersize=8,
                     color=color,
                     label=f'{topology.capitalize()} (Adjacent nodes)')
    plt.xlabel('Distance (km)', fontsize=14)
    plt.ylabel('Quantum Bit Error Rate (QBER)', fontsize=14)
    plt.title(
        f'QBER vs Distance\nLoss: {loss}/km, Detector Efficiency: {efficiency}',
        fontsize=16)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    output_path = os.path.join(output_dir, "qber_vs_distance.png")
    plt.savefig(output_path, dpi=300)
    plt.close()


def create_comparative_parameter_plots(all_param_results, output_dir):
    """Compare P2P performance across parameter combinations."""
    loss_values = sorted(
        set(r["loss_per_km"] for r in all_param_results.values()))
    efficiency_values = sorted(
        set(r["detector_efficiency"] for r in all_param_results.values()))
    reference_distance = 50

    p2p_rate_data = np.zeros((len(loss_values), len(efficiency_values)))
    p2p_qber_data = np.zeros((len(loss_values), len(efficiency_values)))

    for i, loss in enumerate(loss_values):
        for j, eff in enumerate(efficiency_values):
            param_id = f"loss_{loss}_eff_{eff}"
            if param_id in all_param_results:
                result = all_param_results[param_id]
                if reference_distance in result["p2p"]:
                    p2p_rate_data[i,
                                  j] = result["p2p"][reference_distance]["rate"]
                    p2p_qber_data[i,
                                  j] = result["p2p"][reference_distance]["qber"]

    plt.figure(figsize=(10, 8))
    plt.imshow(p2p_rate_data,
               cmap='viridis',
               aspect='auto',
               extent=[
                   min(efficiency_values) - 0.05,
                   max(efficiency_values) + 0.05,
                   max(loss_values) + 0.05,
                   min(loss_values) - 0.05
               ],
               interpolation='nearest')
    plt.colorbar(label='Key Rate (bits/time unit)')
    plt.xlabel('Detector Efficiency', fontsize=14)
    plt.ylabel('Loss per km', fontsize=14)
    plt.title(f'P2P Key Rate at {reference_distance}km', fontsize=16)
    for i, loss in enumerate(loss_values):
        for j, eff in enumerate(efficiency_values):
            plt.text(eff,
                     loss,
                     f'{p2p_rate_data[i, j]:.2f}',
                     ha='center',
                     va='center',
                     color='white',
                     fontweight='bold')
    plt.tight_layout()
    output_path = os.path.join(output_dir, "p2p_key_rate_heatmap.png")
    plt.savefig(output_path, dpi=300)
    plt.close()

    plt.figure(figsize=(10, 8))
    plt.imshow(p2p_qber_data,
               cmap='plasma',
               aspect='auto',
               extent=[
                   min(efficiency_values) - 0.05,
                   max(efficiency_values) + 0.05,
                   max(loss_values) + 0.05,
                   min(loss_values) - 0.05
               ],
               interpolation='nearest')
    plt.colorbar(label='QBER')
    plt.xlabel('Detector Efficiency', fontsize=14)
    plt.ylabel('Loss per km', fontsize=14)
    plt.title(f'P2P QBER at {reference_distance}km', fontsize=16)
    for i, loss in enumerate(loss_values):
        for j, eff in enumerate(efficiency_values):
            plt.text(eff,
                     loss,
                     f'{p2p_qber_data[i, j]:.3f}',
                     ha='center',
                     va='center',
                     color='white',
                     fontweight='bold')
    plt.tight_layout()
    output_path = os.path.join(output_dir, "p2p_qber_heatmap.png")
    plt.savefig(output_path, dpi=300)
    plt.close()

    topologies = ["ring", "star", "bus"]
    for topology in topologies:
        topo_rate_data = np.zeros((len(loss_values), len(efficiency_values)))
        for i, loss in enumerate(loss_values):
            for j, eff in enumerate(efficiency_values):
                param_id = f"loss_{loss}_eff_{eff}"
                if param_id in all_param_results:
                    result = all_param_results[param_id]
                    if reference_distance in result["topologies"][topology]:
                        if result["topologies"][topology][reference_distance][
                                "rates"]:
                            topo_rate_data[i, j] = result["topologies"][
                                topology][reference_distance]["rates"][0]
        plt.figure(figsize=(10, 8))
        plt.imshow(topo_rate_data,
                   cmap='viridis',
                   aspect='auto',
                   extent=[
                       min(efficiency_values) - 0.05,
                       max(efficiency_values) + 0.05,
                       max(loss_values) + 0.05,
                       min(loss_values) - 0.05
                   ],
                   interpolation='nearest')
        plt.colorbar(label='Key Rate (bits/time unit)')
        plt.xlabel('Detector Efficiency', fontsize=14)
        plt.ylabel('Loss per km', fontsize=14)
        plt.title(
            f'{topology.capitalize()} Topology Key Rate at {reference_distance}km',
            fontsize=16)
        for i, loss in enumerate(loss_values):
            for j, eff in enumerate(efficiency_values):
                plt.text(eff,
                         loss,
                         f'{topo_rate_data[i, j]:.2f}',
                         ha='center',
                         va='center',
                         color='white',
                         fontweight='bold')
        plt.tight_layout()
        output_path = os.path.join(output_dir,
                                   f"{topology}_key_rate_heatmap.png")
        plt.savefig(output_path, dpi=300)
        plt.close()

    plt.figure(figsize=(14, 10))
    ref_loss = 0.2
    ref_eff = 0.9

    plt.subplot(1, 2, 1)
    loss_sensitivity = []
    for loss in loss_values:
        param_id = f"loss_{loss}_eff_{ref_eff}"
        if param_id in all_param_results:
            if reference_distance in all_param_results[param_id]["p2p"]:
                loss_sensitivity.append(all_param_results[param_id]["p2p"]
                                        [reference_distance]["rate"])
    plt.plot(loss_values, loss_sensitivity, 'o-', linewidth=2)
    plt.xlabel('Loss per km', fontsize=12)
    plt.ylabel('Key Rate (bits/time unit)', fontsize=12)
    plt.title(f'Sensitivity to Loss\n(Efficiency={ref_eff})', fontsize=14)
    plt.grid(True)

    plt.subplot(1, 2, 2)
    eff_sensitivity = []
    for eff in efficiency_values:
        param_id = f"loss_{ref_loss}_eff_{eff}"
        if param_id in all_param_results:
            if reference_distance in all_param_results[param_id]["p2p"]:
                eff_sensitivity.append(all_param_results[param_id]["p2p"]
                                       [reference_distance]["rate"])
    plt.plot(efficiency_values, eff_sensitivity, 'o-', linewidth=2)
    plt.xlabel('Detector Efficiency', fontsize=12)
    plt.ylabel('Key Rate (bits/time unit)', fontsize=12)
    plt.title(f'Sensitivity to Detector Efficiency\n(Loss={ref_loss}/km)',
              fontsize=14)
    plt.grid(True)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "parameter_sensitivity.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
