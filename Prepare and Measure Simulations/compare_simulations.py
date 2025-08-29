import netsquid as ns
import logging
import os
import matplotlib.pyplot as plt
import numpy as np

from simulation import calculate_qber, TIME_UNIT_SECONDS
from network import create_qkd_network
from protocols import TransmitterProtocol, ReceiverProtocol

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def run_comparison(channel_length_km, detector_efficiency=1.0, loss_per_km=0.2,
                  simulation_duration=300, num_runs=5):
    """Compute single-photon results at a fixed distance, averaged over multiple runs."""
    comparison_results = {
        'perfect': {'rate': 0, 'qber': 0, 'key_length': 0, 'rates': [], 'qbers': []},
    }

    for run in range(num_runs):
        logging.info(f"Run {run+1}/{num_runs} for distance {channel_length_km}km")
        network, nodes = create_qkd_network(2, channel_length_km, detector_efficiency, loss_per_km, "bus")

        ns.sim_reset()
        final_key = []
        sent_bits = []
        measured_bits = []
        sent_bases = []
        recv_bases = []
        tx_ids = []
        rx_ids = []
        protocols = [
            TransmitterProtocol(nodes[0], final_key, sent_bits, 1,
                                sent_bases=sent_bases, pulse_ids=tx_ids),
            ReceiverProtocol(nodes[1], final_key, sent_bits, measured_bits, 0,
                             detection_efficiency=detector_efficiency,
                             recv_bases=recv_bases, recv_pulse_ids=rx_ids)
        ]
        for protocol in protocols:
            protocol.start()
        ns.sim_run(duration=simulation_duration)
        key_rate = len(final_key) / (simulation_duration * TIME_UNIT_SECONDS)
        qber = calculate_qber(sent_bits, measured_bits, sent_bases, recv_bases, tx_ids, rx_ids)
        comparison_results['perfect']['rates'].append(key_rate)
        comparison_results['perfect']['qbers'].append(qber)

    for source_type in comparison_results:
        rates = comparison_results[source_type]['rates']
        qbers = comparison_results[source_type]['qbers']
        if rates:
            comparison_results[source_type]['rate'] = sum(rates) / len(rates)
            comparison_results[source_type]['rate_std'] = np.std(rates) if len(rates) > 1 else 0
            comparison_results[source_type]['qber'] = sum(qbers) / len(qbers)
            comparison_results[source_type]['qber_std'] = np.std(qbers) if len(qbers) > 1 else 0
        logging.info(f"{source_type}: rate={comparison_results[source_type]['rate']:.4f}±{comparison_results[source_type].get('rate_std', 0):.4f}, QBER={comparison_results[source_type]['qber']:.4f}±{comparison_results[source_type].get('qber_std', 0):.4f}")

    return comparison_results

def run_distance_sweep(output_dir, detector_efficiency=1.0, loss_per_km=0.2,
                      simulation_duration=300, num_runs=3):
    """Run single-photon comparisons over multiple distances and save plots."""
    distances = [*range(5, 41, 5), *range(45, 101, 10), *range(110, 201, 20)]
    results = {}
    for dist in distances:
        logging.info(f"\n===== Running comparison at {dist}km =====")
        results[dist] = run_comparison(
            dist, detector_efficiency, loss_per_km,
            simulation_duration, num_runs
        )
    create_comparison_plots(results, distances, output_dir)
    return results

def create_comparison_plots(results, distances, output_dir):
    """Plot key rate and QBER vs distance for single-photon source."""
    comparison_dir = os.path.join(output_dir, "single_photon_comparison")
    os.makedirs(comparison_dir, exist_ok=True)

    plt.figure(figsize=(12, 8))
    perfect_rates = [results[d]['perfect']['rate'] for d in distances]
    plt.plot(distances, perfect_rates, marker='o', linestyle='-',
             linewidth=2.5, markersize=10, color='blue',
             label='Single-Photon Source')
    plt.xlabel('Distance (km)', fontsize=14)
    plt.ylabel('Key Exchange Rate (bits/second)', fontsize=14)
    plt.title('Single-Photon Key Rate vs Distance', fontsize=16)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    output_path = os.path.join(comparison_dir, "single_photon_key_rate.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    logging.info(f"Key rate plot saved to {output_path}")

    plt.figure(figsize=(12, 8))
    perfect_qbers = [results[d]['perfect']['qber'] for d in distances]
    plt.plot(distances, perfect_qbers, marker='o', linestyle='-',
             linewidth=2.5, markersize=10, color='blue',
             label='Single-Photon Source')
    plt.xlabel('Distance (km)', fontsize=14)
    plt.ylabel('Quantum Bit Error Rate (QBER)', fontsize=14)
    plt.title('Single-Photon QBER vs Distance', fontsize=16)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    output_path = os.path.join(comparison_dir, "single_photon_qber.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    logging.info(f"QBER plot saved to {output_path}")

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    COMPARE_OUTPUT_DIR = os.path.join(BASE_DIR, "PM QKD Results")
    os.makedirs(COMPARE_OUTPUT_DIR, exist_ok=True)
    results = run_distance_sweep(
        COMPARE_OUTPUT_DIR
    )
    logging.info("Comparison complete")