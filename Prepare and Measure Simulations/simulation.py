import netsquid as ns
import logging
import time
from protocols import TransmitterProtocol, ReceiverProtocol, RelayProtocol
from network import create_qkd_network, calculate_path
import numpy as np

TIME_UNIT_SECONDS = 1.0

def units_to_seconds(t_units):
    return t_units * TIME_UNIT_SECONDS

def rate_bps(n_bits, duration_units):
    seconds = units_to_seconds(duration_units)
    return (n_bits / seconds) if seconds > 0 else 0.0

def calculate_qber(sent_bits, measured_bits, sent_bases=None, recv_bases=None, sent_ids=None, recv_ids=None):
    """Compute QBER by aligning pulse IDs and sifting by basis; falls back to length alignment."""
    if sent_bases is not None and recv_bases is not None and sent_ids is not None and recv_ids is not None:
        tx = {pid: (bit, bas) for pid, bit, bas in zip(sent_ids, sent_bits, sent_bases)}
        rx = {pid: (meas, bas) for pid, meas, bas in zip(recv_ids, measured_bits, recv_bases)}
        common = set(tx.keys()).intersection(rx.keys())
        kept = 0
        errors = 0
        for pid in common:
            sbit, sbas = tx[pid]
            mbit, rbas = rx[pid]
            if sbas == rbas:
                kept += 1
                if sbit != mbit:
                    errors += 1
        return (errors / kept) if kept > 0 else 0.0

    if len(sent_bits) == 0 or len(measured_bits) == 0:
        return 0.0
    min_length = min(len(sent_bits), len(measured_bits))
    errors = sum(1 for s, m in zip(sent_bits[:min_length], measured_bits[:min_length]) if s != m)
    return errors / min_length

def _h2(p):
    p = min(max(p, 1e-12), 1 - 1e-12)
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)

def estimate_secure_key_len(n_sifted, qber, f_ec=1.16):
    """Asymptotic BB84 secret fraction estimate for EC+PA."""
    secret_fraction = 1.0 - f_ec * _h2(qber) - _h2(qber)
    if secret_fraction < 0:
        return 0
    return int(np.floor(n_sifted * secret_fraction))

def run_simulation(num_nodes, channel_length_km, detector_efficiency, loss_per_km, simulation_duration=1000, topology=None, node_pairs=None):
    """Perfect-photon per-hop QKD with EC/PA, XOR-forwarded end-to-end key rate."""
    network, nodes = create_qkd_network(num_nodes, channel_length_km, detector_efficiency, loss_per_km, topology)
    key_exchange_rates = []
    keys = []
    qbers = []
    sent_bits_list = []
    measured_bits_list = []
    total_distances = []
    hop_statistics = []
    target_key_length = 64

    for (node1, node2) in node_pairs:
        ns.sim_reset()
        final_key = []
        sent_bits = []
        measured_bits = []
        sent_bases = []
        recv_bases = []
        tx_ids = []
        rx_ids = []

        path = calculate_path(node1, node2, num_nodes, topology)
        total_distance = channel_length_km * (len(path) - 1)
        total_distances.append(total_distance)
        logging.info(f"Path Node_{node1}->{node2}, hops={len(path)-1}, total_distance={total_distance}km")

        if len(path) == 2:
            protocols = [
                TransmitterProtocol(nodes[node1], final_key, sent_bits, node2,
                                    target_length=target_key_length,
                                    sent_bases=sent_bases, pulse_ids=tx_ids),
                ReceiverProtocol(nodes[node2], final_key, sent_bits, measured_bits, node1,
                                 target_length=target_key_length,
                                 detection_efficiency=detector_efficiency,
                                 recv_bases=recv_bases, recv_pulse_ids=rx_ids)
            ]
            for protocol in protocols:
                protocol.start()
            elapsed_time = 0
            time_increment = 10
            max_duration = max(simulation_duration, 2000)
            while elapsed_time < max_duration and len(final_key) < target_key_length:
                ns.sim_run(duration=time_increment)
                elapsed_time += time_increment
            qber = calculate_qber(sent_bits, measured_bits, sent_bases, recv_bases, tx_ids, rx_ids)
            n_sifted = len(final_key)
            secure_len = estimate_secure_key_len(n_sifted, qber)
            if secure_len < n_sifted:
                final_key = final_key[:secure_len]
            key_exchange_rate = (secure_len / units_to_seconds(elapsed_time if elapsed_time > 0 else simulation_duration))
        else:
            logging.info(f"Trusted-node multi-hop over {len(path)-1} hops with target={target_key_length}")
            num_hops = len(path) - 1
            hop_keys = [[] for _ in range(num_hops)]
            hop_sent_bits = [[] for _ in range(num_hops)]
            hop_measured_bits = [[] for _ in range(num_hops)]
            hop_sent_bases = [[] for _ in range(num_hops)]
            hop_recv_bases = [[] for _ in range(num_hops)]
            hop_tx_ids = [[] for _ in range(num_hops)]
            hop_rx_ids = [[] for _ in range(num_hops)]
            hop_done_time = [None for _ in range(num_hops)]
            hop_protocols = []
            for i in range(num_hops):
                src = path[i]
                dst = path[i+1]
                tx = TransmitterProtocol(
                    nodes[src], hop_keys[i], hop_sent_bits[i], dst,
                    target_length=target_key_length,
                    sent_bases=hop_sent_bases[i], pulse_ids=hop_tx_ids[i]
                )
                rx = ReceiverProtocol(
                    nodes[dst], hop_keys[i], hop_sent_bits[i], hop_measured_bits[i], src,
                    target_length=target_key_length,
                    detection_efficiency=detector_efficiency,
                    recv_bases=hop_recv_bases[i], recv_pulse_ids=hop_rx_ids[i]
                )
                hop_protocols.extend([tx, rx])
            for p in hop_protocols:
                p.start()
            elapsed_time = 0
            time_increment = 10
            max_duration = max(simulation_duration, 4000)
            while elapsed_time < max_duration:
                ns.sim_run(duration=time_increment)
                elapsed_time += time_increment
                for i in range(num_hops):
                    if hop_done_time[i] is None and len(hop_keys[i]) >= target_key_length:
                        hop_done_time[i] = elapsed_time
                if all(t is not None for t in hop_done_time):
                    break
            hop_qbers = []
            hop_secure_lengths = []
            for i in range(num_hops):
                q_i = calculate_qber(hop_sent_bits[i], hop_measured_bits[i],
                                     hop_sent_bases[i], hop_recv_bases[i],
                                     hop_tx_ids[i], hop_rx_ids[i])
                hop_qbers.append(q_i)
                hop_secure_lengths.append(estimate_secure_key_len(target_key_length, q_i))
            min_secure_len = int(min(hop_secure_lengths)) if hop_secure_lengths else 0
            slowest_time = max(t for t in hop_done_time if t is not None) if any(hop_done_time) else 0
            key_exchange_rate = (min_secure_len / units_to_seconds(slowest_time)) if slowest_time > 0 else 0.0
            qber = max(hop_qbers) if hop_qbers else 0.0
            final_key = hop_keys[0][:min_secure_len] if min_secure_len > 0 else []
            hop_statistics.append({
                'path': path.copy(),
                'distances': [channel_length_km] * num_hops,
                'hop_rates': [
                    (estimate_secure_key_len(target_key_length, hop_qbers[i]) /
                     units_to_seconds(hop_done_time[i])) if hop_done_time[i] else 0.0
                    for i in range(num_hops)
                ],
                'key_lengths': [len(hk) for hk in hop_keys],
                'secure_lengths': hop_secure_lengths,
                'total_distance': total_distance,
                'final_key_length': len(final_key),
                'time_to_generate': slowest_time
            })

        key_exchange_rates.append(key_exchange_rate)
        keys.append(final_key)
        qbers.append(qber)
        sent_bits_list.append(sent_bits)
        measured_bits_list.append(measured_bits)
        logging.info(f"End-to-end secure key rate: {key_exchange_rate:.6f} bits/s, QBER: {qber:.4f}, key_len={len(final_key)}")

    return key_exchange_rates, network, keys, qbers, sent_bits_list, measured_bits_list, total_distances, hop_statistics