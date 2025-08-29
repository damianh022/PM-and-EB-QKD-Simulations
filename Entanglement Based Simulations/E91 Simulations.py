import netsquid as ns
from netsquid.components import QuantumChannel, ClassicalChannel, QuantumDetector
from netsquid.nodes import Node, Network, DirectConnection
from netsquid.protocols import NodeProtocol
import random
from netsquid.qubits import qubitapi as qapi
from netsquid.qubits.operators import Z, X, H
from netsquid.components.models.qerrormodels import FibreLossModel, DepolarNoiseModel
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D
import logging
import numpy as np
import os
from collections import deque
import argparse

import matplotlib
matplotlib.use('Agg')

# Create directory for results
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "EB QKD Results"))
OUTPUT_DIR = os.environ.get("EB_QKD_OUTPUT_DIR", DEFAULT_OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def upper_limit(max_value):
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

# Entanglement source
class EntanglementSourceProtocol(NodeProtocol):
    """
    Can operate in rate-driven mode (default) or await downstream READY signals per connection.
    """
    def __init__(self, node, connections, pair_generation_rate=1.0, downstream_ready_ports=None, **kwargs):
        super().__init__(node)
        self.connections = connections
        self.pair_generation_rate = pair_generation_rate
        self.distributed_pairs = 0
        self.downstream_ready_ports = downstream_ready_ports or {}  # {node_id: port_name}
        logging.info(f"Initialized EntanglementSourceProtocol at {node.name}")

    def run(self):
        while True:
            for node_id1, node_id2 in self.connections:
                # READY-gated mode: wait for both nodes if ports provided
                if self.downstream_ready_ports:
                    if node_id1 in self.downstream_ready_ports:
                        yield self.await_port_input(self.node.ports[self.downstream_ready_ports[node_id1]])
                    if node_id2 in self.downstream_ready_ports:
                        yield self.await_port_input(self.node.ports[self.downstream_ready_ports[node_id2]])
                # Emit entangled pair
                qubits = qapi.create_qubits(2)
                qapi.operate(qubits[0], H)
                qapi.operate([qubits[0], qubits[1]], ns.CX)
                self.node.ports[f'qout_{node_id1}'].tx_output(qubits[0])
                self.node.ports[f'qout_{node_id2}'].tx_output(qubits[1])
                self.distributed_pairs += 1
                logging.debug(f"Source distributed pair #{self.distributed_pairs} to nodes {node_id1} and {node_id2}")
                
                if not self.downstream_ready_ports:
                    yield self.await_timer(1 / self.pair_generation_rate)
            
            if self.downstream_ready_ports:
                yield self.await_timer(0.001)


class NodeMeasurementProtocol(NodeProtocol):
    """
    Measurement protocol:
    - P2P/standard mode: direct sifting with neighbor.
    - Swapping mode: consumes SWAP tokens and PAIRs end-to-end.
    """
    def __init__(self, node, node_id, shared_keys, connected_nodes,
                 enable_swapping=False, upstream_ready_port=None, expected_depth=None):
        super().__init__(node)
        self.node_id = node_id
        self.shared_keys = shared_keys
        self.connected_nodes = connected_nodes
        self.enable_swapping = enable_swapping
        self.measurements = {}
        self.received_bases = {}
        self.original_bits = {}
        self.key_bits = {}
        self.node_to_source_map = {}
        self.pending_measurements = {}
        self.waiting_swaps = {}
        self.swap_records = {}
        self.upstream_ready_port = upstream_ready_port
        self.expected_depth = expected_depth
        self.partner_pending_tokens = set()
        self.key_log_every = 200
        self._key_counter = 0

        for port_name in node.ports:
            if port_name.startswith("qin_Source_"):
                source_name = port_name[4:]
                self.measurements[source_name] = []
                self.received_bases[source_name] = {}
                self.original_bits[source_name] = {}
                self.key_bits[source_name] = {}
                self.pending_measurements[source_name] = deque()
                self.waiting_swaps[source_name] = deque()
                parts = source_name.split('_')
                if len(parts) >= 3 and parts[1].isdigit() and parts[2].isdigit():
                    n1, n2 = int(parts[1]), int(parts[2])
                    other_node = n2 if self.node_id == n1 else n1
                    self.node_to_source_map[other_node] = source_name
        for other_node in connected_nodes:
            self.original_bits.setdefault(other_node, {})
            self.key_bits.setdefault(other_node, {})

        logging.info(f"Initialized NodeMeasurementProtocol for {node.name} (swapping={self.enable_swapping}, expected_depth={self.expected_depth})")

    def run(self):
        def create_qubit_handler(src_name):
            def handler(message):
                self._handle_qubit_with_source(message, src_name)
                return None
            return handler

        def create_classical_handler(node_id):
            def handler(message):
                self._handle_classical_with_node(message, node_id)
                return None
            return handler

        for port_name in self.node.ports:
            if port_name.startswith("qin_Source_"):
                source_name = port_name[4:]
                self.node.ports[port_name].bind_input_handler(create_qubit_handler(source_name))

        already = set()
        for other_node in self.connected_nodes:
            port_name = f"cin_{other_node}"
            if port_name in self.node.ports:
                self.node.ports[port_name].bind_input_handler(create_classical_handler(other_node))
                already.add(port_name)
        for port_name in self.node.ports:
            if port_name.startswith("cin_") and port_name not in already:
                suffix = port_name[4:]
                if suffix.isdigit():
                    self.node.ports[port_name].bind_input_handler(create_classical_handler(int(suffix)))

        # Kickstart READY-gated sources
        if self.enable_swapping and self.upstream_ready_port and self.upstream_ready_port in self.node.ports:
            try:
                self.node.ports[self.upstream_ready_port].tx_output("READY")
            except Exception:
                pass

        while True:
            yield self.await_timer(1000)

    def _handle_qubit_with_source(self, message, source_name):
        try:
            if message is None or not hasattr(message, 'items') or not message.items:
                return
            qubit = message.items[0]
            basis = random.choice(['Z', 'X'])
            meas, _ = qapi.measure(qubit, Z if basis == 'Z' else X)
            index = len(self.measurements.get(source_name, []))
            self.measurements.setdefault(source_name, []).append((basis, meas, index))

            if self.enable_swapping:
                self.pending_measurements[source_name].append((index, basis, meas))
                self._try_process_waiting_swap(source_name)
                if self.upstream_ready_port and self.upstream_ready_port in self.node.ports:
                    self.node.ports[self.upstream_ready_port].tx_output("READY")
                return

            # P2P direct sifting
            for other_node, src_name in self.node_to_source_map.items():
                if src_name == source_name:
                    port_name = f'cout_{other_node}'
                    if port_name in self.node.ports:
                        self.node.ports[port_name].tx_output((self.node_id, source_name, index, basis, meas))
                        logging.debug(f"{self.node.name} measured {meas} in {basis} for {source_name}")
                    break
        except Exception as e:
            logging.warning(f"Error handling qubit in {self.node.name}: {e}")

    def _try_process_waiting_swap(self, source_name, force_token_id=None):
        if not self.pending_measurements.get(source_name) or not self.waiting_swaps.get(source_name):
            return
        token = None
        if force_token_id is not None:
            for i, t in enumerate(self.waiting_swaps[source_name]):
                if t.get('token_id') == force_token_id:
                    token = self.waiting_swaps[source_name][i]
                    del self.waiting_swaps[source_name][i]
                    break
            if token is None:
                return
        else:
            token = self.waiting_swaps[source_name][0]
            for i, t in enumerate(self.waiting_swaps[source_name]):
                tid = t.get('token_id')
                rec = self.swap_records.get(tid, {})
                if tid in self.partner_pending_tokens or rec.get('partner_id') is not None:
                    token = t
                    del self.waiting_swaps[source_name][i]
                    break
            else:
                self.waiting_swaps[source_name].popleft()

        if not self.pending_measurements[source_name]:
            self.waiting_swaps[source_name].appendleft(token)
            return

        idx, basis, bit = self.pending_measurements[source_name].popleft()
        token_id = token['token_id']
        partner_id = token['partner_id']
        parity_z = token['parity_z']
        parity_x = token['parity_x']
        partner_source = token['partner_source']
        swapper_id = token.get('swapper_id')

        rec = self.swap_records.setdefault(token_id, {})
        rec.update({
            'partner_id': partner_id,
            'my_source': source_name,
            'partner_source': partner_source,
            'parity_z': parity_z,
            'parity_x': parity_x,
            'my_index': idx,
            'my_basis': basis,
            'my_bit': bit
        })

        out_port = None
        # Prefer end-to-end partner route if known
        for candidate in (f'cout_{partner_id}', f'cout_{swapper_id}' if swapper_id is not None else None):
            if candidate and candidate in self.node.ports:
                out_port = candidate
                break
        if out_port:
            self.node.ports[out_port].tx_output(('PAIR', token_id, self.node_id, source_name, idx, basis, bit))
        else:
            logging.debug(f"{self.node.name} could not route PAIR token {token_id}")

        self._finalize_pair_if_ready(token_id)

    def _receive_swap_notification(self, token_id, swapper_id, partner_id, my_source, partner_source, parity_z, parity_x, depth, from_id=None):
        # Only accept final-depth tokens if expected_depth provided
        if self.expected_depth is not None and depth < self.expected_depth:
            return
        local_src = self.node_to_source_map.get(from_id) if from_id is not None else my_source
        if local_src is None:
            local_src = my_source
        token = {
            'token_id': token_id,
            'swapper_id': swapper_id,
            'partner_id': partner_id,
            'my_source': local_src,
            'partner_source': partner_source,
            'parity_z': parity_z,
            'parity_x': parity_x
        }
        rec = self.swap_records.setdefault(token_id, {})
        rec.update({
            'partner_id': partner_id,
            'my_source': local_src,
            'partner_source': partner_source,
            'parity_z': parity_z,
            'parity_x': parity_x
        })
        self.waiting_swaps.setdefault(local_src, deque()).append(token)
        if token_id in self.partner_pending_tokens:
            self._try_process_waiting_swap(local_src, force_token_id=token_id)
        else:
            self._try_process_waiting_swap(local_src)

    def _finalize_pair_if_ready(self, token_id):
        rec = self.swap_records.get(token_id)
        if not rec or rec.get('completed'):
            return
        if not (all(k in rec for k in ('my_basis', 'my_bit')) and all(k in rec for k in ('partner_basis', 'partner_bit', 'partner_id'))):
            return
        my_basis = rec['my_basis']
        my_bit = rec['my_bit']
        partner_bit = rec['partner_bit']
        if my_basis == 'Z':
            if rec.get('parity_z', 0) == 1:
                partner_bit ^= 1
        elif my_basis == 'X':
            if rec.get('parity_x', 0) == 1:
                partner_bit ^= 1
        if rec['partner_basis'] == my_basis:
            partner_id = rec['partner_id']
            if self.node_id < partner_id:
                key_id = f"{min(self.node_id, partner_id)}-{max(self.node_id, partner_id)}"
                self.shared_keys.setdefault(key_id, []).append(my_bit)
                self.key_bits.setdefault(partner_id, {})[token_id] = my_bit
                self.original_bits.setdefault(partner_id, {})[token_id] = partner_bit
                self._key_counter += 1
                if self._key_counter % self.key_log_every == 0:
                    logging.info(f"✅ KEY BIT (swap): {self.node.name} adds bit {my_bit} with Node_{partner_id} (token {token_id}, basis {my_basis})")
        rec['completed'] = True
        if token_id in self.partner_pending_tokens:
            self.partner_pending_tokens.discard(token_id)
        try:
            del self.swap_records[token_id]
        except Exception:
            pass

    def _handle_classical_with_node(self, message, sender_node_id):
        try:
            if message is None:
                return
            payload = message.items[0] if hasattr(message, 'items') and message.items else message

            if self.enable_swapping and isinstance(payload, (tuple, list)) and payload:
                tag = payload[0]
                if tag == 'SWAP' and len(payload) >= 9:
                    _, token_id, swapper_id, partner_id, my_source, partner_source, parity_z, parity_x, depth = payload[:9]
                    self._receive_swap_notification(token_id, swapper_id, partner_id, my_source, partner_source, parity_z, parity_x, depth, from_id=sender_node_id)
                    return
                if tag == 'PAIR' and len(payload) >= 7:
                    _, token_id, sender_id, source_name, index, basis, bit = payload
                    rec = self.swap_records.setdefault(token_id, {})
                    rec.update({
                        'partner_id': sender_id,
                        'partner_source': source_name,
                        'partner_index': index,
                        'partner_basis': basis,
                        'partner_bit': bit
                    })
                    if 'my_source' not in rec:
                        self.partner_pending_tokens.add(token_id)
                    else:
                        self._try_process_waiting_swap(rec['my_source'], force_token_id=token_id)
                    self._finalize_pair_if_ready(token_id)
                    return

            # Legacy P2P direct sifting
            if isinstance(payload, (tuple, list)) and len(payload) >= 5:
                sender_id, source_name, index, basis, remote_meas = payload
                if source_name in self.measurements:
                    measured_indices = [m[2] for m in self.measurements[source_name]]
                    if index in measured_indices:
                        idx = measured_indices.index(index)
                        own_basis, own_bit, _ = self.measurements[source_name][idx]
                        if own_basis == basis:
                            key_bit = own_bit
                            self.key_bits.setdefault(sender_id, {})[index] = key_bit
                            self.original_bits.setdefault(sender_id, {})[index] = remote_meas
                            key_id = f"{min(self.node_id, sender_id)}-{max(self.node_id, sender_id)}"
                            self.shared_keys.setdefault(key_id, []).append(key_bit)
                            logging.debug(f"KEY BIT: {self.node.name} adds bit {key_bit} to key with Node_{sender_id}")
        except Exception as e:
            logging.warning(f"Error handling classical message in {self.node.name}: {e}")

# Swapper protocol for multi-hop entanglement swapping
class EntanglementSwappingProtocol(NodeProtocol):
    def __init__(self, node, node_id, shared_keys, connected_nodes, swap_success_prob=1.0, upstream_ready_ports=None, path=None, num_hops=None):
        super().__init__(node)
        self.node_id = node_id
        self.shared_keys = shared_keys
        self.connected_nodes = connected_nodes
        self.swap_success_prob = swap_success_prob
        self.stored_qubits = {}
        self.swap_history = []
        self.total_swap_attempts = 0
        self.successful_swaps = 0
        self.end_to_end_pairs = {}
        self.qubit_hop_count = {}
        self.token_counter = 0
        self.token_routes = {}
        self.upstream_ready_ports = upstream_ready_ports or {}
        self.neighbor_frames = {}
        self.seen_swaps = set()
        self._seen_swaps_order = deque()
        self._max_seen_swaps = 10000
        self._route_keys_order = deque()
        self._max_routes = 10000
        self.swap_log_every = 200
        self.left_neighbor = None
        self.right_neighbor = None
        if path and self.node_id in path:
            idx = path.index(self.node_id)
            if idx > 0: self.left_neighbor = path[idx - 1]
            if idx < len(path) - 1: self.right_neighbor = path[idx + 1]
        self.num_hops = num_hops
        logging.info(f"Initialized EntanglementSwappingProtocol at {node.name} (left={self.left_neighbor}, right={self.right_neighbor}, hops={self.num_hops})")

    def run(self):
        def create_qubit_handler(source_name):
            def handler(message):
                self._handle_incoming_qubit(message, source_name)
                return None
            return handler
        def create_classical_handler(from_id):
            def handler(message):
                self._handle_classical_with_node(message, from_id)
                return None
            return handler
        for port_name in self.node.ports:
            if port_name.startswith("qin_Source_"):
                source_name = port_name[4:]
                self.node.ports[port_name].bind_input_handler(create_qubit_handler(source_name))
            elif port_name.startswith("cin_"):
                suffix = port_name[4:]
                if suffix.isdigit():
                    from_id = int(suffix)
                    self.node.ports[port_name].bind_input_handler(create_classical_handler(from_id))
        if self.upstream_ready_ports:
            for _, port_name in self.upstream_ready_ports.items():
                if port_name in self.node.ports:
                    try: self.node.ports[port_name].tx_output("READY")
                    except Exception: pass
        while True:
            if self.left_neighbor is not None and self.right_neighbor is not None:
                if self.stored_qubits.get(self.left_neighbor) and self.stored_qubits.get(self.right_neighbor):
                    self._perform_swap(self.left_neighbor, self.right_neighbor)
            yield self.await_timer(0.1)

    def _handle_incoming_qubit(self, message, source_name):
        try:
            if message is None:
                return
            payload = message.items[0] if hasattr(message, 'items') and message.items else message
            qubit = payload
            neighbor_id = self._extract_neighbor_id(source_name)
            if neighbor_id is None:
                return
            dq = self.stored_qubits.setdefault(neighbor_id, deque())
            dq.append({'qubit': qubit, 'source': source_name})
            logging.debug(f"{self.node.name} stored qubit from Node_{neighbor_id} via {source_name}")
        except Exception as e:
            logging.warning(f"Error handling incoming qubit in {self.node.name}: {e}")

    def _extract_neighbor_id(self, source_name):
        if not isinstance(source_name, str):
            return None
        if source_name.startswith("Source_"):
            parts = source_name.split('_')
            if len(parts) >= 3 and parts[1].isdigit() and parts[2].isdigit():
                n1, n2 = int(parts[1]), int(parts[2])
                if self.node_id == n1: return n2
                if self.node_id == n2: return n1
        return None

    def _perform_swap(self, node1, node2):
        self.total_swap_attempts += 1
        if not self.stored_qubits.get(node1) or not self.stored_qubits.get(node2):
            return
        entry1 = self.stored_qubits[node1].popleft()
        entry2 = self.stored_qubits[node2].popleft()
        qubit1 = entry1['qubit']; qubit2 = entry2['qubit']
        src1 = entry1['source']; src2 = entry2['source']
        swap_succeeds = random.random() < self.swap_success_prob
        if swap_succeeds:
            try:
                qapi.operate([qubit1, qubit2], ns.CX)
                qapi.operate(qubit1, H)
                m1, _ = qapi.measure(qubit1, Z)
                m2, _ = qapi.measure(qubit2, Z)
                local_z = int(m1); local_x = int(m2)
            except Exception as e:
                logging.warning(f"BSM operation failed at {self.node.name}: {e}")
                for nid in (node1, node2):
                    port = self.upstream_ready_ports.get(nid)
                    if port and port in self.node.ports:
                        self.node.ports[port].tx_output("READY")
                return
            f1 = self.neighbor_frames.get(node1, {'z': 0, 'x': 0, 'depth': 0})
            f2 = self.neighbor_frames.get(node2, {'z': 0, 'x': 0, 'depth': 0})
            try: d1 = int(f1.get('depth', 0))
            except Exception: d1 = 0
            try: d2 = int(f2.get('depth', 0))
            except Exception: d2 = 0
            agg_z = int(f1.get('z', 0)) ^ int(f2.get('z', 0)) ^ local_z
            agg_x = int(f1.get('x', 0)) ^ int(f2.get('x', 0)) ^ local_x
            agg_depth = d1 + d2 + 1
            if self.num_hops is not None:
                agg_depth = max(1, min(self.num_hops, agg_depth))

            self.successful_swaps += 1
            self.token_counter += 1
            token_id = (self.node_id << 20) + self.token_counter
            self.token_routes[token_id] = (node1, node2)
            self._route_keys_order.append(token_id)
            if len(self._route_keys_order) > self._max_routes:
                old = self._route_keys_order.popleft()
                self.token_routes.pop(old, None)

            msg_left = ('SWAP', token_id, self.node_id, node2, src1, src2, agg_z, agg_x, agg_depth)
            msg_right = ('SWAP', token_id, self.node_id, node1, src2, src1, agg_z, agg_x, agg_depth)
            if f"cout_{node1}" in self.node.ports:
                self.node.ports[f"cout_{node1}"].tx_output(msg_left)
            if f"cout_{node2}" in self.node.ports:
                self.node.ports[f"cout_{node2}"].tx_output(msg_right)
            if self.successful_swaps % self.swap_log_every == 0:
                logging.info(f"SWAP: {self.node.name} depth {agg_depth} Z:{agg_z} X:{agg_x} between Node_{node1} and Node_{node2}; token {token_id}")
        else:
            try: qapi.measure(qubit1)
            except Exception: pass
            try: qapi.measure(qubit2)
            except Exception: pass
            logging.debug(f"SWAP FAILED at {self.node.name}")

        for nid in (node1, node2):
            port = self.upstream_ready_ports.get(nid)
            if port and port in self.node.ports:
                self.node.ports[port].tx_output("READY")

    def _handle_classical_with_node(self, message, from_id):
        try:
            if message is None:
                return
            payload = message.items[0] if hasattr(message, 'items') and message.items else message
            if not (isinstance(payload, (tuple, list)) and payload):
                return

            tag = payload[0]
            if tag == 'PAIR':
                if len(payload) < 7:
                    return
                _, token_id, sender_id, *_ = payload
                route = self.token_routes.get(token_id)
                if route:
                    left, right = route
                    other = right if sender_id == left else left
                    out_port = f"cout_{other}"
                else:
                    if self.left_neighbor is not None and self.right_neighbor is not None:
                        other = self.right_neighbor if from_id == self.left_neighbor else self.left_neighbor
                        out_port = f"cout_{other}"
                    else:
                        candidates = [n for n in self.connected_nodes if n != from_id]
                        out_port = f"cout_{candidates[0]}" if candidates else None
                if out_port and out_port in self.node.ports:
                    self.node.ports[out_port].tx_output(payload)
                else:
                    logging.warning(f"{self.node.name} cannot forward PAIR token {token_id}")
                return

            if tag == 'SWAP':
                if len(payload) < 9:
                    return
                _, token_id, swapper_id, partner_id, my_source, partner_source, parity_z, parity_x, depth = payload
                try: parity_z = int(parity_z)
                except Exception: parity_z = 0
                try: parity_x = int(parity_x)
                except Exception: parity_x = 0
                try: depth = int(depth)
                except Exception: depth = 0
                if self.num_hops is not None:
                    depth = max(0, min(self.num_hops, depth))
                self.neighbor_frames[from_id] = {'z': parity_z, 'x': parity_x, 'depth': depth}
                if token_id in self.seen_swaps:
                    return
                self.seen_swaps.add(token_id)
                self._seen_swaps_order.append(token_id)
                if len(self._seen_swaps_order) > self._max_seen_swaps:
                    old = self._seen_swaps_order.popleft()
                    self.seen_swaps.discard(old)
                if self.left_neighbor is not None and self.right_neighbor is not None:
                    other = self.right_neighbor if from_id == self.left_neighbor else self.left_neighbor
                    out_port = f"cout_{other}"
                    if out_port in self.node.ports:
                        self.node.ports[out_port].tx_output(('SWAP', token_id, swapper_id, partner_id, my_source, partner_source, parity_z, parity_x, depth))
                return
        except Exception as e:
            logging.warning(f"Error handling classical message at {self.node.name}: {e}")

def create_quantum_channel(length_km, loss_per_km, detector_efficiency=1.0):
    """Create a quantum channel with specified length and loss rate."""
    channel = QuantumChannel(
        "QuantumChannel",
        length=length_km,
        models={
            "quantum_loss_model": FibreLossModel(p_loss_init=0, p_loss_length=loss_per_km),
            "quantum_noise_model": DepolarNoiseModel(depolar_rate=1-detector_efficiency)
        }
    )
    return channel

def create_classical_channel(length_km):
    """Create a classical channel with specified length."""
    return ClassicalChannel("ClassicalChannel", length=length_km)

def calculate_qber(original_bits, key_bits):
    """Calculate Quantum Bit Error Rate between original and measured bits."""
    if not original_bits or not key_bits:
        return 0.0

    # Get matching indices
    common_indices = set(original_bits.keys()) & set(key_bits.keys())
    if not common_indices:
        return 0.0

    # Count errors
    errors = sum(1 for idx in common_indices if original_bits[idx] != key_bits[idx])
    return errors / len(common_indices)

def create_qkd_network(num_nodes, channel_length_km, detector_efficiency, loss_per_km, topology):
    """Create a QKD network with specified topology."""
    network = Network("EB-QKD Network")

    # Create regular nodes
    nodes = [Node(f"Node_{i}") for i in range(num_nodes)]

    # Determine potential node pairs based on topology
    node_pairs = []
    if topology == "fully_connected":
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                node_pairs.append((i, j))

    elif topology == "star":
        central_node = 0
        for i in range(1, num_nodes):
            node_pairs.append((central_node, i))

    elif topology == "ring":
        for i in range(num_nodes):
            j = (i + 1) % num_nodes
            node_pairs.append((i, j))

    elif topology == "bus":

        for i in range(num_nodes - 1):
            node_pairs.append((i, i+1))

    # Create source nodes (one between each communicating pair)
    sources = {}  # Stores unique source nodes
    source_lookup = {} 

    for node1_id, node2_id in node_pairs:
        # Normalize the pair (smaller ID first) for consistent source creation
        n1, n2 = min(node1_id, node2_id), max(node1_id, node2_id)
        key = (n1, n2)

        if key not in sources:
            source_id = f"Source_{n1}_{n2}"
            source_node = Node(source_id)
            sources[key] = source_node

        # Add both directions to lookup dictionary
        source_lookup[(node1_id, node2_id)] = sources[key]
        source_lookup[(node2_id, node1_id)] = sources[key]

    # Add all nodes to network
    network.add_nodes(nodes)
    network.add_nodes(list(sources.values()))

    # Add ports to regular nodes
    for i in range(num_nodes):
        # Find all sources that connect to this node
        connected_sources = set()
        for (n1, n2), source in source_lookup.items():
            if n1 == i or n2 == i:
                connected_sources.add(source)

        # Quantum ports (receiving from sources)
        for source in connected_sources:
            nodes[i].add_ports([f"qin_{source.name}"])

        # Classical ports (communicating with connected nodes only)
        connected_nodes = set()
        for (n1, n2) in node_pairs:
            if n1 == i:
                connected_nodes.add(n2)
            elif n2 == i:
                connected_nodes.add(n1)

        nodes[i].add_ports([f"cout_{j}" for j in connected_nodes])
        nodes[i].add_ports([f"cin_{j}" for j in connected_nodes])

    # Add ports to source nodes
    for (n1, n2), source_node in sources.items():
        # Quantum ports (sending to regular nodes)
        source_node.add_ports([f"qout_{n1}", f"qout_{n2}"])

    # Track source connections for protocol setup
    entanglement_connections = {}

    # Create connections based on topology
    for (node1_id, node2_id), source_node in source_lookup.items():
        # Ensure we only process each source once (when first node ID is smaller)
        if node1_id > node2_id:
            continue

        # Create quantum channels from source to each node (half the total distance each)
        q_chan_1 = create_quantum_channel(channel_length_km/2, loss_per_km, detector_efficiency)
        q_conn_1 = DirectConnection(f"q_conn_{source_node.name}_{node1_id}", channel_AtoB=q_chan_1)

        q_chan_2 = create_quantum_channel(channel_length_km/2, loss_per_km, detector_efficiency)
        q_conn_2 = DirectConnection(f"q_conn_{source_node.name}_{node2_id}", channel_AtoB=q_chan_2)

        # Add quantum connections from source to each node
        network.add_connection(
            source_node, nodes[node1_id],
            connection=q_conn_1,
            label="quantum",
            port_name_node1=f"qout_{node1_id}",
            port_name_node2=f"qin_{source_node.name}"
        )

        network.add_connection(
            source_node, nodes[node2_id],
            connection=q_conn_2,
            label="quantum",
            port_name_node1=f"qout_{node2_id}",
            port_name_node2=f"qin_{source_node.name}"
        )

        # Create classical channel between the nodes (full distance)
        c_chan_12 = create_classical_channel(channel_length_km)
        c_conn_12 = DirectConnection(f"c_conn_{node1_id}_{node2_id}", channel_AtoB=c_chan_12)

        c_chan_21 = create_classical_channel(channel_length_km)
        c_conn_21 = DirectConnection(f"c_conn_{node2_id}_{node1_id}", channel_AtoB=c_chan_21)

        # Add classical connections between nodes
        network.add_connection(
            nodes[node1_id], nodes[node2_id],
            connection=c_conn_12,
            label="classical",
            port_name_node1=f"cout_{node2_id}",
            port_name_node2=f"cin_{node1_id}"
        )

        network.add_connection(
            nodes[node2_id], nodes[node1_id],
            connection=c_conn_21,
            label="classical",
            port_name_node1=f"cout_{node1_id}",
            port_name_node2=f"cin_{node2_id}"
        )

        # Track which node pair this source connects
        if source_node not in entanglement_connections:
            entanglement_connections[source_node] = []
        entanglement_connections[source_node].append((node1_id, node2_id))

    return network, nodes, entanglement_connections

def visualize_network(network, topology="fully_connected"):
    """Create a visualization of the network topology."""
    plt.figure(figsize=(12, 10))
    G = nx.MultiGraph()

    for node in network.nodes:
        node_name = str(node)
        is_source = "Source" in node_name
        node_color = 'red' if is_source else 'skyblue'
        node_size = 2000 if is_source else 3000
        G.add_node(node_name, color=node_color, size=node_size)

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
        # For ring, place regular nodes in a circle and sources between them
        regular_nodes = [n for n in G.nodes if "Source" not in n]
        source_nodes = [n for n in G.nodes if "Source" in n]

        # Create regular node positions in a circle
        pos_regular = nx.circular_layout(regular_nodes)

        # Place source nodes between their connected nodes
        pos = pos_regular.copy()
        for source in source_nodes:
            # get the connected node indices from the source name
            try:
                # get node IDs from Source_X_Y format
                parts = source.split('_')
                if len(parts) >= 3:
                    node1 = f"Node_{parts[1]}"
                    node2 = f"Node_{parts[2]}"
                    if node1 in pos_regular and node2 in pos_regular:
                        
                        pos[source] = ((pos_regular[node1][0] + pos_regular[node2][0])/2, 
                                      (pos_regular[node1][1] + pos_regular[node2][1])/2)
                    else:
                        
                        pos[source] = (0.1*(random.random()-0.5), 0.1*(random.random()-0.5))
                else:
                   
                    pos[source] = (0.1*(random.random()-0.5), 0.1*(random.random()-0.5))
            except:
                
                pos[source] = (0.1*(random.random()-0.5), 0.1*(random.random()-0.5))

    elif topology == "bus":
        # For bus topology, create a horizontal layout with sources between nodes
        regular_nodes = [n for n in G.nodes if "Source" not in n]
        source_nodes = [n for n in G.nodes if "Source" in n]

        # Sort regular nodes by index
        regular_nodes.sort(key=lambda x: int(x.split('_')[1]))

        # Position regular nodes in a horizontal line
        node_count = len(regular_nodes)
        x_positions = np.linspace(0, 1, node_count)
        y_position = 0.5  # Center vertically

        pos_regular = {regular_nodes[i]: (x_positions[i], y_position) for i in range(node_count)}

        # Place source nodes between their connected regular nodes
        pos = pos_regular.copy()
        for source in source_nodes:
            # get the connected node indices from the source name
            try:
                # get node IDs from Source_X_Y format
                parts = source.split('_')
                if len(parts) >= 3:
                    node1 = f"Node_{parts[1]}"
                    node2 = f"Node_{parts[2]}"
                    if node1 in pos_regular and node2 in pos_regular:
                        
                        pos[source] = ((pos_regular[node1][0] + pos_regular[node2][0])/2, 
                                      (pos_regular[node1][1] + pos_regular[node2][1])/2 - 0.1)  # Offset slightly
                    else:
                       
                        pos[source] = (random.random(), y_position - 0.1)
                else:
                    
                    pos[source] = (random.random(), y_position - 0.1)
            except:
                
                pos[source] = (random.random(), y_position - 0.1)

        plt.xlim(-0.1, 1.1)
        plt.ylim(0.2, 0.8)

    elif topology == "star":
        # For star, central node in center, others in circle, sources between connections
        regular_nodes = [n for n in G.nodes if "Source" not in n]
        source_nodes = [n for n in G.nodes if "Source" in n]

        # Use kamada_kawai for initial positions
        pos_initial = nx.kamada_kawai_layout(G)

        # Ensure central Node_0 is at the center
        central_node = "Node_0"
        if central_node in pos_initial:
            pos_initial[central_node] = (0, 0)


        non_central_nodes = [n for n in regular_nodes if n != central_node]
        angle_step = 2 * np.pi / len(non_central_nodes)
        radius = 0.5

        pos_regular = {central_node: (0, 0)} if central_node in regular_nodes else {}

        for i, node in enumerate(non_central_nodes):
            angle = i * angle_step
            pos_regular[node] = (radius * np.cos(angle), radius * np.sin(angle))

        # Place source nodes between their connected nodes
        pos = pos_regular.copy()
        for source in source_nodes:
            # get the connected node indices from the source name
            try:
                # get node IDs from Source_X_Y format
                parts = source.split('_')
                if len(parts) >= 3:
                    node1 = f"Node_{parts[1]}"
                    node2 = f"Node_{parts[2]}"
                    if node1 in pos_regular and node2 in pos_regular:
                        # Position the source halfway between its connected nodes
                        pos[source] = ((pos_regular[node1][0] + pos_regular[node2][0])/2, 
                                      (pos_regular[node1][1] + pos_regular[node2][1])/2)
                    else:
                        angle = random.random() * 2 * np.pi
                        pos[source] = (0.25 * np.cos(angle), 0.25 * np.sin(angle))
                else:
                    angle = random.random() * 2 * np.pi
                    pos[source] = (0.25 * np.cos(angle), 0.25 * np.sin(angle))
            except:
                angle = random.random() * 2 * np.pi
                pos[source] = (0.25 * np.cos(angle), 0.25 * np.sin(angle))

    else:  
        
        regular_nodes = [n for n in G.nodes if "Source" not in n]
        source_nodes = [n for n in G.nodes if "Source" in n]

        # Get regular node positions first
        pos_regular = nx.spring_layout(G.subgraph(regular_nodes), seed=42)

        # Place source nodes between their connected regular nodes
        pos = pos_regular.copy()
        for source in source_nodes:
            # Extract the connected node indices from the source name
            try:
                # get node IDs from Source_X_Y format
                parts = source.split('_')
                if len(parts) >= 3:
                    node1 = f"Node_{parts[1]}"
                    node2 = f"Node_{parts[2]}"
                    if node1 in pos_regular and node2 in pos_regular:
                        # Position the source halfway between its connected nodes
                        pos[source] = ((pos_regular[node1][0] + pos_regular[node2][0])/2, 
                                      (pos_regular[node1][1] + pos_regular[node2][1])/2)
                    else:
                        
                        pos[source] = (random.random()-0.5, random.random()-0.5)
                else:
                    
                    pos[source] = (random.random()-0.5, random.random()-0.5)
            except:
                
                pos[source] = (random.random()-0.5, random.random()-0.5)

    
    node_colors = [G.nodes[node].get('color', 'skyblue') for node in G.nodes()]
    node_sizes = [G.nodes[node].get('size', 3000) for node in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors)

   
    regular_labels = {node: node for node in G.nodes() if "Source" not in node}
    source_labels = {node: node for node in G.nodes() if "Source" in node}

    nx.draw_networkx_labels(G, pos, labels=regular_labels, font_size=16, font_weight='bold')
    nx.draw_networkx_labels(G, pos, labels=source_labels, font_size=10)

    # Draw edges
    for u, v, key, data in G.edges(keys=True, data=True):
        style = data.get('style', 'solid')
        color = data.get('color', 'black')
        weight = data.get('weight', 1)
        curve = "arc3,rad=0.2" if data['label'] == "Quantum Channel" else "arc3,rad=-0.2"
        nx.draw_networkx_edges(
            G, pos, edgelist=[(u, v)],
            edge_color=color,
            style=style,
            width=weight,
            connectionstyle=curve
        )

    # Add legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='skyblue', markersize=15, label='QKD Node'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Entanglement Source'),
        Line2D([0], [0], color='blue', lw=2, label='Quantum Channel'),
        Line2D([0], [0], color='green', lw=2, linestyle='dashed', label='Classical Channel')
    ]
    plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98), fontsize=16)
    plt.title(f"Distributed Source EB-QKD Network ({topology} topology)", fontsize=22, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()

    # Save figure
    output_path = os.path.join(OUTPUT_DIR, f"eb_distributed_source_network_{topology}.png")
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    logging.info(f"Network visualisation saved as '{output_path}'")
    plt.close()

def visualize_p2p_network(network):
    """Create a visualization of a P2P EB-QKD network with source in middle."""
    G = nx.MultiGraph()

    
    for node in network.nodes:
        node_name = str(node)
        is_source = "Source" in node_name
        node_color = 'red' if is_source else 'skyblue'
        node_size = 2000 if is_source else 3000
        G.add_node(node_name, color=node_color, size=node_size)

    
    for connection in network.connections:
        connection_str = str(connection)
        if "conn|" in connection_str and "<->" in connection_str:
            parts = connection_str.split("|")[1]
            node1_name, node2_name = parts.split("<->")

            if 'quantum' in connection_str:
                G.add_edge(node1_name, node2_name, label="Quantum Channel", color='blue', style='solid', weight=2)
            elif 'classical' in connection_str:
                G.add_edge(node1_name, node2_name, label="Classical Channel", color='green', style='dashed', weight=2)

    # Position nodes in a horizontal line with source in middle
    pos = {
        "Node_0": (-1, 0),
        "Source_0_1": (0, 0),
        "Node_1": (1, 0)
    }

    # Draw nodes with different colors
    node_colors = [G.nodes[node].get('color', 'skyblue') for node in G.nodes()]
    node_sizes = [G.nodes[node].get('size', 3000) for node in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors)

    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=16, font_weight='bold')

    # Draw edges
    for u, v, key, data in G.edges(keys=True, data=True):
        style = data.get('style', 'solid')
        color = data.get('color', 'black')
        weight = data.get('weight', 1)
        curve = "arc3,rad=0.2" if data['label'] == "Quantum Channel" else "arc3,rad=-0.2"
        nx.draw_networkx_edges(
            G, pos, edgelist=[(u, v)],
            edge_color=color,
            style=style,
            width=weight,
            connectionstyle=curve
        )

    # Add legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='skyblue', markersize=15, label='QKD Node'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Entanglement Source'),
        Line2D([0], [0], color='blue', lw=2, label='Quantum Channel'),
        Line2D([0], [0], color='green', lw=2, linestyle='dashed', label='Classical Channel')
    ]
    plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98), fontsize=16)

    plt.title(f"P2P EB-QKD Network with Entanglement Source in Middle", fontsize=22, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()

    # Save figure
    output_path = os.path.join(OUTPUT_DIR, f"p2p_eb_qkd_network.png")
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    logging.info(f"P2P network visualisation saved as '{output_path}'")
    plt.close()

def visualize_star_central_source_quantum_only(num_leaves=5):
    """Visualise a star where the central node is a source and leaves connect via quantum channels only."""
    
    network = Network("EB-QKD Star (Central Source, Quantum-Only)")
    source = Node("Source_Center")
    leaves = [Node(f"Node_{i}") for i in range(1, num_leaves + 1)]
    network.add_nodes([source] + leaves)

    
    source.add_ports([f"qout_{i}" for i in range(1, num_leaves + 1)])
    for i, leaf in enumerate(leaves, start=1):
        leaf.add_ports([f"qin_Source_Center"])

        # Create quantum channel only
        q_chan = create_quantum_channel(length_km=1, loss_per_km=0.2, detector_efficiency=0.9)
        q_conn = DirectConnection(f"q_conn_Source_Center_{i}", channel_AtoB=q_chan)
        network.add_connection(
            source, leaf,
            connection=q_conn,
            label="quantum",
            port_name_node1=f"qout_{i}",
            port_name_node2="qin_Source_Center"
        )

    # visualise
    plt.figure(figsize=(10, 8))
    G = nx.MultiGraph()
    for node in network.nodes:
        node_name = str(node)
        is_source = "Source" in node_name
        node_color = 'red' if is_source else 'skyblue'
        node_size = 2500 if is_source else 2800
        G.add_node(node_name, color=node_color, size=node_size)

    for connection in network.connections:
        connection_str = str(connection)
        if "conn|" in connection_str and "<->" in connection_str:
            parts = connection_str.split("|")[1]
            node1_name, node2_name = parts.split("<->")
            if 'quantum' in connection_str:
                G.add_edge(node1_name, node2_name, label="Quantum Channel", color='blue', style='solid', weight=2)

    
    pos = {}
    pos["Source_Center"] = (0.0, 0.0)
    angle_step = 2 * np.pi / num_leaves if num_leaves > 0 else 1
    radius = 1.0
    for i, leaf in enumerate(G.nodes()):
        if leaf == "Source_Center":
            continue
        angle = i * angle_step
        pos[leaf] = (radius * np.cos(angle), radius * np.sin(angle))

    # Draw
    node_colors = [G.nodes[node].get('color', 'skyblue') for node in G.nodes()]
    node_sizes = [G.nodes[node].get('size', 3000) for node in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors)
    nx.draw_networkx_labels(G, pos, font_size=14, font_weight='bold')

    for u, v, key, data in G.edges(keys=True, data=True):
        style = data.get('style', 'solid')
        color = data.get('color', 'black')
        weight = data.get('weight', 1)
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], edge_color=color, style=style, width=weight)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='skyblue', markersize=12, label='QKD Node'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Entanglement Source'),
        Line2D([0], [0], color='blue', lw=2, label='Quantum Channel')
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=14)
    plt.title("EB-QKD Star Visualisation (Central Source, Quantum-Only Links)", fontsize=22, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()

    out = os.path.join(OUTPUT_DIR, "eb_star_central_source_quantum_only.png")
    plt.savefig(out, bbox_inches='tight', dpi=300)
    logging.info(f"Star network visualisation saved as '{out}'")
    plt.close()

def run_p2p_simulation(channel_length_km, detector_efficiency, loss_per_km, simulation_duration):
    """Run a point-to-point EB-QKD simulation with just two nodes."""
    logging.info(f"Starting P2P simulation: {channel_length_km}km")

    # Reset NetSquid
    ns.sim_reset()

    
    network = Network("P2P EB-QKD Network")

    
    node0 = Node("Node_0")
    node1 = Node("Node_1")
    source = Node("Source_0_1")

    # Add nodes to the network
    network.add_nodes([node0, node1, source])

    # Add ports to the nodes
    node0.add_ports(["qin_Source_0_1", "cout_1", "cin_1"])
    node1.add_ports(["qin_Source_0_1", "cout_0", "cin_0"])
    source.add_ports(["qout_0", "qout_1"])

    # Create quantum channels from source to each node (half the total distance each)
    q_chan_0 = create_quantum_channel(channel_length_km/2, loss_per_km, detector_efficiency)
    q_conn_0 = DirectConnection(f"q_conn_Source_0_1_0", channel_AtoB=q_chan_0)

    q_chan_1 = create_quantum_channel(channel_length_km/2, loss_per_km, detector_efficiency)
    q_conn_1 = DirectConnection(f"q_conn_Source_0_1_1", channel_AtoB=q_chan_1)

    # Add quantum connections from source to each node
    network.add_connection(
        source, node0,
        connection=q_conn_0,
        label="quantum",
        port_name_node1="qout_0",
        port_name_node2="qin_Source_0_1"
    )

    network.add_connection(
        source, node1,
        connection=q_conn_1,
        label="quantum",
        port_name_node1="qout_1",
        port_name_node2="qin_Source_0_1"
    )

    # Create classical channels between the nodes
    c_chan_01 = create_classical_channel(channel_length_km)
    c_conn_01 = DirectConnection(f"c_conn_0_1", channel_AtoB=c_chan_01)

    c_chan_10 = create_classical_channel(channel_length_km)
    c_conn_10 = DirectConnection(f"c_conn_1_0", channel_AtoB=c_chan_10)

    # Add classical connections
    network.add_connection(
        node0, node1,
        connection=c_conn_01,
        label="classical",
        port_name_node1="cout_1",
        port_name_node2="cin_0"
    )

    network.add_connection(
        node1, node0,
        connection=c_conn_10,
        label="classical",
        port_name_node1="cout_0",
        port_name_node2="cin_1"
    )

    # to store shared keys
    shared_keys = {}

    # Create protocols
    protocols = []

    # Source protocol
    source_protocol = EntanglementSourceProtocol(source, [(0, 1)], pair_generation_rate=1.0)
    protocols.append(source_protocol)

    
    node0_protocol = NodeMeasurementProtocol(node0, 0, shared_keys, [1])
    protocols.append(node0_protocol)

    node1_protocol = NodeMeasurementProtocol(node1, 1, shared_keys, [0])
    protocols.append(node1_protocol)

    
    for protocol in protocols:
        protocol.start()

    # Run simulation in chunks for better monitoring
    chunk_size = 10
    num_chunks = simulation_duration // chunk_size
    for i in range(num_chunks):
        ns.sim_run(duration=chunk_size)
        if shared_keys and i % 10 == 0:  # Log less frequently
            logging.info(f"Simulation progress: {(i+1)*chunk_size}/{simulation_duration}, Keys: {len(shared_keys)}")

    # Calculate results
    key_id = "0-1"  # For P2P, there's only one key

    # Calculate key rate and other metrics
    if key_id in shared_keys:
        key_length = len(shared_keys[key_id])
        key_rate = key_length / simulation_duration
        qber = 0.0 

        # Calculate QBER if we have the data
        if hasattr(node0_protocol, 'original_bits') and hasattr(node0_protocol, 'key_bits'):
            if 1 in node0_protocol.original_bits and 1 in node0_protocol.key_bits:
                qber = calculate_qber(node0_protocol.original_bits[1], node0_protocol.key_bits[1])
    else:
        key_length = 0
        key_rate = 0
        qber = 0

    # Visualize the network
    if channel_length_km == 20:  
        plt.figure(figsize=(10, 8))
        visualize_p2p_network(network)

    return {
        "key_rate": key_rate,
        "key_length": key_length,
        "qber": qber,
        "channel_length": channel_length_km,
        "total_distance": channel_length_km,  # Same as channel length for P2P
        "photon_travel": channel_length_km/2  # Each photon travels half the distance
    }

def run_p2p_distance_sweep():
    """Run P2P simulations at different distances."""
    distances = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 110, 120, 130, 150]
    detector_efficiency = 0.9
    loss_per_km = 0.2
    simulation_duration = 500

    results = []

    for distance in distances:
        logging.info(f"\n===== Running P2P EB-QKD at {distance}km =====")
        result = run_p2p_simulation(
            distance,
            detector_efficiency,
            loss_per_km,
            simulation_duration
        )
        results.append(result)

    return results

def create_p2p_plot(results):
    """Create a P2P key rate vs. distance plot similar to relay protocol."""
    plt.figure(figsize=(7, 5))  # match PM baseline size

    distances = [r["channel_length"] for r in results]
    key_rates = [r["key_rate"] for r in results]
    qbers = [r["qber"] for r in results]

    # Key rate vs distance 
    plt.plot(distances, key_rates, marker='X', linestyle='-',
             linewidth=1.5, markersize=6, color='#d62728')
    plt.xlabel('Total End-to-End Distance (km)', fontsize=13)
    plt.ylabel('Key Exchange Rate (bits/second)', fontsize=13)
    plt.title('Point-to-Point Key Exchange Rate vs. Distance\n(No Relay Nodes)', fontsize=16, fontweight='bold')
    plt.grid(True)
    plt.xlim(10, 200)
    plt.xticks(np.arange(10, 201, 10), fontsize=9)
    plt.yticks(fontsize=9)
    y_max = max(key_rates) if key_rates else 0
    plt.ylim(0, upper_limit(y_max))
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "p2p_key_rate_vs_distance.png")
    plt.savefig(output_path, dpi=300)
    plt.close()

    
    plt.figure(figsize=(7, 5))  
    plt.plot(distances, qbers, marker='X', linestyle='-',
             linewidth=1.5, markersize=6, color='#d62728')
    plt.xlabel('Total End-to-End Distance (km)', fontsize=13)
    plt.ylabel('Quantum Bit Error Rate (QBER)', fontsize=13)
    plt.title('Point-to-Point QBER vs. Distance\n(No Relay Nodes)', fontsize=16, fontweight='bold')
    plt.grid(True)
    plt.xlim(10, 200)
    plt.xticks(np.arange(10, 201, 10), fontsize=9)
    plt.yticks(fontsize=9)
    y_max_qber = max(qbers) if qbers else 0
    plt.ylim(0, upper_limit(y_max_qber))
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "p2p_qber_vs_distance.png")
    plt.savefig(output_path, dpi=300)
    plt.close()


def run_simulation(num_nodes, channel_length_km, detector_efficiency, loss_per_km, topology, simulation_duration):
    """Run an EB-QKD simulation with the specified parameters."""
    logging.info(f"Starting simulation: {topology} topology, {channel_length_km}km")

    # Reset NetSquid
    ns.sim_reset()

    
    network, nodes, entanglement_connections = create_qkd_network(
        num_nodes, channel_length_km, detector_efficiency, loss_per_km, topology
    )

    shared_keys = {}

    
    connected_nodes_map = {}
    for i in range(num_nodes):
        connected_nodes_map[i] = []

    if topology == "fully_connected":
        # All nodes are connected to each other
        for i in range(num_nodes):
            connected_nodes_map[i] = [j for j in range(num_nodes) if j != i]

    elif topology == "star":
        # Peripheral nodes connect only to center node
        central_node = 0
        for i in range(num_nodes):
            if i == central_node:
                connected_nodes_map[i] = [j for j in range(num_nodes) if j != i]
            else:
                connected_nodes_map[i] = [central_node]

    elif topology == "ring":
        # Each node connects only to adjacent nodes
        for i in range(num_nodes):
            connected_nodes_map[i] = [(i-1) % num_nodes, (i+1) % num_nodes]

    elif topology == "bus":
        
        for i in range(num_nodes):
            connected = []
            if i > 0:  # Connect to previous node
                connected.append(i-1)
            if i < num_nodes - 1:  # Connect to next node
                connected.append(i+1)
            connected_nodes_map[i] = connected

    # Create protocols
    protocols = []

    # Source protocols (one for each entanglement source)
    for source_node, connections in entanglement_connections.items():
        source_protocol = EntanglementSourceProtocol(source_node, connections, pair_generation_rate=1.0)
        protocols.append(source_protocol)

    # Node protocols for measuring qubits
    node_measurements = {}
    for i in range(num_nodes):
        if connected_nodes_map[i]:
            node_protocol = NodeMeasurementProtocol(nodes[i], i, shared_keys, connected_nodes_map[i])
            protocols.append(node_protocol)
            node_measurements[i] = node_protocol

   
    for protocol in protocols:
        protocol.start()

    
    chunk_size = 10
    num_chunks = simulation_duration // chunk_size
    for i in range(num_chunks):
        ns.sim_run(duration=chunk_size)
        if shared_keys:
            logging.info(f"Simulation progress: {(i+1)*chunk_size}/{simulation_duration}, Keys: {len(shared_keys)}")

    # Calculate key exchange rates and QBER
    key_rates = {}
    qbers = {}
    total_bits = 0

    for key_id, key in shared_keys.items():
        rate = len(key) / simulation_duration
        key_rates[key_id] = rate
        total_bits += len(key)

        # Get node pair from key_id
        node1, node2 = map(int, key_id.split('-'))

        # Calculate QBER if we have data from both nodes
        if node1 in node_measurements and node2 in node_measurements:
            # Get bits from one node
            if node2 in node_measurements[node1].original_bits and node2 in node_measurements[node1].key_bits:
                original_bits = node_measurements[node1].original_bits[node2]
                key_bits = node_measurements[node1].key_bits[node2]
                qbers[key_id] = calculate_qber(original_bits, key_bits)
            elif node1 in node_measurements[node2].original_bits and node1 in node_measurements[node2].key_bits:
                original_bits = node_measurements[node2].original_bits[node1]
                key_bits = node_measurements[node2].key_bits[node1]
                qbers[key_id] = calculate_qber(original_bits, key_bits)
            else:
                qbers[key_id] = 0.0
        else:
            qbers[key_id] = 0.0

        logging.info(f"Key {key_id}: {len(key)} bits, Rate: {rate:.4f} bits/time unit, QBER: {qbers[key_id]:.4f}")

    # Calculate overall metrics
    avg_rate = sum(key_rates.values()) / len(key_rates) if key_rates else 0
    avg_qber = sum(qbers.values()) / len(qbers) if qbers else 0

    # Return results
    results = {
        "topology": topology,
        "channel_length_km": channel_length_km,
        "photon_travel_km": channel_length_km/2,  # Each photon travels half the distance
        "detector_efficiency": detector_efficiency,
        "loss_per_km": loss_per_km,
        "simulation_duration": simulation_duration,
        "shared_keys": shared_keys,
        "key_rates": key_rates,
        "qbers": qbers,
        "avg_rate": avg_rate,
        "avg_qber": avg_qber,
        "total_bits": total_bits
    }

    return results

def run_distance_sweep(num_nodes, distances, detector_efficiency, loss_per_km, topology, simulation_duration):
    """Run simulations at different distances for a specific topology."""
    results = {}
    total = len(distances)
    for idx, distance in enumerate(distances, start=1):
        logging.info(f"\n===== Running {topology} topology at {distance}km ({idx}/{total}) =====")
        sim_results = run_simulation(
            num_nodes,
            distance,
            detector_efficiency,
            loss_per_km,
            topology,
            simulation_duration
        )
        results[distance] = sim_results
    return results

def create_topology_comparison_plots(all_results, distances, topologies):
    """Create comparison plots between different topologies."""
    # Create directory for comparison plots
    comparison_dir = os.path.join(OUTPUT_DIR, "topology_comparison")
    os.makedirs(comparison_dir, exist_ok=True)

    # Key rate comparison
    plt.figure(figsize=(12, 8))

    # Define colors for each topology
    topology_colors = {
        'fully_connected': '#1f77b4',  # Blue
        'star': '#ff7f0e',             # Orange
        'ring': '#2ca02c',             # Green
        'bus': '#d62728'               # Red
    }

    # Plot key rate for each topology
    for topology in topologies:
        rates = []
        actual_distances = []
        for distance in distances:
            if distance in all_results[topology]:
                rates.append(all_results[topology][distance]["avg_rate"])
                actual_distances.append(all_results[topology][distance]["photon_travel_km"])
            else:
                rates.append(0)
                actual_distances.append(0)

        plt.plot(distances, rates, marker='o', linestyle='-', 
                 linewidth=2.5, markersize=10, 
                 color=topology_colors.get(topology, 'black'),
                 label=f'{topology.capitalize()} (Photon travel: {actual_distances[0]}km)')

    plt.xlabel('End-to-End Distance (km)', fontsize=14)
    plt.ylabel('Average Key Rate (bits/time unit)', fontsize=14)
    plt.title('Key Rate Comparison - Distributed Source Model\nPhotons travel only half the total distance', fontsize=16, fontweight='bold')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    output_path = os.path.join(comparison_dir, "key_rate_comparison.png")
    plt.savefig(output_path, dpi=300)
    plt.close()

    # QBER comparison
    plt.figure(figsize=(12, 8))

    # Plot QBER for each topology
    for topology in topologies:
        qbers = []
        for distance in distances:
            if distance in all_results[topology]:
                qbers.append(all_results[topology][distance]["avg_qber"])
            else:
                qbers.append(0)

        plt.plot(distances, qbers, marker='o', linestyle='-', 
                 linewidth=2.5, markersize=10, 
                 color=topology_colors.get(topology, 'black'),
                 label=f'{topology.capitalize()}')

    plt.xlabel('End-to-End Distance (km)', fontsize=14)
    plt.ylabel('Average QBER', fontsize=14)
    plt.title('QBER Comparison Across Topologies', fontsize=16)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    output_path = os.path.join(comparison_dir, "qber_comparison.png")
    plt.savefig(output_path, dpi=300)
    plt.close()

def run_multi_hop_simulation(topology, num_nodes, total_distance, num_hops,
                             detector_efficiency=0.9, loss_per_km=0.2, swap_success_prob=1.0,
                             simulation_duration=500, pair_generation_rate=5.0, seed=None, target_bits=256):
    logging.info(f"Starting multi-hop simulation: {topology} topology, {total_distance}km, {num_hops} hops")
    ns.sim_reset()
    if seed is not None:
        try: ns.set_random_state(int(seed))
        except Exception: pass
        try: random.seed(int(seed))
        except Exception: pass
        try: np.random.seed(int(seed) % (2**32 - 1))
        except Exception: pass

    total_required_nodes = num_hops + 2
    if num_nodes < total_required_nodes:
        num_nodes = total_required_nodes
        logging.info(f"Increasing number of nodes to {num_nodes} to accommodate {num_hops} hops")

    hop_distance = total_distance / (num_hops + 1)
    network, nodes, entanglement_connections = create_qkd_network(
        num_nodes, hop_distance, detector_efficiency, loss_per_km, topology
    )

    shared_keys = {}

    # Path selection
    if topology == "star":
        central_node = 0
        if num_hops == 0:
            end_nodes = [central_node, 1]
            path = [central_node, 1]
            swap_nodes = []
        else:
            end_nodes = [1, 2]
            path = [end_nodes[0], central_node, end_nodes[1]]
            swap_nodes = [central_node]
            num_hops = 1
    else:
        end_nodes = [0, num_hops + 1]
        path = list(range(end_nodes[0], end_nodes[1] + 1))
        swap_nodes = path[1:-1]

    logging.info(f"Selected path: {path}, end nodes: {end_nodes}, swap nodes: {swap_nodes}")

    
    pair_to_source = {}
    for source_node, conns in entanglement_connections.items():
        if conns:
            i, j = conns[0]
            pair_to_source[(min(i, j), max(i, j))] = source_node

    path_edges = {(min(a, b), max(a, b)) for a, b in zip(path[:-1], path[1:])}
    node_connections = {i: [] for i in range(num_nodes)}
    for idx, node_id in enumerate(path):
        if idx > 0: node_connections[node_id].append(path[idx - 1])
        if idx < len(path) - 1: node_connections[node_id].append(path[idx + 1])

    protocols = []

    # Sources on path edges
    for a, b in path_edges:
        source_node = pair_to_source.get((a, b))
        if source_node is None:
            logging.warning(f"No entanglement source found for edge {(a, b)}")
            continue
        protocols.append(EntanglementSourceProtocol(
            source_node, [(a, b)], pair_generation_rate=pair_generation_rate, downstream_ready_ports=None
        ))

    # Endpoints
    end_a, end_b = end_nodes
    for node_id in end_nodes:
        partner = end_b if node_id == end_a else end_a
        protocols.append(NodeMeasurementProtocol(
            nodes[node_id], node_id, shared_keys, [partner],
            enable_swapping=(num_hops > 0),
            expected_depth=(num_hops if num_hops > 0 else None),
            upstream_ready_port=None
        ))

    # Swapper protocols
    if num_hops > 0:
        for node_id in swap_nodes:
            protocols.append(EntanglementSwappingProtocol(
                nodes[node_id], node_id, shared_keys, node_connections[node_id],
                swap_success_prob=swap_success_prob, path=path, num_hops=num_hops, upstream_ready_ports={}
            ))

    # Ensure end-to-end classical connectivity between endpoints
    try:
        # add missing ports
        for a, b in [(end_a, end_b), (end_b, end_a)]:
            if f"cout_{b}" not in nodes[a].ports: nodes[a].add_ports([f"cout_{b}"])
            if f"cin_{b}" not in nodes[a].ports: nodes[a].add_ports([f"cin_{b}"])
        # channels
        c_ab = create_classical_channel(total_distance)
        c_ba = create_classical_channel(total_distance)
        conn_ab = DirectConnection(f"c_conn_{end_a}_{end_b}_e2e", channel_AtoB=c_ab)
        conn_ba = DirectConnection(f"c_conn_{end_b}_{end_a}_e2e", channel_AtoB=c_ba)
        network.add_connection(nodes[end_a], nodes[end_b], connection=conn_ab, label="classical",
                               port_name_node1=f"cout_{end_b}", port_name_node2=f"cin_{end_a}")
        network.add_connection(nodes[end_b], nodes[end_a], connection=conn_ba, label="classical",
                               port_name_node1=f"cout_{end_a}", port_name_node2=f"cin_{end_b}")
    except Exception as e:
        logging.warning(f"Failed to add end-to-end classical link: {e}")

    for p in protocols:
        p.start()

    start_time = ns.sim_time()
    chunk_size = 50
    num_chunks = max(1, simulation_duration // chunk_size)
    e2e_key_id = f"{min(end_a, end_b)}-{max(end_a, end_b)}"
    for i in range(num_chunks):
        try:
            ns.sim_run(duration=chunk_size)
        except KeyboardInterrupt:
            logging.error("Multi-hop simulation interrupted by user.")
            break
        if target_bits and e2e_key_id in shared_keys and len(shared_keys[e2e_key_id]) >= target_bits:
            logging.info(f"Target bits {target_bits} reached, stopping early.")
            break

    elapsed = max(ns.sim_time() - start_time, 1e-9)
    key_len = len(shared_keys.get(e2e_key_id, []))
    key_rate = key_len / elapsed if elapsed > 0 else 0.0

    return {
        "topology": topology,
        "num_nodes": num_nodes,
        "num_hops": num_hops,
        "total_distance": total_distance,
        "hop_distance": hop_distance,
        "end_nodes": end_nodes,
        "swap_nodes": swap_nodes,
        "end_to_end_key_rate": key_rate,
        "end_to_end_key_length": key_len,
        "elapsed_time": elapsed,
        "simulation_duration": simulation_duration,
        "swap_success_prob": swap_success_prob
    }

def run_multi_hop_distance_sweep(topology, num_nodes, distances, num_hops_list,
                                 detector_efficiency=0.9, loss_per_km=0.2,
                                 swap_success_prob=1.0, simulation_duration=500, pair_generation_rate=5.0,
                                 num_trials=2, target_bits=256):
    results = {h: {} for h in num_hops_list}
    for distance in distances:
        for hops in num_hops_list:
            logging.info(f"\n===== {topology} | {distance} km | {hops} hops =====")
            trial_rates, trial_bits = [], []
            last = None
            for t in range(num_trials):
                seed = (hash(topology) & 0xFFFF) * 10**6 + hops * 10**4 + int(distance * 10) + t
                res = run_multi_hop_simulation(
                    topology=topology, num_nodes=num_nodes, total_distance=distance, num_hops=hops,
                    detector_efficiency=detector_efficiency, loss_per_km=loss_per_km,
                    swap_success_prob=swap_success_prob, simulation_duration=simulation_duration,
                    pair_generation_rate=pair_generation_rate, seed=seed, target_bits=target_bits
                )
                last = res
                trial_rates.append(res.get("end_to_end_key_rate", 0.0))
                trial_bits.append(res.get("end_to_end_key_length", 0))
            rate_mean = float(np.mean(trial_rates)) if trial_rates else 0.0
            rate_std = float(np.std(trial_rates)) if len(trial_rates) > 1 else 0.0
            bits_mean = float(np.mean(trial_bits)) if trial_bits else 0.0
            averaged = dict(last or {})
            averaged["end_to_end_key_rate"] = rate_mean
            averaged["end_to_end_key_rate_mean"] = rate_mean
            averaged["end_to_end_key_rate_std"] = rate_std
            averaged["end_to_end_key_length_avg"] = bits_mean
            averaged["num_trials"] = num_trials
            results[hops][distance] = averaged
    return results

# Plot key rate vs distance for each topology and hop count, plus best-of curve
def plot_multi_hop_comparison(results_by_topology, distances, num_hops_list, output_dir=OUTPUT_DIR):
    comp_dir = os.path.join(output_dir, "multi_hop_comparison")
    os.makedirs(comp_dir, exist_ok=True)
    hop_colors = {0: '#17becf', 1: '#1f77b4', 2: '#ff7f0e', 3: '#2ca02c', 4: '#d62728', 5: '#9467bd', 6: '#8c564b'}
    topology_markers = {'fully_connected': 'o', 'star': 's', 'ring': '^', 'bus': 'D'}

    # Per-topology plots
    for topology, res in results_by_topology.items():
        plt.figure(figsize=(8, 6))
        for h in num_hops_list:
            if h in res:
                d_with = sorted([d for d in distances if d in res[h]])
                rates = [res[h][d].get("end_to_end_key_rate", 0.0) for d in d_with]
                plt.plot(d_with, rates, marker=topology_markers.get(topology, 'o'), linestyle='-',
                         linewidth=1.5, markersize=6, color=hop_colors.get(h, 'black'),
                         label=f'{h} hop{"s" if h != 1 else ""}')
        plt.xlabel('Total End-to-End Distance (km)', fontsize=11)
        plt.ylabel('Key Exchange Rate (bits/time unit)', fontsize=11)
        plt.title(f'Key Exchange Rate vs Distance - {topology.capitalize()} (Multi-hop)', fontsize=16, fontweight='bold')
        plt.legend(fontsize=10, loc='upper right')
        plt.grid(True)
        plt.xticks(fontsize=9); plt.yticks(fontsize=9); plt.tight_layout()
        out = os.path.join(comp_dir, f"{topology}_multi_hop_comparison.png")
        plt.savefig(out, dpi=300); plt.close()

    # Best-of across hops per topology
    plt.figure(figsize=(8, 6))
    for topology, res in results_by_topology.items():
        best = []
        for d in sorted(distances):
            best_rate = 0.0
            for h in num_hops_list:
                if h in res and d in res[h]:
                    r = res[h][d].get("end_to_end_key_rate", 0.0)
                    if r > best_rate:
                        best_rate = r
            best.append(best_rate)
        plt.plot(sorted(distances), best, marker=topology_markers.get(topology, 'o'), linestyle='-',
                 linewidth=1.5, markersize=6, label=topology.capitalize())
    plt.xlabel('Total End-to-End Distance (km)', fontsize=11)
    plt.ylabel('Best Key Exchange Rate (bits/time unit)', fontsize=11)
    plt.title('Best Key Exchange Rate vs Distance (Optimal Hops)', fontsize=16, fontweight='bold')
    plt.legend(fontsize=10, loc='upper right')
    plt.grid(True)
    plt.xticks(fontsize=9); plt.yticks(fontsize=9); plt.tight_layout()
    out = os.path.join(comp_dir, "topology_comparison_best_rates.png")
    plt.savefig(out, dpi=300); plt.close()

# Plot average end-to-end rate vs number of hops per topology
def plot_key_rate_vs_hops(results_by_topology, distances, num_hops_list, output_dir=OUTPUT_DIR):
    plt.figure(figsize=(8, 6))
    topology_colors = {'fully_connected': '#1f77b4', 'star': '#ff7f0e', 'ring': '#2ca02c', 'bus': '#d62728'}
    for topology, res in results_by_topology.items():
        avgs = []
        for h in num_hops_list:
            if h in res:
                rates = [res[h][d].get("end_to_end_key_rate", 0.0) for d in distances if d in res[h]]
                avgs.append(sum(rates) / len(rates) if rates else 0.0)
            else:
                avgs.append(0.0)
        plt.plot(num_hops_list, avgs, marker='o', linestyle='-', linewidth=1.5,
                 color=topology_colors.get(topology, 'black'), label=topology.capitalize(), markersize=6)
    plt.xlabel('Number of Entanglement Swapping Nodes (Hops)', fontsize=11)
    plt.ylabel('Average End-to-End Key Exchange Rate (bits/time unit)', fontsize=11)
    plt.title('Key Exchange Rate vs Number of Swapping Nodes', fontsize=16, fontweight='bold')
    plt.legend(fontsize=10, loc='upper right')
    plt.grid(True)
    plt.xticks(fontsize=9); plt.yticks(fontsize=9); plt.tight_layout()
    out = os.path.join(output_dir, "multi_hop_comparison/key_rate_vs_hops.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=300); plt.close()

# Multi-hop analysis runner (topologies, hops, plots)
def run_multi_hop_analysis():
    logging.info("\n===== Starting Multi-Hop Entanglement Swapping Analysis =====")
    # Parameters
    num_nodes = 8
    detector_efficiency = 0.9
    loss_per_km = 0.2
    simulation_duration = 1200
    distances = list(range(50, 1001, 50))  # 50 km steps from 50 to 1000 km
    topologies = ["star", "ring", "bus"]
    hops_by_topology = {
        "star": [0, 1],         # center-leaf and leaf-leaf via center
        "ring": [0, 1, 2, 3],   # more hops for ring
        "bus": [0, 1, 2, 3, 4]  
    }

    all_results = {}
    for topology in topologies:
        num_hops_list = hops_by_topology[topology]
        logging.info(f"\n===== Running multi-hop sweep for {topology} =====")
        res = run_multi_hop_distance_sweep(
            topology=topology,
            num_nodes=num_nodes,
            distances=distances,
            num_hops_list=num_hops_list,
            detector_efficiency=detector_efficiency,
            loss_per_km=loss_per_km,
            swap_success_prob=1.0,
            simulation_duration=simulation_duration,
            pair_generation_rate=10.0,
            num_trials=2,
            target_bits=256
        )
        all_results[topology] = res

    # Plots
    flat_hops = sorted(set(h for v in hops_by_topology.values() for h in v))
    plot_multi_hop_comparison(all_results, distances, flat_hops, output_dir=OUTPUT_DIR)
    plot_key_rate_vs_hops(all_results, distances, flat_hops, output_dir=OUTPUT_DIR)

    # Summary
    logging.info("\n===== MULTI-HOP RESULTS SUMMARY =====")
    for topology, res in all_results.items():
        logging.info(f"\n{topology.upper()} TOPOLOGY:")
        for h in hops_by_topology[topology]:
            rates = [res[h][d]["end_to_end_key_rate"] for d in distances if d in res[h]]
            if rates:
                logging.info(f"  {h} hop{'s' if h != 1 else ''}: avg rate = {sum(rates)/len(rates):.6f} bits/unit")

    logging.info("===== Multi-Hop analysis complete =====")

def main():
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Basic parameters
    num_nodes = 4  # Number of regular nodes (not including source)
    detector_efficiency = 0.9
    loss_per_km = 0.2
    simulation_duration = 500  # Reduced for quicker testing

    # Distances to simulate (km)
    distances = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]  # Reduced for quicker testing

    # Available topologies
    topologies = ["star", "bus", "ring"]  # Reduced for quicker testing

    # CLI to control multi-hop explicitly
    parser = argparse.ArgumentParser(description="EB-QKD simulations")
    parser.add_argument("--multihop", dest="multihop", action="store_true", help="Run multi-hop analysis")
    parser.add_argument("--no-multihop", dest="multihop", action="store_false", help="Skip multi-hop analysis")
    parser.set_defaults(multihop=None)
    args, _ = parser.parse_known_args()

    env_val = os.environ.get("EB_RUN_MULTI_HOP")
    if args.multihop is not None:
        run_multihop = args.multihop
        src = "CLI"
    elif env_val is not None:
        run_multihop = (env_val == "1")
        src = f"env EB_RUN_MULTI_HOP={env_val}"
    else:
        run_multihop = True
        src = "default"
    logging.info(f"Multi-hop setting: {run_multihop} (source: {src})")

    # Visualize each topology
    for topology in topologies:
        logging.info(f"\n===== Visualizing {topology} topology =====")
        network, nodes, _ = create_qkd_network(num_nodes, 20, detector_efficiency, loss_per_km, topology)
        visualize_network(network, topology=topology)

    # Run distance sweep for each topology
    all_results = {}
    for topology in topologies:
        logging.info(f"\n===== Running distance sweep for {topology} topology =====")
        results = run_distance_sweep(
            num_nodes, 
            distances, 
            detector_efficiency, 
            loss_per_km, 
            topology, 
            simulation_duration
        )
        all_results[topology] = results
        logging.info(f"Completed distance sweep for {topology} topology.")

    # Create comparison plots
    create_topology_comparison_plots(all_results, distances, topologies)
    logging.info("Created topology comparison plots.")

    # Add the P2P distance sweep
    logging.info("\n===== Running P2P EB-QKD with source in middle =====")
    p2p_results = run_p2p_distance_sweep()
    create_p2p_plot(p2p_results)
    logging.info("Created P2P plots.")

    # Print summary
    logging.info("\n===== RESULTS SUMMARY =====")
    for topology in topologies:
        logging.info(f"\n{topology.upper()} TOPOLOGY:")
        for distance, result in all_results[topology].items():
            logging.info(f"  Distance: {distance}km, Rate: {result['avg_rate']:.4f} bits/time unit, Total bits: {result['total_bits']}, QBER: {result['avg_qber']:.4f}")

    # Run full multi-hop analysis and plots
    if run_multihop:
        run_multi_hop_analysis()
    else:
        logging.info("Skipping multi-hop analysis (pass --multihop or set EB_RUN_MULTI_HOP=1 to enable).")

    logging.info("All EB simulations complete.")

if __name__ == "__main__":
    main()