import netsquid as ns
import random
import logging
from netsquid.qubits import qubitapi as qapi
from netsquid.qubits.ketstates import s0, s1, h0, h1
from netsquid.qubits.operators import Z, X, H
from netsquid.protocols import NodeProtocol

class TransmitterProtocol(NodeProtocol):
    def __init__(self, node, final_key, sent_bits, target_node_index, target_length=None,
                 sent_bases=None, pulse_ids=None):
        super().__init__(node)
        self.final_key = final_key
        self.sent_bits = sent_bits
        self.sent_bases = sent_bases if sent_bases is not None else []
        self.pulse_ids = pulse_ids if pulse_ids is not None else []
        self.target_node_index = target_node_index
        self.target_length = target_length  # Optional target key length
        self.completed = False
        self._pulse_counter = 0

    def run(self):
        while not self.completed:
            # Generate a random bit and basis
            bit = random.randint(0, 1)
            basis = random.choice(['Z', 'X'])
            pulse_id = self._pulse_counter
            self._pulse_counter += 1
            logging.debug(f"Transmitter {self.node.name} generated bit: {bit} in basis: {basis}, pid={pulse_id}")

            # Prepare a qubit based on the random bit and basis
            qubit, = qapi.create_qubits(1)
            if basis == 'Z':
                qapi.assign_qstate(qubit, s0 if bit == 0 else s1)
            else:
                qapi.assign_qstate(qubit, h0 if bit == 0 else h1)
            
            # Send the qubit over the quantum channel
            self.node.ports[f'qout_{self.target_node_index}'].tx_output(qubit)
            logging.debug(f"Qubit sent from {self.node.name} to Node_{self.target_node_index} (pid={pulse_id})")
            
            # Send only the basis and pulse identifier over the classical channel
            message = {'basis': basis, 'pulse_id': pulse_id}
            self.node.ports[f'cout_{self.target_node_index}'].tx_output(message)
            logging.debug(f"Basis sent from {self.node.name} to Node_{self.target_node_index} (pid={pulse_id})")
            
            # Log TX info
            self.sent_bits.append(bit)
            self.sent_bases.append(basis)
            self.pulse_ids.append(pulse_id)
            
            # Check if we've reached the target key length
            if self.target_length and len(self.final_key) >= self.target_length:
                logging.info(f"Transmitter {self.node.name} reached target key length of {self.target_length}")
                self.completed = True
                break
                
            yield self.await_timer(1)

class ReceiverProtocol(NodeProtocol):
    def __init__(self, node, final_key, sent_bits, measured_bits, source_node_index, target_length=None,
                 detection_efficiency=1.0, recv_bases=None, recv_pulse_ids=None):
        super().__init__(node)
        self.final_key = final_key
        self.sent_bits = sent_bits
        self.measured_bits = measured_bits
        self.recv_bases = recv_bases if recv_bases is not None else []
        self.recv_pulse_ids = recv_pulse_ids if recv_pulse_ids is not None else []
        self.source_node_index = source_node_index
        self.target_length = target_length  # Optional target key length
        self.detection_efficiency = detection_efficiency
        self.completed = False

    def run(self):
        while not self.completed:
            logging.debug(f"Waiting for qubit at {self.node.name} qin_{self.source_node_index}")
            yield self.await_port_input(self.node.ports[f'qin_{self.source_node_index}'])
            port_input = self.node.ports[f'qin_{self.source_node_index}'].rx_input()
            if port_input is None or len(port_input.items) == 0:
                logging.debug(f"No qubit received at qin_{self.source_node_index}")
                # Still wait for/classically receive basis to avoid backlog
                yield self.await_port_input(self.node.ports[f'cin_{self.source_node_index}'])
                _ = self.node.ports[f'cin_{self.source_node_index}'].rx_input()
                continue
            qubit = port_input.items[0]
            logging.debug(f"Qubit received at {self.node.name} qin_{self.source_node_index}")

            # Simple detection efficiency model
            if random.random() > self.detection_efficiency:
                logging.debug(f"Detection failed due to inefficiency at {self.node.name}")
                # Still receive the basis message to keep sync, but do not record measurement
                yield self.await_port_input(self.node.ports[f'cin_{self.source_node_index}'])
                _ = self.node.ports[f'cin_{self.source_node_index}'].rx_input()
                continue

            # Choose a random basis for measurement
            basis = random.choice(['Z', 'X'])
            logging.debug(f"Receiver {self.node.name} chose basis: {basis}")

            # Measure the qubit in the chosen basis
            if basis == 'Z':
                measurement, _ = qapi.measure(qubit, observable=Z)
            else:
                measurement, _ = qapi.measure(qubit, observable=X)
            logging.debug(f"Qubit received and measured: {measurement}")

            # Receive the message over the classical channel
            logging.debug(f"Waiting for message at {self.node.name} cin_{self.source_node_index}")
            yield self.await_port_input(self.node.ports[f'cin_{self.source_node_index}'])
            message = self.node.ports[f'cin_{self.source_node_index}'].rx_input().items[0]
            sent_basis = message['basis']
            pulse_id = message.get('pulse_id')
            logging.debug(f"Received basis over classical channel (pid={pulse_id}, basis={sent_basis})")

            # Log RX info
            self.measured_bits.append(measurement)
            self.recv_bases.append(basis)
            self.recv_pulse_ids.append(pulse_id)

            # Sifting: keep the bit if the bases match
            if basis == sent_basis:
                self.final_key.append(measurement)
                logging.debug(f"Bases match. Measurement result: {measurement} added to key")
                
                # Check if we've reached the target key length
                if self.target_length and len(self.final_key) >= self.target_length:
                    logging.info(f"Receiver {self.node.name} reached target key length of {self.target_length}")
                    self.completed = True
                    break
            else:
                logging.debug("Bases do not match. Discarding measurement.")

class RelayProtocol(NodeProtocol):
    def __init__(self, node, final_key, source_node_index, target_node_index, is_last=False, target_length=50):
        super().__init__(node)
        self.final_key = final_key  # This is the hop's key that will be used in XORing
        self.source_node_index = source_node_index
        self.target_node_index = target_node_index
        self.is_last = is_last
        self.target_length = target_length
        self.completed = False
        
        # Initialize keys for both links with pre-generated random values
        self.source_key = []
        self.target_key = []
        
        # Pre-generate target key bits (to help speed up simulation)
        self._pre_generate_target_key()
        
        # Create and add subprotocols during initialization
        source_handler = SourceLinkHandler(self.node, self.source_key, self.source_node_index, self.target_length)
        target_handler = TargetLinkHandler(self.node, self.target_key, self.target_node_index, self.target_length, 
                                          self._pre_generated_target_bits)
        
        self.add_subprotocol(source_handler, "source_handler")
        self.add_subprotocol(target_handler, "target_handler")
    
    def _pre_generate_target_key(self):
        """Pre-generate random bits for the target link to speed up simulation"""
        self._pre_generated_target_bits = [random.randint(0, 1) for _ in range(self.target_length * 2)]
        
    def run(self):
        # Start the subprotocols
        self.start_subprotocols()
        
        # Wait for both source and target keys to reach target length
        check_interval = 2  # Check progress every 2 time units
        while not self.completed:
            # Both keys must reach target length
            if len(self.source_key) >= self.target_length and len(self.target_key) >= self.target_length:
                logging.info(f"Relay {self.node.name} has completed both keys: Source: {len(self.source_key)}, Target: {len(self.target_key)}")
                
                # Clear the final key and XOR the keys to create a new one of target length
                self.final_key.clear()
                
                # XOR the keys and add to final key
                for i in range(self.target_length):
                    xor_bit = self.source_key[i] ^ self.target_key[i]
                    self.final_key.append(xor_bit)
                
                logging.info(f"Relay {self.node.name} completed XORed key of length {len(self.final_key)}")
                self.completed = True
                break
            
            # Log progress periodically
            if ns.sim_time() % 20 == 0:  # Every 20 time units
                logging.debug(f"Relay {self.node.name} progress: Source key: {len(self.source_key)}/{self.target_length}, Target key: {len(self.target_key)}/{self.target_length}")
                
            yield self.await_timer(check_interval)

class SourceLinkHandler(NodeProtocol):
    def __init__(self, node, source_key, source_node_index, target_length):
        super().__init__(node)
        self.source_key = source_key
        self.source_node_index = source_node_index
        self.target_length = target_length
        
    def run(self):
        logging.info(f"Starting source link handler for {self.node.name} with source {self.source_node_index}")
        while len(self.source_key) < self.target_length:
            # Wait for a qubit from source node
            yield self.await_port_input(self.node.ports[f'qin_{self.source_node_index}'])
            port_input = self.node.ports[f'qin_{self.source_node_index}'].rx_input()
            
            if port_input is None or len(port_input.items) == 0:
                logging.warning(f"No qubit received from source at {self.node.name}")
                continue
                
            qubit = port_input.items[0]
            
            # Always measure in Z basis for trusted node protocol
            measurement, _ = qapi.measure(qubit, observable=Z)
            logging.debug(f"{self.node.name} received and measured qubit from source: {measurement}")
            
            # Wait for classical message
            yield self.await_port_input(self.node.ports[f'cin_{self.source_node_index}'])
            message = self.node.ports[f'cin_{self.source_node_index}'].rx_input().items[0]
            
            # In trusted node operation, we always accept the measurement
            self.source_key.append(measurement)
            if len(self.source_key) % 10 == 0:
                logging.debug(f"{self.node.name} source key progress: {len(self.source_key)}/{self.target_length}")

class TargetLinkHandler(NodeProtocol):
    def __init__(self, node, target_key, target_node_index, target_length, pre_generated_bits=None):
        super().__init__(node)
        self.target_key = target_key
        self.target_node_index = target_node_index
        self.target_length = target_length
        self.pre_generated_bits = pre_generated_bits or []
        self.bit_index = 0
        
    def run(self):
        logging.info(f"Starting target link handler for {self.node.name} with target {self.target_node_index}")
        
        # Generate keys much faster (every 0.5 time units) to ensure they're ready for XORing
        while len(self.target_key) < self.target_length:
            # Use pre-generated bit if available, otherwise generate a new one
            if self.bit_index < len(self.pre_generated_bits):
                bit = self.pre_generated_bits[self.bit_index]
                self.bit_index += 1
            else:
                bit = random.randint(0, 1)
            
            # Prepare qubit in Z basis for maximum efficiency
            qubit, = qapi.create_qubits(1)
            if bit == 1:
                qapi.operate(qubit, X)
                
            # Send the qubit to target
            self.node.ports[f'qout_{self.target_node_index}'].tx_output(qubit)
            
            # Send classical message with basis
            message = {'basis': 'Z', 'original_bit': bit}
            self.node.ports[f'cout_{self.target_node_index}'].tx_output(message)
            
            # Add bit to target key
            self.target_key.append(bit)
            
            if len(self.target_key) % 10 == 0:
                logging.debug(f"{self.node.name} target key progress: {len(self.target_key)}/{self.target_length}")
            
            # Generate keys faster than the source side to ensure we're not the bottleneck
            yield self.await_timer(0.5)