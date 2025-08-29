from netsquid.components import QuantumChannel, ClassicalChannel
from netsquid.nodes import Node, Network, DirectConnection
from netsquid.components.models.qerrormodels import FibreLossModel
import logging

def create_quantum_channel(length_km, loss_per_km):
    return QuantumChannel(
        "QuantumChannel",
        length=length_km,
        models={"quantum_loss_model": FibreLossModel(p_loss_init=0, p_loss_length=loss_per_km)}
    )

def create_classical_channel(length_km):
    return ClassicalChannel("ClassicalChannel", length=length_km)

def create_network_connections(network, nodes, channel_length_km, loss_per_km, topology):
    if topology == "fully_connected":
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                q_channel_ij = create_quantum_channel(channel_length_km, loss_per_km)
                q_conn_ij = DirectConnection(f"quantum_connection_{i}_{j}", channel_AtoB=q_channel_ij)
                network.add_connection(
                    nodes[i], nodes[j],
                    connection=q_conn_ij,
                    label="quantum",
                    port_name_node1=f"qout_{j}",
                    port_name_node2=f"qin_{i}"
                )
                q_channel_ji = create_quantum_channel(channel_length_km, loss_per_km)
                q_conn_ji = DirectConnection(f"quantum_connection_{j}_{i}", channel_AtoB=q_channel_ji)
                network.add_connection(
                    nodes[j], nodes[i],
                    connection=q_conn_ji,
                    label="quantum",
                    port_name_node1=f"qout_{i}",
                    port_name_node2=f"qin_{j}"
                )
                c_channel_ij = create_classical_channel(channel_length_km)
                c_conn_ij = DirectConnection(f"classical_connection_{i}_{j}", channel_AtoB=c_channel_ij)
                network.add_connection(
                    nodes[i], nodes[j],
                    connection=c_conn_ij,
                    label="classical",
                    port_name_node1=f"cout_{j}",
                    port_name_node2=f"cin_{i}"
                )
                c_channel_ji = create_classical_channel(channel_length_km)
                c_conn_ji = DirectConnection(f"classical_connection_{j}_{i}", channel_AtoB=c_channel_ji)
                network.add_connection(
                    nodes[j], nodes[i],
                    connection=c_conn_ji,
                    label="classical",
                    port_name_node1=f"cout_{i}",
                    port_name_node2=f"cin_{j}"
                )

    elif topology == "star":
        central_node = nodes[0]
        for i in range(1, len(nodes)):
            q_channel_0i = create_quantum_channel(channel_length_km, loss_per_km)
            q_conn_0i = DirectConnection(f"quantum_connection_0_{i}", channel_AtoB=q_channel_0i)
            network.add_connection(
                central_node, nodes[i],
                connection=q_conn_0i,
                label="quantum",
                port_name_node1=f"qout_{i}",
                port_name_node2=f"qin_0"
            )
            q_channel_i0 = create_quantum_channel(channel_length_km, loss_per_km)
            q_conn_i0 = DirectConnection(f"quantum_connection_{i}_0", channel_AtoB=q_channel_i0)
            network.add_connection(
                nodes[i], central_node,
                connection=q_conn_i0,
                label="quantum",
                port_name_node1=f"qout_0",
                port_name_node2=f"qin_{i}"
            )
            c_channel_0i = create_classical_channel(channel_length_km)
            c_conn_0i = DirectConnection(f"classical_connection_0_{i}", channel_AtoB=c_channel_0i)
            network.add_connection(
                central_node, nodes[i],
                connection=c_conn_0i,
                label="classical",
                port_name_node1=f"cout_{i}",
                port_name_node2=f"cin_0"
            )
            c_channel_i0 = create_classical_channel(channel_length_km)
            c_conn_i0 = DirectConnection(f"classical_connection_{i}_0", channel_AtoB=c_channel_i0)
            network.add_connection(
                nodes[i], central_node,
                connection=c_conn_i0,
                label="classical",
                port_name_node1=f"cout_0",
                port_name_node2=f"cin_{i}"
            )

    elif topology == "ring":
        for i in range(len(nodes)):
            j = (i + 1) % len(nodes)
            q_channel_ij = create_quantum_channel(channel_length_km, loss_per_km)
            q_conn_ij = DirectConnection(f"quantum_connection_{i}_{j}", channel_AtoB=q_channel_ij)
            network.add_connection(
                nodes[i], nodes[j],
                connection=q_conn_ij,
                label="quantum",
                port_name_node1=f"qout_{j}",
                port_name_node2=f"qin_{i}"
            )
            q_channel_ji = create_quantum_channel(channel_length_km, loss_per_km)
            q_conn_ji = DirectConnection(f"quantum_connection_{j}_{i}", channel_AtoB=q_channel_ji)
            network.add_connection(
                nodes[j], nodes[i],
                connection=q_conn_ji,
                label="quantum",
                port_name_node1=f"qout_{i}",
                port_name_node2=f"qin_{j}"
            )
            c_channel_ij = create_classical_channel(channel_length_km)
            c_conn_ij = DirectConnection(f"classical_connection_{i}_{j}", channel_AtoB=c_channel_ij)
            network.add_connection(
                nodes[i], nodes[j],
                connection=c_conn_ij,
                label="classical",
                port_name_node1=f"cout_{j}",
                port_name_node2=f"cin_{i}"
            )
            c_channel_ji = create_classical_channel(channel_length_km)
            c_conn_ji = DirectConnection(f"classical_connection_{j}_{i}", channel_AtoB=c_channel_ji)
            network.add_connection(
                nodes[j], nodes[i],
                connection=c_conn_ji,
                label="classical",
                port_name_node1=f"cout_{i}",
                port_name_node2=f"cin_{j}"
            )

    elif topology == "bus":
        for i in range(len(nodes) - 1):
            j = i + 1
            q_channel_ij = create_quantum_channel(channel_length_km, loss_per_km)
            q_conn_ij = DirectConnection(f"quantum_connection_{i}_{j}", channel_AtoB=q_channel_ij)
            network.add_connection(
                nodes[i], nodes[j],
                connection=q_conn_ij,
                label="quantum",
                port_name_node1=f"qout_{j}",
                port_name_node2=f"qin_{i}"
            )
            q_channel_ji = create_quantum_channel(channel_length_km, loss_per_km)
            q_conn_ji = DirectConnection(f"quantum_connection_{j}_{i}", channel_AtoB=q_channel_ji)
            network.add_connection(
                nodes[j], nodes[i],
                connection=q_conn_ji,
                label="quantum",
                port_name_node1=f"qout_{i}",
                port_name_node2=f"qin_{j}"
            )
            c_channel_ij = create_classical_channel(channel_length_km)
            c_conn_ij = DirectConnection(f"classical_connection_{i}_{j}", channel_AtoB=c_channel_ij)
            network.add_connection(
                nodes[i], nodes[j],
                connection=c_conn_ij,
                label="classical",
                port_name_node1=f"cout_{j}",
                port_name_node2=f"cin_{i}"
            )
            c_channel_ji = create_classical_channel(channel_length_km)
            c_conn_ji = DirectConnection(f"classical_connection_{j}_{i}", channel_AtoB=c_channel_ji)
            network.add_connection(
                nodes[j], nodes[i],
                connection=c_conn_ji,
                label="classical",
                port_name_node1=f"cout_{i}",
                port_name_node2=f"cin_{j}"
            )

def create_qkd_network(num_nodes, channel_length_km, detector_efficiency, loss_per_km, topology):
    network = Network("QKD Network")
    nodes = [Node(f"Node_{i}") for i in range(num_nodes)]
    network.add_nodes(nodes)
    for i in range(num_nodes):
        nodes[i].add_ports([f"qout_{j}" for j in range(num_nodes)])
        nodes[i].add_ports([f"qin_{j}" for j in range(num_nodes)])
        nodes[i].add_ports([f"cout_{j}" for j in range(num_nodes)])
        nodes[i].add_ports([f"cin_{j}" for j in range(num_nodes)])
    create_network_connections(network, nodes, channel_length_km, loss_per_km, topology)
    return network, nodes

def calculate_ring_path(source, target, num_nodes):
    clockwise_distance = (target - source) % num_nodes
    counter_clockwise_distance = (source - target) % num_nodes
    if clockwise_distance <= counter_clockwise_distance:
        path = [(source + i) % num_nodes for i in range(clockwise_distance + 1)]
    else:
        path = [(source - i) % num_nodes for i in range(counter_clockwise_distance + 1)]
    return path

def calculate_path(source, target, num_nodes, topology):
    if topology == "ring":
        return calculate_ring_path(source, target, num_nodes)
    elif topology == "star":
        if source == 0 or target == 0:
            return [source, target]
        else:
            return [source, 0, target]
    elif topology == "bus":
        if source <= target:
            return list(range(source, target + 1))
        else:
            return list(range(source, target - 1, -1))
    elif topology == "fully_connected":
        return [source, target]
    else:
        logging.error(f"Unknown topology: {topology}")
        return [source, target]